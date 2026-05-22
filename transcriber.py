"""
Module for GPU-accelerated audio transcription (complete pipeline).

Pipeline (all stages on CUDA):
    A. Diarization  - pyannote.audio speaker-diarization-3.1 (float16)
    B. ASR          - faster-whisper large-v3 (int8_float16, ~1.6 GB VRAM)

Output format:
    [00:12] Speaker 1: Conversation text...
    [00:15] Speaker 2: Reply in local dialect...

VRAM Consumption (approximate):
    pyannote diarization  ~1.5 GB  (float16, unloaded after step A)
    faster-whisper ASR    ~1.6 GB  (int8_float16, stays in memory)
    ──────────────────────────────
    Peak (A+B concurrent) ~3.1 GB
    Together with Qwen (~5 GB) it stays well within the 11 GB VRAM limit.

Requirements:
    pip install pyannote.audio faster-whisper soundfile torch torchaudio
    Hugging Face token (for pyannote.audio):
        export HF_TOKEN=hf_...   or pass via argument hf_token=
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import sys
import types
import torch
import torchaudio

# МАВПЯЧИЙ ПАТЧ (Monkey Patch) ДЛЯ PYANNOTE ТА SPEECHBRAIN
# 1. Mock k2 module to prevent SpeechBrain lazy loader errors
if "k2" not in sys.modules:
    sys.modules["k2"] = types.ModuleType("k2")

# 2. Patch SpeechBrain LazyModule path check to support Windows backslashes
try:
    import speechbrain.utils.importutils as sb_importutils
    _orig_ensure_module = sb_importutils.LazyModule.ensure_module
    def _patched_ensure_module(self, stacklevel: int):
        import inspect, sys
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            pass
        if importer_frame is not None and (
            importer_frame.filename.endswith("/inspect.py") or
            importer_frame.filename.endswith("\\inspect.py")
        ):
            raise AttributeError()
        return _orig_ensure_module(self, stacklevel)
    sb_importutils.LazyModule.ensure_module = _patched_ensure_module
except Exception:
    pass

# 3. torchaudio patches
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

if "torchaudio.backend" not in sys.modules:
    sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
if "torchaudio.backend.common" not in sys.modules:
    tbc = types.ModuleType("torchaudio.backend.common")
    class AudioMetaData: pass
    tbc.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend.common"] = tbc

# 4. Patch torchaudio.info using torchcodec (since it's missing in modern torchaudio)
if not hasattr(torchaudio, "info"):
    def _patched_info(filepath, format=None):
        from torchcodec.decoders import AudioDecoder
        from torchaudio.backend.common import AudioMetaData
        decoder = AudioDecoder(filepath)
        meta = decoder.metadata
        info_obj = AudioMetaData()
        info_obj.sample_rate = meta.sample_rate or 44100
        info_obj.num_channels = meta.num_channels or 2
        duration = meta.duration_seconds or 0.0
        info_obj.num_frames = int(duration * info_obj.sample_rate)
        info_obj.bits_per_sample = 16
        info_obj.encoding = "PCM_S"
        return info_obj
    torchaudio.info = _patched_info

# 5. Force weights_only=False in torch.load (pyannote sets it to True explicitly)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Helper Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """A single transcription segment tied to a specific speaker."""
    start: float          # seconds
    end: float            # seconds
    speaker: str          # "Speaker 1", "Speaker 2", ...
    text: str             # recognized text

    def to_line(self) -> str:
        """Formats the segment as [MM:SS] Speaker N: text."""
        minutes = int(self.start) // 60
        seconds = int(self.start) % 60
        return f"[{minutes:02d}:{seconds:02d}] {self.speaker}: {self.text.strip()}"


# ──────────────────────────────────────────────────────────────────────────────
#  CUDA Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _free_vram(label: str = "") -> None:
    """Runs garbage collection and clears the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("VRAM cleared%s.", f" ({label})" if label else "")


def _log_vram(label: str = "") -> None:
    """Logs the current VRAM usage (only if CUDA is available)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    logger.info("VRAM %s - allocated: %.2f GB / reserved: %.2f GB", label, allocated, reserved)


# ──────────────────────────────────────────────────────────────────────────────
#  Step A: Speaker Diarization (pyannote.audio)
# ──────────────────────────────────────────────────────────────────────────────

def _diarize(
    wav_path: str,
    hf_token: str,
    num_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """
    Determines time segments for each speaker.

    Args:
        wav_path:     Path to the .wav file.
        hf_token:     Hugging Face token to download the pyannote model.
        num_speakers: Number of speakers (None = auto-detect).

    Returns:
        A list of tuples: (start_sec, end_sec, speaker_label).
    """
    logger.info("[Step B] Speaker diarization: %s", os.path.basename(wav_path))

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        logger.warning("pyannote.audio is not installed. Diarization disabled - all text will be assigned to 'Speaker 1'.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[Step B] pyannote device: %s", device)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    # float16 - reduces VRAM consumption by almost half (~1.5 GB instead of ~2.8 GB)
    pipeline = pipeline.to(device)

    # Set float16 for torch layers inside the pipeline
    if device.type == "cuda":
        for attr_name in dir(pipeline):
            try:
                attr = getattr(pipeline, attr_name)
                if isinstance(attr, torch.nn.Module):
                    attr.half()
            except Exception:
                pass

    _log_vram("during diarization")

    diarize_kwargs: dict = {}
    if num_speakers:
        diarize_kwargs["num_speakers"] = num_speakers

    diarization = pipeline(wav_path, **diarize_kwargs)

    segments: List[Tuple[float, float, str]] = []
    speaker_map: dict[str, str] = {}
    counter = 1

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {counter}"
            counter += 1
        segments.append((turn.start, turn.end, speaker_map[speaker]))

    logger.info("[Step B] Diarization completed: %d segments, %d speakers.", len(segments), counter - 1)

    # Unload pipeline - frees ~1.5 GB VRAM before ASR
    del pipeline
    _free_vram("after diarization")
    _log_vram("after diarization")

    return segments


# ──────────────────────────────────────────────────────────────────────────────
#  Step C: ASR (faster-whisper, large-v3, int8_float16)
# ──────────────────────────────────────────────────────────────────────────────

class _WhisperASR:
    """
    Wrapper for faster-whisper.

    The model is loaded once and stays in VRAM (~1.6 GB).
    compute_type="int8_float16" - INT8 quantization of activations + FP16 weights:
        reduces VRAM from ~3.1 GB (float16) to ~1.6 GB without noticeable quality loss.
    """

    _SUPPORTED_MODELS = ("large-v3", "large-v3-turbo", "large-v2")

    def __init__(self, model_size: str = "large-v3") -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"

        logger.info(
            "[Step C] Loading faster-whisper '%s' (device=%s, compute_type=%s)...",
            model_size, device, compute_type,
        )

        # Limit CPU threads (beam search) to save RAM
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,          # minimize CPU RAM usage
            cpu_threads=4,          # limit threads for beam search
            download_root=os.path.join(os.path.dirname(__file__), "models", "whisper"),
        )
        _log_vram("after loading Whisper")
        logger.info("[Step C] faster-whisper ready.")

    def transcribe_segment(
        self,
        wav_path: str,
        start: float,
        end: float,
        language: str = "uk",
        beam_size: int = 5,
        max_context_tokens: int = 224,
    ) -> str:
        """
        Transcribes a single audio time segment.

        max_context_tokens=224 - limits the context length of the Whisper decoder,
        protecting against OOM on very long segments.
        """
        segments, _ = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=False,
            condition_on_previous_text=False,    # disable: reduces VRAM
            clip_timestamps=f"{start},{end}",    # process only the segment
            max_new_tokens=max_context_tokens,
            vad_filter=True,                     # skip silence
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(seg.text for seg in segments).strip()
    def transcribe_full(
        self,
        wav_path: str,
        language: str = "uk",
        beam_size: int = 5,
        max_context_tokens: int = 444,
    ) -> List[Tuple[float, float, str]]:
        """
        Transcribes the entire file without diarization.
        Returns a list of (start, end, text).

        max_new_tokens=444: Whisper has a limit of 448 tokens, but the prompt (language/task)
        takes a few tokens. 444 = safe maximum.
        """
        segments, _ = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            max_new_tokens=max_context_tokens,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        
        words_list = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words_list.append((w.start, w.end, w.word))
            else:
                words_list.append((seg.start, seg.end, seg.text))
        return words_list


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: conversion to WAV
# ──────────────────────────────────────────────────────────────────────────────

def _to_wav(input_path: str, output_wav: str, target_sr: int = 16000) -> None:
    """
    Converts an audio file to mono WAV 16 kHz (requirement for pyannote and Whisper).

    First tries soundfile (does not require FFmpeg),
    then av (already installed as a dependency for pyannote.audio),
    then torchaudio as a fallback.
    """
    # --- Attempt 1: soundfile (no FFmpeg needed, supports mp3/wav/flac/ogg) ---
    try:
        import soundfile as sf  # type: ignore
        import numpy as np
        data, file_sr = sf.read(input_path, dtype="float32", always_2d=True)
        # mono
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        # resampling if needed
        if file_sr != target_sr:
            try:
                from scipy.signal import resample_poly  # type: ignore
                from math import gcd
                g = gcd(file_sr, target_sr)
                data = resample_poly(data, target_sr // g, file_sr // g).astype(np.float32)
            except ImportError:
                # scipy is not installed, use numpy resampling (lower quality)
                num_samples = int(len(data) * target_sr / file_sr)
                data = np.interp(
                    np.linspace(0, len(data), num_samples),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
        # save as PCM_16 WAV
        data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        sf.write(output_wav, data_int16, target_sr, format="WAV", subtype="PCM_16")
        return
    except Exception as exc:
        logger.debug("soundfile failed to convert (%s), trying av...", exc)

    # --- Attempt 2: av (already installed with pyannote.audio, supports many formats) ---
    try:
        import av  # type: ignore
        import numpy as np
        container = av.open(input_path)
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=target_sr,
        )
        frames = []
        for frame in container.decode(stream):
            frame = resampler.resample(frame)
            frames.append(frame.to_ndarray())
        container.close()
        if frames:
            audio_data = np.concatenate(frames, axis=1).flatten().astype(np.int16)
        else:
            audio_data = np.zeros(target_sr, dtype=np.int16)
        import soundfile as sf
        sf.write(output_wav, audio_data, target_sr, format="WAV", subtype="PCM_16")
        return
    except Exception as exc:
        logger.debug("av failed to convert (%s), trying torchaudio...", exc)

    # --- Attempt 3: torchaudio (fallback) ---
    try:
        import torchaudio  # type: ignore
        waveform, sr = torchaudio.load(input_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        torchaudio.save(output_wav, waveform, target_sr)
        return
    except Exception as exc:
        logger.warning("All conversion methods failed: %s", exc)
        raise RuntimeError(f"Failed to convert {input_path} to WAV") from exc


# ──────────────────────────────────────────────────────────────────────────────
#  Merge Diarization and ASR
# ──────────────────────────────────────────────────────────────────────────────

def _merge_diarization_asr(
    diar_segments: List[Tuple[float, float, str]],
    asr_segments: List[Tuple[float, float, str]],
) -> List[TranscriptSegment]:
    """
    Merges diarization results (who is speaking) with ASR results (what was said).

    For each ASR segment, finds the most overlapping diarization segment.
    """
    result: List[TranscriptSegment] = []

    current_speaker = None
    current_start = 0.0
    current_end = 0.0
    current_words = []

    for asr_start, asr_end, text in asr_segments:
        text_clean = text.strip()
        if not text_clean:
            continue

        best_speaker = "Speaker 1"
        best_overlap = 0.0

        for d_start, d_end, speaker in diar_segments:
            overlap = max(0.0, min(asr_end, d_end) - max(asr_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        if current_speaker is None:
            current_speaker = best_speaker
            current_start = asr_start
            current_end = asr_end
            current_words = [text_clean]
        elif best_speaker == current_speaker and (asr_start - current_end) < 2.0:
            current_end = max(current_end, asr_end)
            current_words.append(text_clean)
        else:
            result.append(TranscriptSegment(
                start=current_start,
                end=current_end,
                speaker=current_speaker,
                text=" ".join(current_words)
            ))
            current_speaker = best_speaker
            current_start = asr_start
            current_end = asr_end
            current_words = [text_clean]

    if current_speaker is not None:
        result.append(TranscriptSegment(
            start=current_start,
            end=current_end,
            speaker=current_speaker,
            text=" ".join(current_words)
        ))

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Main Class
# ──────────────────────────────────────────────────────────────────────────────

class AudioTranscriber:
    """
    GPU audio transcriber for calls.

    Pipeline:
        A. pyannote.audio - speaker diarization 3.1 (float16, CUDA)
        B. faster-whisper - ASR large-v3 (int8_float16, ~1.6 GB VRAM)

    All work is done sequentially to minimize peak VRAM consumption.
    After each heavy step, memory is cleared via torch.cuda.empty_cache().

    Args:
        hf_token:       Hugging Face token (to download pyannote model).
                        If not provided - read from env HF_TOKEN.
        whisper_model:  Whisper model size ("large-v3" or "large-v3-turbo").
        language:       ASR language (default "uk" - Ukrainian;
                        faster-whisper supports code-switching automatically).
        num_speakers:   Number of speakers (None = auto-detect by pyannote).
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        whisper_model: str = "large-v3",
        language: str = "uk",
        num_speakers: Optional[int] = None,
    ) -> None:
        self._hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._language = language
        self._num_speakers = num_speakers

        if not self._hf_token:
            logger.warning(
                "HF_TOKEN not found. Speaker diarization will be disabled. "
                "Set env HF_TOKEN or pass hf_token= in the constructor."
            )

        logger.info("Initializing AudioTranscriber (GPU-pipeline)...")
        logger.info("  CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("  GPU: %s", torch.cuda.get_device_name(0))
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info("  Total VRAM: %.1f GB", total_vram)

        # Load Whisper immediately upon initialization (stays in VRAM)
        self._asr = _WhisperASR(model_size=whisper_model)

    # ------------------------------------------------------------------ #
    #  Public API: transcribe
    # ------------------------------------------------------------------ #

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes an audio file (mp3/wav/ogg/...) and returns structured text.

        Output format:
            [00:12] Speaker 1: Conversation text...
            [00:15] Speaker 2: Reply in local dialect...

        Args:
            audio_path: Path to the audio file (any format supported by torchaudio).

        Returns:
            Transcription string or empty string in case of a critical error.
        """
        logger.info("=== Transcription: %s ===", os.path.basename(audio_path))

        with tempfile.TemporaryDirectory(prefix="transcriber_") as tmpdir:
            # ── 1. Conversion to WAV 16kHz mono ──────────────────────────
            raw_wav = os.path.join(tmpdir, "raw.wav")
            try:
                _to_wav(audio_path, raw_wav)
                logger.info("Converted to WAV 16kHz mono.")
            except Exception as exc:
                logger.error("Error converting to WAV: %s", exc)
                return ""

            # ── A. Diarization ────────────────────────────────────────────
            diar_segments: List[Tuple[float, float, str]] = []
            if self._hf_token:
                try:
                    diar_segments = _diarize(
                        raw_wav,
                        hf_token=self._hf_token,
                        num_speakers=self._num_speakers,
                    )
                except Exception as exc:
                    logger.error("[Step A] Diarization error: %s. Continuing without it.", exc)
                    _free_vram("after diarization error")
            else:
                logger.info("[Step A] Diarization skipped (no HF_TOKEN).")

            # ── B. ASR ────────────────────────────────────────────────────
            _log_vram("before ASR")
            try:
                asr_raw = self._asr.transcribe_full(raw_wav, language=self._language)
            except Exception as exc:
                logger.error("[Step B] ASR error: %s", exc)
                return ""

            logger.info("[Step B] ASR completed: %d segments.", len(asr_raw))
            _log_vram("after ASR")

        # ── Merge Diarization and ASR ──────────────────────────────────────
        merged = _merge_diarization_asr(diar_segments, asr_raw)

        if not merged:
            logger.warning("Transcription is empty.")
            return ""

        result = "\n".join(seg.to_line() for seg in merged)
        logger.info(
            "=== Transcription completed: %d lines, %d characters ===",
            len(merged), len(result),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Public API: transcribe_bilingual (two texts for combined analysis)
    # ------------------------------------------------------------------ #

    def transcribe_bilingual(self, audio_path: str) -> Tuple[str, str]:
        """
        Transcribes an audio file in TWO languages: Ukrainian (uk) and Russian (ru).
        
        Returns a tuple (uk_text, ru_text) for combined AI analysis.
        The AI receives both texts and autonomously decides how to use them.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple (uk_text, ru_text) or ("", "") in case of error.
        """
        logger.info("=== Bilingual transcription: %s ===", os.path.basename(audio_path))

        with tempfile.TemporaryDirectory(prefix="transcriber_") as tmpdir:
            # ── 1. Conversion to WAV 16kHz mono ──────────────────────────
            raw_wav = os.path.join(tmpdir, "raw.wav")
            try:
                _to_wav(audio_path, raw_wav)
                logger.info("Converted to WAV 16kHz mono.")
            except Exception as exc:
                logger.error("Error converting to WAV: %s", exc)
                return "", ""

            # ── A. Diarization (shared for both languages) ─────────────────────
            diar_segments: List[Tuple[float, float, str]] = []
            if self._hf_token:
                try:
                    diar_segments = _diarize(
                        raw_wav,
                        hf_token=self._hf_token,
                        num_speakers=self._num_speakers,
                    )
                except Exception as exc:
                    logger.error("[Step A] Diarization error: %s. Continuing without it.", exc)
                    _free_vram("after diarization error")
            else:
                logger.info("[Step A] Diarization skipped (no HF_TOKEN).")

            # ── B. ASR in both languages ──────────────────────────────────────
            _log_vram("before ASR")
            try:
                asr_uk = self._asr.transcribe_full(raw_wav, language="uk")
                logger.info("[Step B-uk] ASR Ukrainian completed: %d segments.", len(asr_uk))
                
                asr_ru = self._asr.transcribe_full(raw_wav, language="ru")
                logger.info("[Step B-ru] ASR Russian completed: %d segments.", len(asr_ru))
            except Exception as exc:
                logger.error("[Step B] ASR error: %s", exc)
                return "", ""

            _log_vram("after ASR")

        # ── Merge Diarization and ASR for both languages ──────────────────────────
        merged_uk = _merge_diarization_asr(diar_segments, asr_uk)
        merged_ru = _merge_diarization_asr(diar_segments, asr_ru)

        if not merged_uk or not merged_ru:
            logger.warning("Bilingual transcription is empty.")
            return "", ""

        result_uk = "\n".join(seg.to_line() for seg in merged_uk)
        result_ru = "\n".join(seg.to_line() for seg in merged_ru)
        
        logger.info(
            "=== Bilingual transcription completed: UK=%d lines, RU=%d lines ===",
            len(merged_uk), len(merged_ru),
        )
        return result_uk, result_ru

    # ------------------------------------------------------------------ #
    #  Public API: transcribe_and_save (backward compatibility for main.py)
    # ------------------------------------------------------------------ #

    def transcribe_and_save(self, mp3_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Transcribes an audio file and saves the result to a .txt file.

        Returns a tuple (transcription text, path to .txt file).
        Maintains backward compatibility with the original API for main.py.
        """
        text = self.transcribe(mp3_path)

        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info("Transcription saved: %s", txt_path)
        return text, txt_path
