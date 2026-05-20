"""
Модуль для транскрибації аудіо-файлів.

Конвертує .mp3 → .wav за допомогою soundfile (без ffmpeg),
після чого передає .wav у SpeechRecognition для розпізнавання мовлення.
"""

import io
import logging
import os
import tempfile

import soundfile as sf
import speech_recognition as sr

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Клас для транскрибації .mp3 файлів.

    Використовує soundfile для декодування mp3 без системного ffmpeg
    та SpeechRecognition (Google Web Speech API) для розпізнавання мовлення.
    """

    def __init__(self, language: str = "uk-UA") -> None:
        self._language = language
        self._recognizer = sr.Recognizer()

    def _convert_mp3_to_wav_bytes(self, mp3_path: str) -> bytes:
        """
        Читає .mp3 через soundfile та повертає вміст .wav у байтах.
        soundfile >= 0.11.0 має вбудований декодер mp3 (libsndfile з підтримкою MPEG).
        """
        data, sample_rate = sf.read(mp3_path, dtype="int16")
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, data, sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        return wav_buffer.read()

    def transcribe(self, mp3_path: str) -> str:
        """
        Транскрибує один .mp3 файл.

        Повертає рядок транскрипції або порожній рядок у разі помилки.
        """
        logger.info("Транскрибація: %s", os.path.basename(mp3_path))

        try:
            wav_bytes = self._convert_mp3_to_wav_bytes(mp3_path)
        except Exception as exc:
            logger.error("Помилка конвертації mp3→wav (%s): %s", mp3_path, exc)
            return ""

        # Записуємо wav у тимчасовий файл для SpeechRecognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(wav_bytes)
            tmp_wav_path = tmp_wav.name

        try:
            with sr.AudioFile(tmp_wav_path) as source:
                audio_data = self._recognizer.record(source)
            text = self._recognizer.recognize_google(
                audio_data, language=self._language
            )
            logger.info(
                "Транскрибація '%s' завершена (%d символів).",
                os.path.basename(mp3_path),
                len(text),
            )
            return text
        except sr.UnknownValueError:
            logger.warning("Мову не розпізнано у файлі: %s", mp3_path)
            return ""
        except sr.RequestError as exc:
            logger.error("Помилка запиту до Google Speech API: %s", exc)
            return ""
        finally:
            os.unlink(tmp_wav_path)

    def transcribe_and_save(self, mp3_path: str, output_dir: str) -> tuple[str, str]:
        """
        Транскрибує .mp3 та зберігає текст у .txt файл поруч із аудіо.

        Повертає кортеж (текст транскрипції, шлях до .txt файлу).
        """
        text = self.transcribe(mp3_path)

        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info("Транскрипцію збережено: %s", txt_path)
        return text, txt_path
