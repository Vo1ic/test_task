"""
Main orchestrator script for the QA call audit automation.

Execution flow:
1. Connects to Google Drive via Service Account or OAuth2.
2. Downloads .mp3 files from the "Calls" folder to a temporary directory.
3. GPU transcription of each file (diarization -> Whisper ASR).
4. Saves .txt transcription to Drive next to the audio file.
5. LLM analysis of the transcription (Qwen2.5-7B on GPU via llama-cpp-python).
6. Writes results to the created Google Sheets spreadsheet (21 columns).
7. Adds a summary =SUM(...) formula in the 'Final Score' column.

Environment requirements:
    GOOGLE_CLIENT_SECRET - path to OAuth 2.0 Client ID JSON (default: client_secret.json)
    HF_TOKEN             - Hugging Face token (required for pyannote.audio speaker diarization)
"""

import logging
import os
import sys
import tempfile

# Automatically load .env (if it exists) to avoid manual token injection
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Fix for Windows: configure UTF-8 console encoding, update PATH from registry (for ffmpeg/rust/build tools),
# and register paths to CUDA DLLs globally before importing any libraries.
if sys.platform == "win32":
    # 1. Налаштовуємо стандартне кодування на UTF-8 для уникнення UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 2. Оновлюємо PATH з реєстру, щоб підтягнути нові системні/користувацькі змінні середовища без перезапуску оболонки
    try:
        import winreg
        paths = []
        for hkey, subkey in [
            (winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment")
        ]:
            try:
                with winreg.OpenKey(hkey, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    paths.append(val)
            except FileNotFoundError:
                pass
        if paths:
            os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    # 3. Додаємо nvidia DLLs
    import site
    import glob
    for site_pkg in site.getsitepackages():
        for nv_bin in glob.glob(os.path.join(site_pkg, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(nv_bin)
            except Exception:
                pass
            if nv_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = nv_bin + os.pathsep + os.environ["PATH"]

from analyzer import ConversationAnalyzer
from google_services import GoogleServicesClient
from sheets_updater import SheetsUpdater
from transcriber import AudioTranscriber

# ──────────────────────────────────────────────────────────────────────────────
#  Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Path to the OAuth 2.0 Client ID (JSON) file.
# Can be overridden via GOOGLE_CLIENT_SECRET environment variable.
CLIENT_SECRET_PATH = os.environ.get("GOOGLE_CLIENT_SECRET", "client_secret.json")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting Auto Service QA Call Audit Script")
    logger.info("=" * 60)

    # 1. Initialize Google Services
    logger.info("Step 1/6: Initializing Google Services (OAuth2)...")
    google_client = GoogleServicesClient(client_secret_path=CLIENT_SECRET_PATH)

    # 2. Get the list of .mp3 files
    logger.info("Step 2/6: Fetching .mp3 file list from Google Drive...")
    mp3_files_meta = google_client.get_mp3_files_meta()

    if not mp3_files_meta:
        logger.warning("No files found for processing. Exiting.")
        return

    logger.info("Found %d file(s) for processing.", len(mp3_files_meta))

    # 3. Initialize Transcriber and LLM Analyzer
    logger.info("Step 3/6: Initializing GPU Transcriber and ConversationAnalyzer...")
    # HF_TOKEN is required for speaker diarization (pyannote.audio).
    # Without it, diarization will be disabled and all text marked as 'Speaker 1'.
    transcriber = AudioTranscriber(
        hf_token=os.environ.get("HF_TOKEN", ""),
        whisper_model="large-v3",
        language="uk",
        num_speakers=2,          # usually 2 speakers: manager + client
    )
    # On the first run, it automatically downloads the Qwen2.5-7B GGUF model (~4.5 GB)
    # and loads it fully into GPU (n_gpu_layers=-1).
    analyzer = ConversationAnalyzer()

    # 4. Create a new spreadsheet and initialize SheetsUpdater
    logger.info("Step 4/6: Creating a new Google Sheets spreadsheet...")
    spreadsheet = google_client.create_report_spreadsheet()
    sheets_updater = SheetsUpdater(spreadsheet=spreadsheet)
    logger.info("Spreadsheet ready: %s", sheets_updater.spreadsheet_url)

    # 5. Get the ID of the folder where audio files will be copied and transcriptions uploaded
    # We use REPORT_FOLDER_ID as the primary target workspace folder
    from google_services import REPORT_FOLDER_ID
    target_folder_id = REPORT_FOLDER_ID

    # 6. Process each file sequentially (copy -> download -> process -> delete)
    logger.info("Step 5/6: Processing files...")
    with tempfile.TemporaryDirectory(prefix="calls_") as tmp_dir:
        for idx, file_meta in enumerate(mp3_files_meta, start=1):
            file_id = file_meta["id"]
            filename = file_meta["name"]
            logger.info("--- [%d/%d] Processing: %s ---", idx, len(mp3_files_meta), filename)

            # 6.1 Copy audio file to the target workspace folder on Google Drive
            try:
                google_client.copy_file_to_folder(file_id, filename, target_folder_id)
            except Exception as exc:
                logger.warning("Failed to copy %s to the Drive folder: %s", filename, exc)

            # 6.2 Download the file locally for transcription
            mp3_path = google_client.download_single_file(file_id, filename, tmp_dir)

            # 6.3 Transcription (GPU pipeline: two languages - UK + RU)
            transcription_uk, transcription_ru = transcriber.transcribe_bilingual(
                audio_path=mp3_path,
            )
            # Note: Text truncation to fit the LLM context limit is handled
            # automatically inside ConversationAnalyzer.analyze_bilingual() via _MAX_TEXT_CHARS.

            # 6.4 Upload .txt transcription to Drive next to the copied audio
            if target_folder_id:
                base_name = os.path.splitext(os.path.basename(mp3_path))[0]
                
                # Save Ukrainian version
                if transcription_uk:
                    txt_path_uk = os.path.join(tmp_dir, f"{base_name}_uk.txt")
                    with open(txt_path_uk, "w", encoding="utf-8") as f:
                        f.write(transcription_uk)
                    try:
                        google_client.upload_text_file(txt_path_uk, target_folder_id)
                    except Exception as exc:
                        logger.warning("Failed to upload Ukrainian transcription to Drive: %s", exc)
                
                # Save Russian version
                if transcription_ru:
                    txt_path_ru = os.path.join(tmp_dir, f"{base_name}_ru.txt")
                    with open(txt_path_ru, "w", encoding="utf-8") as f:
                        f.write(transcription_ru)
                    try:
                        google_client.upload_text_file(txt_path_ru, target_folder_id)
                    except Exception as exc:
                        logger.warning("Failed to upload Russian transcription to Drive: %s", exc)

            # 6.5 Combined LLM analysis (both texts)
            analysis_result = analyzer.analyze_bilingual(
                text_uk=transcription_uk,
                text_ru=transcription_ru,
            )

            # 6.6 Write row to Google Sheets (21 columns)
            # We use the combined text for storage
            combined_text = f"[UK]\n{transcription_uk}\n\n[RU]\n{transcription_ru}"
            sheets_updater.add_call_record(
                filename=filename,
                transcription_text=combined_text,
                analysis=analysis_result,
            )

            # 6.7 Cleanup local files (save space)
            try:
                os.remove(mp3_path)
                logger.debug("Local files for %s deleted.", filename)
            except Exception as e:
                logger.warning("Failed to delete temporary files: %s", e)

    # 7. Summary =SUM(...) formula
    logger.info("Step 6/6: Adding final score summary formula...")
    sheets_updater.add_total_sum_formula()

    logger.info("=" * 60)
    logger.info("Processing completed successfully!")
    logger.info("Results available at: %s", sheets_updater.spreadsheet_url)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
