"""
Головний оркестратор скрипту автоматизації QA-аудиту дзвінків.

Порядок виконання:
1. Підключення до Google Drive через Service Account.
2. Завантаження .mp3 файлів з папки "Дзвінки" у тимчасову директорію.
3. Транскрибація кожного файлу (mp3 → wav → текст).
4. Збереження .txt транскрипції на Drive поруч із аудіофайлом.
5. LLM-аналіз транскрипції (llama-cpp-python, локально, без платних API).
6. Запис результатів у скопійовану Google Sheets таблицю (21 колонка).
7. Додавання підсумкової формули =SUM(...) у стовпці "Фінальний рахунок".
"""

import logging
import os
import sys
import tempfile

from analyzer import ConversationAnalyzer
from google_services import GoogleServicesClient
from sheets_updater import SheetsUpdater
from transcriber import AudioTranscriber

# ──────────────────────────────────────────────────────────────────────────────
#  Налаштування логування
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
#  Конфігурація
# ──────────────────────────────────────────────────────────────────────────────
# Шлях до файлу OAuth 2.0 Client ID (JSON).
# Можна задати через змінну середовища GOOGLE_CLIENT_SECRET.
CLIENT_SECRET_PATH = os.environ.get("GOOGLE_CLIENT_SECRET", "client_secret.json")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Запуск скрипту QA-аудиту дзвінків автосервісу")
    logger.info("=" * 60)

    # 1. Ініціалізація Google Services
    logger.info("Крок 1/6: Ініціалізація Google Services (OAuth2)...")
    google_client = GoogleServicesClient(client_secret_path=CLIENT_SECRET_PATH)

    # 2. Отримання списку .mp3 файлів
    logger.info("Крок 2/6: Отримання списку .mp3 файлів з Google Drive...")
    mp3_files_meta = google_client.get_mp3_files_meta()

    if not mp3_files_meta:
        logger.warning("Файлів для обробки не знайдено. Завершення роботи.")
        return

    logger.info("Знайдено %d файл(ів) для обробки.", len(mp3_files_meta))

    # 3. Ініціалізація транскрайбера та LLM-аналізатора
    logger.info("Крок 3/6: Ініціалізація Transcriber та ConversationAnalyzer...")
    transcriber = AudioTranscriber(language="uk-UA")
    # При першому запуску автоматично завантажує GGUF-модель (~1.9 ГБ)
    analyzer = ConversationAnalyzer()

    # 4. Створення нової таблиці та ініціалізація SheetsUpdater
    logger.info("Крок 4/6: Створення нової Google Sheets таблиці...")
    spreadsheet = google_client.create_report_spreadsheet()
    sheets_updater = SheetsUpdater(spreadsheet=spreadsheet)
    logger.info("Таблицю підготовлено: %s", sheets_updater.spreadsheet_url)

    # 5. Отримання ID папки, куди будуть копіюватись аудіо і завантажуватись транскрипції
    # Використовуємо REPORT_FOLDER_ID, оскільки це "ваша робоча папка" за ТЗ
    from google_services import REPORT_FOLDER_ID
    target_folder_id = REPORT_FOLDER_ID

    # 6. Обробка кожного файлу по черзі (скопіював -> завантажив -> обробив -> видалив)
    logger.info("Крок 5/6: Обробка файлів...")
    with tempfile.TemporaryDirectory(prefix="calls_") as tmp_dir:
        for idx, file_meta in enumerate(mp3_files_meta, start=1):
            file_id = file_meta["id"]
            filename = file_meta["name"]
            logger.info("--- [%d/%d] Обробка: %s ---", idx, len(mp3_files_meta), filename)

            # 6.1 Копіювання аудіофайлу у вашу робочу папку (на Google Drive)
            try:
                google_client.copy_file_to_folder(file_id, filename, target_folder_id)
            except Exception as exc:
                logger.warning("Не вдалося скопіювати %s на ваш Drive: %s", filename, exc)

            # 6.2 Завантаження файлу локально (для транскрибації)
            mp3_path = google_client.download_single_file(file_id, filename, tmp_dir)

            # 6.3 Транскрибація
            transcription_text, txt_path = transcriber.transcribe_and_save(
                mp3_path=mp3_path,
                output_dir=tmp_dir,
            )

            # Якщо текст занадто великий для LLM (обрізаємо до ~40000 символів для безпеки)
            if len(transcription_text) > 40000:
                logger.warning("Текст транскрипції занадто довгий (%d символів), обрізаємо...", len(transcription_text))
                transcription_text = transcription_text[:40000]

            # 6.4 Завантаження .txt транскрипції на Drive у вашу робочу папку (поруч із скопійованим аудіо)
            if target_folder_id and os.path.exists(txt_path):
                try:
                    google_client.upload_text_file(txt_path, target_folder_id)
                except Exception as exc:
                    logger.warning("Не вдалося завантажити транскрипцію на Drive: %s", exc)

            # 6.5 LLM-аналіз транскрипції
            analysis_result = analyzer.analyze(text=transcription_text)

            # 6.5 Запис рядка у Google Sheets (21 колонка)
            sheets_updater.add_call_record(
                filename=filename,
                transcription_text=transcription_text,
                analysis=analysis_result,
            )

            # 6.6 Очищення локальних файлів (збереження місця)
            try:
                os.remove(mp3_path)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                logger.debug("Локальні файли для %s видалено.", filename)
            except Exception as e:
                logger.warning("Не вдалося видалити тимчасові файли: %s", e)

    # 7. Підсумкова формула =SUM(...)
    logger.info("Крок 6/6: Додавання підсумкової формули балів...")
    sheets_updater.add_total_sum_formula()

    logger.info("=" * 60)
    logger.info("Обробка завершена успішно!")
    logger.info("Результати доступні за посиланням: %s", sheets_updater.spreadsheet_url)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
