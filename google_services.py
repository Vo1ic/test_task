"""
Модуль для роботи з Google Drive та Google Sheets.
Відповідає за OAuth2 авторизацію користувача,
завантаження файлів з Drive та створення таблиць.
"""

import io
import os
import json
import logging
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import gspread

logger = logging.getLogger(__name__)

# Дозволи, необхідні для роботи з Drive та Sheets
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ID розшареної папки на Google Drive (для читання аудіо)
SHARED_FOLDER_ID = "1R3hDscEc_Ujh1FytqWROg4tS__qcO1Ub"

# ID папки для збереження звіту (Google Sheets)
REPORT_FOLDER_ID = "1jGBXn4w_vkb-NwneUJowEOhY-YNch7NL"

CALLS_SUBFOLDER_NAME = "Дзвінки"


class GoogleServicesClient:
    """
    Клієнт для взаємодії з Google Drive та Google Sheets через OAuth2.
    """

    def __init__(self, client_secret_path: str = "client_secret.json", token_path: str = "token.json") -> None:
        self._credentials = self._authenticate(client_secret_path, token_path)
        self._drive_service = build("drive", "v3", credentials=self._credentials)
        self._gc = gspread.authorize(self._credentials)
        logger.info("GoogleServicesClient успішно ініціалізовано (OAuth2).")

    def _authenticate(self, client_secret_path: str, token_path: str) -> Credentials:
        """
        Обробляє OAuth2 авторизацію. Завантажує існуючий токен або відкриває браузер.
        """
        creds = None
        
        # Завантажуємо існуючий токен, якщо він є
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except Exception as e:
                logger.warning("Не вдалося завантажити token.json: %s", e)

        # Якщо токена немає, або він невалідний
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Оновлення існуючого токена доступу...")
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning("Не вдалося оновити токен (%s), потрібна нова авторизація.", e)
                    creds = None

            if not creds:
                logger.info("Потрібна авторизація. Відкриття браузера...")
                if not os.path.exists(client_secret_path):
                    raise FileNotFoundError(
                        f"Файл {client_secret_path} не знайдено! "
                        "Будь ласка, завантажте OAuth 2.0 Client ID (тип Desktop App) з Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secret_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Зберігаємо токен для наступних запусків
            with open(token_path, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
                logger.info("Новий токен доступу збережено у %s", token_path)

        return creds

    # ------------------------------------------------------------------ #
    #  Drive — пошук папок та файлів
    # ------------------------------------------------------------------ #

    def _find_folder_id(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Знаходить ID папки за назвою всередині батьківської папки."""
        query = (
            f"name='{folder_name}' "
            f"and '{parent_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        response = (
            self._drive_service.files()
            .list(q=query, fields="files(id, name)", supportsAllDrives=True,
                  includeItemsFromAllDrives=True)
            .execute()
        )
        files = response.get("files", [])
        if not files:
            logger.warning("Папку '%s' не знайдено в '%s'.", folder_name, parent_id)
            return None
        return files[0]["id"]

    def _list_mp3_files(self, folder_id: str) -> list[dict]:
        """Повертає список усіх .mp3 файлів у вказаній папці."""
        query = (
            f"'{folder_id}' in parents "
            f"and mimeType='audio/mpeg' "
            f"and trashed=false"
        )
        response = (
            self._drive_service.files()
            .list(q=query, fields="files(id, name)", supportsAllDrives=True,
                  includeItemsFromAllDrives=True)
            .execute()
        )
        return response.get("files", [])

    # ------------------------------------------------------------------ #
    #  Drive — завантаження файлів
    # ------------------------------------------------------------------ #

    def get_mp3_files_meta(self) -> list[dict]:
        """
        Знаходить папку 'Дзвінки' у спільній папці та повертає список метаданих
        всіх .mp3 файлів (id, name).
        """
        calls_folder_id = self._find_folder_id(CALLS_SUBFOLDER_NAME, SHARED_FOLDER_ID)
        if calls_folder_id is None:
            logger.error("Не вдалося знайти папку '%s'.", CALLS_SUBFOLDER_NAME)
            return []

        mp3_files = self._list_mp3_files(calls_folder_id)
        if not mp3_files:
            logger.warning("У папці '%s' немає .mp3 файлів.", CALLS_SUBFOLDER_NAME)
            return []

        return mp3_files

    def download_single_file(self, file_id: str, file_name: str, local_dir: str) -> str:
        """
        Завантажує один файл за ID у вказану локальну директорію.
        Повертає локальний шлях до файлу.
        """
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, file_name)

        logger.info("Завантаження файлу: %s", file_name)
        request = self._drive_service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        )
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        with open(local_path, "wb") as f:
            f.write(buffer.getvalue())

        logger.info("Збережено: %s", local_path)
        return local_path

    # ------------------------------------------------------------------ #
    #  Drive — робота з транскрипціями та копіювання
    # ------------------------------------------------------------------ #

    def copy_file_to_folder(self, file_id: str, new_name: str, destination_folder_id: str) -> str:
        """
        Копіює існуючий файл на Drive у вказану папку.
        Повертає ID створеної копії.
        """
        file_metadata = {
            "name": new_name,
            "parents": [destination_folder_id]
        }
        logger.info("Копіювання файлу '%s' у вашу робочу папку...", new_name)
        response = self._drive_service.files().copy(
            fileId=file_id, body=file_metadata, supportsAllDrives=True, fields="id"
        ).execute()
        return response["id"]

    def upload_text_file(self, local_txt_path: str, parent_folder_id: str) -> None:
        """Завантажує .txt файл транскрипції на Google Drive поруч з аудіо."""
        from googleapiclient.http import MediaFileUpload
        file_name = os.path.basename(local_txt_path)
        file_metadata = {"name": file_name, "parents": [parent_folder_id]}
        media = MediaFileUpload(local_txt_path, mimetype="text/plain")
        self._drive_service.files().create(
            body=file_metadata, media_body=media,
            supportsAllDrives=True, fields="id"
        ).execute()
        logger.info("Транскрипцію завантажено на Drive: %s", file_name)

    def get_calls_folder_id(self) -> Optional[str]:
        """Повертає ID папки 'Дзвінки' для завантаження транскрипцій."""
        return self._find_folder_id(CALLS_SUBFOLDER_NAME, SHARED_FOLDER_ID)

    # ------------------------------------------------------------------ #
    #  Sheets — створення нової таблиці
    # ------------------------------------------------------------------ #

    def create_report_spreadsheet(self) -> gspread.Spreadsheet:
        """
        Створює нову порожню Google Sheets таблицю напряму у папці звіту.
        Тепер, завдяки OAuth2, файл створюється від імені користувача і використовує його квоту.
        """
        title = f"QA Аудит дзвінків — {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        logger.info("Створення нової таблиці: '%s' у цільовій папці...", title)

        file_metadata = {
            "name": title,
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "parents": [REPORT_FOLDER_ID],
        }
        response = (
            self._drive_service.files()
            .create(
                body=file_metadata,
                supportsAllDrives=True,
                fields="id",
            )
            .execute()
        )
        spreadsheet_id = response["id"]
        logger.info(
            "Таблицю '%s' створено (id=%s).", title, spreadsheet_id
        )
        return self._gc.open_by_key(spreadsheet_id)
