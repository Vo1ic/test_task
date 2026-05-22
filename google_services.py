"""
Module for interacting with Google Drive and Google Sheets.
Handles OAuth2 user authorization, downloading files from Drive, and creating spreadsheets.
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

# Required scopes for Drive and Sheets API
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# ID of the shared Google Drive folder (for reading audio files)
SHARED_FOLDER_ID = "1R3hDscEc_Ujh1FytqWROg4tS__qcO1Ub"

# ID of the folder for saving the report (Google Sheets)
REPORT_FOLDER_ID = "1jGBXn4w_vkb-NwneUJowEOhY-YNch7NL"

CALLS_SUBFOLDER_NAME = "Дзвінки"


class GoogleServicesClient:
    """
    Client for interacting with Google Drive and Google Sheets via OAuth2.
    """

    def __init__(self, client_secret_path: str = "client_secret.json", token_path: str = "token.json") -> None:
        self._credentials = self._authenticate(client_secret_path, token_path)
        self._drive_service = build("drive", "v3", credentials=self._credentials)
        self._gc = gspread.authorize(self._credentials)
        logger.info("GoogleServicesClient successfully initialized (OAuth2).")

    def _authenticate(self, client_secret_path: str, token_path: str) -> Credentials:
        """
        Handles OAuth2 authorization. Loads an existing token or opens the browser.
        """
        creds = None
        
        # Load existing token if available
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except Exception as e:
                logger.warning("Failed to load token.json: %s", e)

        # If token doesn't exist or is invalid
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing existing access token...")
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning("Failed to refresh token (%s), new authorization required.", e)
                    creds = None

            if not creds:
                logger.info("Authorization required. Opening browser...")
                if not os.path.exists(client_secret_path):
                    raise FileNotFoundError(
                        f"File {client_secret_path} not found! "
                        "Please download OAuth 2.0 Client ID (Desktop App type) from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secret_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the token for future runs
            with open(token_path, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
                logger.info("New access token saved to %s", token_path)

        return creds

    # ------------------------------------------------------------------ #
    #  Drive - Folder and File Search
    # ------------------------------------------------------------------ #

    def _find_folder_id(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Finds the ID of a folder by name within a parent folder."""
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
            logger.warning("Folder '%s' not found in '%s'.", folder_name, parent_id)
            return None
        return files[0]["id"]

    def _list_mp3_files(self, folder_id: str) -> list[dict]:
        """Returns a list of all .mp3 files in the specified folder."""
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
    #  Drive - File Download
    # ------------------------------------------------------------------ #

    def get_mp3_files_meta(self) -> list[dict]:
        """
        Locates the 'Calls' folder in the shared folder and returns metadata
        for all .mp3 files (id, name).
        """
        calls_folder_id = self._find_folder_id(CALLS_SUBFOLDER_NAME, SHARED_FOLDER_ID)
        if calls_folder_id is None:
            logger.error("Failed to find folder '%s'.", CALLS_SUBFOLDER_NAME)
            return []

        mp3_files = self._list_mp3_files(calls_folder_id)
        if not mp3_files:
            logger.warning("No .mp3 files found in folder '%s'.", CALLS_SUBFOLDER_NAME)
            return []

        return mp3_files

    def download_single_file(self, file_id: str, file_name: str, local_dir: str) -> str:
        """
        Downloads a single file by ID to the specified local directory.
        Returns the local path to the file.
        """
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, file_name)

        logger.info("Downloading file: %s", file_name)
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

        logger.info("Saved: %s", local_path)
        return local_path

    # ------------------------------------------------------------------ #
    #  Drive - Transcriptions and Copying
    # ------------------------------------------------------------------ #

    def copy_file_to_folder(self, file_id: str, new_name: str, destination_folder_id: str) -> str:
        """
        Copies an existing file on Drive to the specified folder.
        Returns the ID of the created copy.
        """
        file_metadata = {
            "name": new_name,
            "parents": [destination_folder_id]
        }
        logger.info("Copying file '%s' to the target folder...", new_name)
        response = self._drive_service.files().copy(
            fileId=file_id, body=file_metadata, supportsAllDrives=True, fields="id"
        ).execute()
        return response["id"]

    def upload_text_file(self, local_txt_path: str, parent_folder_id: str) -> None:
        """Uploads a .txt transcription file to Google Drive alongside the audio."""
        from googleapiclient.http import MediaFileUpload
        file_name = os.path.basename(local_txt_path)
        file_metadata = {"name": file_name, "parents": [parent_folder_id]}
        media = MediaFileUpload(local_txt_path, mimetype="text/plain")
        self._drive_service.files().create(
            body=file_metadata, media_body=media,
            supportsAllDrives=True, fields="id"
        ).execute()
        logger.info("Transcription uploaded to Drive: %s", file_name)

    def get_calls_folder_id(self) -> Optional[str]:
        """Returns the ID of the 'Calls' folder for uploading transcriptions."""
        return self._find_folder_id(CALLS_SUBFOLDER_NAME, SHARED_FOLDER_ID)

    # ------------------------------------------------------------------ #
    #  Sheets - Spreadsheet Creation
    # ------------------------------------------------------------------ #

    def create_report_spreadsheet(self) -> gspread.Spreadsheet:
        """
        Creates a new empty Google Sheets document directly in the report folder.
        Uses OAuth2 to create the file on behalf of the user using their quota.
        """
        title = f"QA Call Audit — {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        logger.info("Creating new spreadsheet: '%s' in the target folder...", title)

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
            "Spreadsheet '%s' created (id=%s).", title, spreadsheet_id
        )
        return self._gc.open_by_key(spreadsheet_id)
