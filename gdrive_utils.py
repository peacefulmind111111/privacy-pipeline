"""Utility helpers for uploading files to Google Drive."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except Exception:  # pragma: no cover - dependencies may be missing during tests
    service_account = build = MediaFileUpload = None


def upload_file_to_gdrive(
    local_path: str, folder_id: str, credentials_file: Optional[str] = None
) -> Optional[str]:
    """Upload ``local_path`` to Google Drive ``folder_id``.

    Returns the file ID on success. If the Google libraries are not available,
    the function returns ``None`` without raising errors, which makes it safe
    to call in environments without Google credentials."""
    if build is None:
        return None

    creds = None
    if credentials_file:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=["https://www.googleapis.com/auth/drive.file"]
        )

    service = build("drive", "v3", credentials=creds)
    file_metadata = {"name": Path(local_path).name, "parents": [folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    return file.get("id")
