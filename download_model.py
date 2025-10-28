"""
Download model from Google Drive if not already present
"""

import os
from pathlib import Path
import urllib.request
import sys


def get_google_drive_file_id(url):
    """Extract file ID from Google Drive sharing URL"""
    if 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    elif '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    return None


def download_google_drive_file(file_id, output_path, chunk_size=8192):
    """Download file from Google Drive using file ID"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"Downloading model from Google Drive...")
    print(f"Output: {output_path}")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as out_file:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                              end='', file=sys.stderr)

                print("\n[SUCCESS] Model downloaded successfully!", file=sys.stderr)
                return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download: {e}", file=sys.stderr)
        return False


def ensure_model_exists(model_path, google_drive_url):
    """
    Check if model exists, if not download from Google Drive

    Args:
        model_path: Path where model should be saved
        google_drive_url: Google Drive sharing URL

    Returns:
        True if model exists or was downloaded successfully
        False if download failed
    """
    model_path = Path(model_path)

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {model_path} ({size_mb:.1f} MB)")
        return True

    print(f"Model not found at {model_path}")
    print(f"Downloading from Google Drive...")

    file_id = get_google_drive_file_id(google_drive_url)
    if not file_id:
        print(f"[ERROR] Could not extract file ID from URL")
        return False

    model_path.parent.mkdir(parents=True, exist_ok=True)

    return download_google_drive_file(file_id, model_path)


if __name__ == "__main__":
    model_url = "https://drive.google.com/file/d/1xM0ZkliwQ4o1KnFIq4j23BNYpZf8nVU7/view?usp=sharing"
    model_path = Path("models/best_model.pt")

    if ensure_model_exists(model_path, model_url):
        print(f"\nModel ready at: {model_path}")
    else:
        print(f"\nFailed to download model")
        sys.exit(1)
