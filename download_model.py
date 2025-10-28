"""
Download model from Google Drive if not already present
Uses gdown library for reliable Google Drive downloads
"""

from pathlib import Path
import sys


def get_google_drive_file_id(url):
    """Extract file ID from Google Drive sharing URL"""
    if 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    elif '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    return None


def download_google_drive_file(file_id, output_path):
    """Download file from Google Drive using gdown library"""
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown library not installed")
        print("Install with: pip install gdown")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = str(output_path)

    print(f"Downloading model from Google Drive...")
    print(f"Output: {output_path}")

    try:
        gdown.download(url, output_path, quiet=False)

        # Verify file was downloaded
        if Path(output_path).exists() and Path(output_path).stat().st_size > 1000000:
            print("\n[SUCCESS] Model downloaded successfully!")
            return True
        else:
            print("\n[ERROR] Downloaded file is too small or missing")
            return False

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
