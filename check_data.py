import sys
from pathlib import Path

FOLDERS = [
    Path("data/train/cats"),
    Path("data/train/dogs"),
    Path("data/val/cats"),
    Path("data/val/dogs"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def supports_unicode() -> bool:
    enc = sys.stdout.encoding or ""
    return "UTF" in enc.upper()


HEADER = "\U0001F4CA Dataset Summary" if supports_unicode() else "Dataset Summary"
WARN = "\u26A0\uFE0F WARNING" if supports_unicode() else "WARNING"


def count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> int:
    print(HEADER)
    any_empty = False

    for folder in FOLDERS:
        count = count_images(folder)
        rel = folder.as_posix().replace("data/", "")
        print(f"{rel}: {count} images")
        if count == 0:
            print(f"{WARN}: {rel} folder is empty")
            any_empty = True

    return 1 if any_empty else 0


if __name__ == "__main__":
    sys.exit(main())
