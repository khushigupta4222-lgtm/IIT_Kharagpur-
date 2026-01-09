from pathlib import Path


def load_example(example_dir):
    example_path = Path(example_dir)

    novel_path = example_path / "novel.txt"
    backstory_path = example_path / "backstory.txt"

    if not novel_path.exists():
        raise FileNotFoundError(f"Missing file: {novel_path}")

    if not backstory_path.exists():
        raise FileNotFoundError(f"Missing file: {backstory_path}")

    novel = novel_path.read_text(encoding="utf-8", errors="ignore")
    backstory = backstory_path.read_text(encoding="utf-8", errors="ignore")

    return novel, backstory
