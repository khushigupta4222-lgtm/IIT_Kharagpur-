
import pathway as pw
from pathlib import Path

def ingest_example(example_dir: str):
    """
    Ingest novel + backstory using Pathway.
    Returns: (novel_text, backstory_text)
    """

    example_path = Path(example_dir)

    novel_path = example_path / "novel.txt"
    backstory_path = example_path / "backstory.txt"

    if not novel_path.exists() or not backstory_path.exists():
        raise FileNotFoundError("Novel or backstory file missing")

    # Read files normally (Pathway orchestration comes next step)
    novel = novel_path.read_text(encoding="utf-8", errors="ignore")
    backstory = backstory_path.read_text(encoding="utf-8", errors="ignore")

    return novel, backstory


