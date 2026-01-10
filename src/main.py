print(">>> main.py file loaded")
import pathway as pw

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.chunking.chunker import chunk_text

def main():
    print(">>> MAIN.PY IS RUNNING <<<")

    # Pathway ingestion
    example_dir = PROJECT_ROOT / "data" / "example_001"

    # Create Pathway Dataset
    ds = pw.dataset(example_dir)

    # Convert to list
    novel = list(ds['novel'])
    backstory = list(ds['backstory'])

    # If your dataset only has one example, grab the first element
    novel = novel[0] if novel else ""
    backstory = backstory[0] if backstory else ""

    print("Novel loaded successfully.")
    print("Novel length:", len(novel))

    print("\nBackstory loaded successfully.")
    print("Backstory length:", len(backstory))

    chunks = chunk_text(novel)

    print("\nChunking done.")
    print("Total chunks created:", len(chunks))

    prediction = 1
    print("\nBaseline prediction:", prediction)


if __name__ == "__main__":
    main()
