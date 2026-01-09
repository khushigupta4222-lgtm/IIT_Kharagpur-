import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.load_data import load_example
from src.chunking.chunker import chunk_text


def main():
    example_dir = PROJECT_ROOT / "data" / "example_001"

    novel, backstory = load_example(str(example_dir))

    print("Novel loaded successfully.")
    print("Novel length:", len(novel))

    print("\nBackstory loaded successfully.")
    print("Backstory length:", len(backstory))

    chunks = chunk_text(novel)

    print("\nChunking done.")
    print("Total chunks created:", len(chunks))

    prediction = 1
    print("\nBaseline prediction:", prediction)


if __name__ == "_main_":
    main()
