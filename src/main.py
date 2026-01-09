import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.load_data import load_example


def main():
    example_dir = "data/example_001"

    novel, backstory = load_example(example_dir)

    print("Novel loaded successfully.")
    print("Novel length (characters):", len(novel))

    print("\nBackstory loaded successfully.")
    print("Backstory length (characters):", len(backstory))

    prediction = 1
    print("\nBaseline prediction:", prediction)


if __name__ == "__main__":
    main()

