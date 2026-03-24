"""
Data pipeline: load all corpora, optionally translate German rows to English,
and print statistics.

Usage:
    python run_pipeline.py                          # load only, no translation
    python run_pipeline.py --translate              # load + translate German rows
    python run_pipeline.py --corpora gmhp7k jigsaw # load a subset
    python run_pipeline.py --translate --batch-size 64 --device mps
"""
import argparse
from pathlib import Path

from src.dataset import UnifiedCorpusDataset

DATA_DIR  = Path(__file__).parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent / "data" / "cache"


def main(corpora=None, translate=False, batch_size=32, device=None):
    print("=== Netiquette Multilabel Classification — Data Pipeline ===\n")

    # ── 1. Load ──────────────────────────────────────────────────────
    dataset = UnifiedCorpusDataset(DATA_DIR)
    dataset.load(corpora)

    # ── 2. Translate (optional) ───────────────────────────────────────
    if translate:
        from src.preprocessing import GermanToEnglishTranslator

        translator = GermanToEnglishTranslator(
            cache_path=CACHE_DIR / "translations_de_en.parquet",
            batch_size=batch_size,
            device=device,
        )
        dataset._df = translator.translate(dataset.df)

    # ── 3. Statistics ─────────────────────────────────────────────────
    print("\n--- Label distribution ---")
    print(dataset.label_stats().to_string())

    print("\n--- Split distribution ---")
    print(dataset.split_stats().to_string(index=False))

    if translate:
        remaining_de = (dataset.df["language"] == "de").sum()
        print(f"\nNon-English rows remaining: {remaining_de}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=None,
                        help="Subset of corpora to load. Defaults to all.")
    parser.add_argument("--translate", action="store_true",
                        help="Translate German rows to English.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Translation batch size (default: 32).")
    parser.add_argument("--device", default=None,
                        help="Device for translation: cpu / cuda / mps (auto-detect if omitted).")
    args = parser.parse_args()

    main(
        corpora=args.corpora,
        translate=args.translate,
        batch_size=args.batch_size,
        device=args.device,
    )
