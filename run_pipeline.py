"""
Data pipeline: load all corpora, optionally translate English rows to German,
optionally assign missing train/val/test splits, print statistics, and save
the processed dataset to data/processed/.

Usage
-----
    # Load all corpora and save (no translation, no split assignment)
    python run_pipeline.py

    # Translate English rows to German before saving
    python run_pipeline.py --translate

    # Assign random stratified splits to rows that lack a predefined split
    python run_pipeline.py --assign-splits

    # Full pipeline: translate + assign splits
    python run_pipeline.py --translate --assign-splits

    # Process a subset of corpora only
    python run_pipeline.py --corpora gmhp7k jigsaw

    # Custom translation settings
    python run_pipeline.py --translate --batch-size 64 --device mps

Output filenames (written to data/processed/)
----------------------------------------------
    unified.parquet
    unified_translated.parquet
    unified_with_splits.parquet
    unified_translated_with_splits.parquet
"""

import argparse
import sys
from pathlib import Path

from src.dataset import UnifiedCorpusDataset
from src.dataset.schema import SCHEMA_COLUMNS

DATA_DIR = Path(__file__).parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

_TRANSLATION_METADATA_COLUMNS = [
    "text_original",
    "is_translated",
    "translation_status",
]


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------

def _output_filename(translate: bool, assign_splits: bool) -> str:
    """Return the parquet filename that reflects the processing options applied."""
    parts = ["unified"]
    if translate:
        parts.append("translated")
    if assign_splits:
        parts.append("with_splits")
    return "_".join(parts) + ".parquet"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _load(corpora):
    print("=== Netiquette Multilabel Classification — Data Pipeline ===\n")
    print("── Stage 1: Loading corpora ─────────────────────────────────────")
    dataset = UnifiedCorpusDataset(DATA_DIR)
    dataset.load(corpora)
    return dataset


def _translate(dataset, batch_size, device):
    print("\n── Stage 2: Translation (en → de) ───────────────────────────────")
    from src.preprocessing import EnglishToGermanTranslator

    translator = EnglishToGermanTranslator(
        cache_path=CACHE_DIR / "translations_en_de.parquet",
        batch_size=batch_size,
        device=device,
    )
    dataset._df = translator.translate(dataset.df)

    if "translation_status" in dataset._df.columns:
        n_success = int((dataset._df["translation_status"] == "success").sum())
        n_failed = int((dataset._df["translation_status"] == "failed").sum())
        n_empty = int((dataset._df["translation_status"] == "empty").sum())

        print(f"  Rows translated successfully : {n_success:,}")
        print(f"  Rows failed                  : {n_failed:,}")
        print(f"  Rows skipped (empty)         : {n_empty:,}")


def _assign_splits(dataset):
    print("\n── Stage 3: Split assignment ─────────────────────────────────────")
    try:
        from src.dataset.splits import assign_missing_splits
    except ImportError:
        print(
            "  ERROR: src/dataset/splits.py does not exist yet.\n"
            "  To use --assign-splits, implement assign_missing_splits() in that module.\n"
            "  Expected signature:\n"
            "    def assign_missing_splits(df: pd.DataFrame) -> pd.DataFrame:\n"
            "        ...\n"
            "  It should assign 'train'/'val'/'test' to rows where split is None,\n"
            "  using a stratified random strategy appropriate for multilabel data.",
            file=sys.stderr,
        )
        sys.exit(1)

    before = int(dataset.df["split"].isna().sum())
    dataset._df = assign_missing_splits(dataset.df)
    after = int(dataset.df["split"].isna().sum())

    print(f"  Rows assigned a split: {before - after:,}  (null splits remaining: {after:,})")


def _print_stats(dataset, translate):
    print("\n── Statistics ───────────────────────────────────────────────────")
    print("\nLabel distribution:")
    print(dataset.label_stats().to_string())

    print("\nSplit distribution:")
    print(dataset.split_stats().to_string(index=False))

    if translate:
        n_de = int((dataset.df["language"] == "de").sum())
        print(f"\nGerman rows in final dataset: {n_de:,}")

        if "is_translated" in dataset.df.columns:
            n_silver = int(dataset.df["is_translated"].sum())
            n_gold_de = int(
                ((dataset.df["language"] == "de") & (~dataset.df["is_translated"])).sum()
            )
            print(f"German gold rows           : {n_gold_de:,}")
            print(f"Translated silver rows     : {n_silver:,}")


def _save(dataset, translate, assign_splits):
    print("\n── Stage 4: Saving processed dataset ────────────────────────────")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    filename = _output_filename(translate, assign_splits)
    out_path = PROCESSED_DIR / filename

    save_df = dataset.df.drop(
        columns=[c for c in _TRANSLATION_METADATA_COLUMNS if c in dataset.df.columns]
    )

    save_df.to_parquet(out_path, index=False)

    print(f"  Saved  : {out_path}")
    print(f"  Rows   : {len(save_df):,}")
    print(f"  Columns: {save_df.columns.tolist()}")

    assert list(save_df.columns) == SCHEMA_COLUMNS, (
        f"Unexpected columns — expected {SCHEMA_COLUMNS}, "
        f"got {save_df.columns.tolist()}"
    )
    print("  Schema : OK (matches SCHEMA_COLUMNS)")

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(corpora=None, translate=False, assign_splits=False, batch_size=32, device=None):
    dataset = _load(corpora)

    if translate:
        _translate(dataset, batch_size, device)

    if assign_splits:
        _assign_splits(dataset)

    _print_stats(dataset, translate)
    _save(dataset, translate, assign_splits)

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load, process, and save the unified multilabel corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=None,
        help="Subset of corpora to load (default: all).",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate English rows to German using MarianMT.",
    )
    parser.add_argument(
        "--assign-splits",
        action="store_true",
        dest="assign_splits",
        help="Assign train/val/test splits to rows that lack a predefined split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Translation batch size (default: 32).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for translation: cpu / cuda / mps (auto-detect if omitted).",
    )
    args = parser.parse_args()

    main(
        corpora=args.corpora,
        translate=args.translate,
        assign_splits=args.assign_splits,
        batch_size=args.batch_size,
        device=args.device,
    )
