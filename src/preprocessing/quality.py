"""
src/preprocessing/quality.py
-----------------------------
Utilities for annotating the unified corpus DataFrame with data-quality
metadata that distinguishes gold (authentic German) data from silver
(machine-translated) and original-English rows.

This distinction is a prerequisite for:
  - Correct cross-lingual transfer experiments (train on silver, evaluate on gold)
  - Ablation studies (gold-only vs. gold + silver)
  - Preventing evaluation leakage from silver data into the test split
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLD_SOURCES: frozenset[str] = frozenset({"gmhp7k", "hocon34k", "gutefrage", "rp_mod"})
"""
Corpora that contain authentic, human-authored German text with human
annotations.  These are the only sources used for final evaluation.

All other sources (jigsaw, detox, wikipedia_attacks)
are English-origin and enter the pipeline as silver data after machine
translation via EnglishToGermanTranslator.
"""

_VALID_QUALITY_VALUES: frozenset[str] = frozenset({"gold", "silver", "original_en"})


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def add_data_quality_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the unified corpus DataFrame with gold/silver/original_en metadata.

    This function is non-destructive: it operates on a copy of ``df`` and
    never mutates the input.

    Parameters
    ----------
    df:
        The unified corpus DataFrame produced by ``UnifiedCorpusDataset.load()``
        and optionally processed by ``EnglishToGermanTranslator.translate()``.
        Must contain at minimum the columns:
          - ``source``        (str)  : corpus identifier
          - ``is_translated`` (bool) : True if the row was machine-translated
          - ``language``      (str)  : ISO 639-1 language code ('en' or 'de')

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the following changes:

        **New columns:**

        ``is_gold`` (bool)
            True  → source is in GOLD_SOURCES (gmhp7k, hocon34k, gutefrage).
            False → all other sources (silver-origin English datasets).

        ``data_quality`` (str, one of 'gold' | 'silver' | 'original_en')
            Priority logic (higher priority wins):
              1. gold       : ``is_gold == True``  (regardless of is_translated)
              2. silver     : ``is_gold == False`` and ``is_translated == True``
              3. original_en: ``is_gold == False`` and ``is_translated != True``
                              (rows that are still in English, not yet translated)

        **Modified columns:**

        ``language``
            Rows where ``is_translated == True`` are corrected from 'en' to 'de'.
            The original loaders set language='en' at load time; after translation
            the text field contains German, so this correction is required for
            any downstream code that uses ``language`` to filter by target language.

    Notes
    -----
    Gold sources always receive ``data_quality='gold'`` even if ``is_translated``
    were True (which should never occur in a correctly constructed pipeline, but
    the priority rule makes the assignment unambiguous regardless).
    """
    _validate_required_columns(df)
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. is_gold
    # ------------------------------------------------------------------
    df["is_gold"] = df["source"].isin(GOLD_SOURCES)

    # ------------------------------------------------------------------
    # 2. data_quality
    #
    # Build from least-specific (default) to most-specific so that each
    # .where() call can override the previous assignment without chaining
    # complex boolean expressions.
    #
    # pd.Series.where(cond, other) keeps the value where cond is True
    # and replaces it with `other` where cond is False — so we invert:
    #   .where(~mask, replacement)  means "set replacement where mask is True"
    # ------------------------------------------------------------------
    translated_mask = df["is_translated"] == True  # noqa: E712  (handles None/NaN safely)

    data_quality = (
        pd.Series("original_en", index=df.index, dtype="object")
        .where(~translated_mask, other="silver")   # translated but not gold → silver
        .where(~df["is_gold"],   other="gold")      # gold source always wins
    )
    df["data_quality"] = data_quality

    # ------------------------------------------------------------------
    # 3. Fix language column for translated rows
    # ------------------------------------------------------------------
    df.loc[translated_mask, "language"] = "de"

    return df


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_quality_summary(df: pd.DataFrame) -> None:
    """
    Print a pivot table of row counts broken down by source × data_quality.

    Output format::

        === Data Quality Summary ===
                            gold   silver  original_en   total
        source
        detox                  0    75000            0   75000
        gmhp7k              7000        0            0    7000
        ...
        TOTAL               XXXX     XXXX         XXXX    XXXX

          gold              XXXX rows  (XX.X%)
          silver            XXXX rows  (XX.X%)
          original_en       XXXX rows  (XX.X%)
          TOTAL             XXXX rows

    Parameters
    ----------
    df:
        DataFrame that has already been processed by
        ``add_data_quality_metadata()``.  Must contain columns
        ``source`` and ``data_quality``.
    """
    if "data_quality" not in df.columns:
        raise ValueError(
            "Column 'data_quality' not found. "
            "Run add_data_quality_metadata(df) before calling print_quality_summary()."
        )

    col_order = ["gold", "silver", "original_en"]

    pivot = (
        df.groupby(["source", "data_quality"], observed=True)
        .size()
        .rename("n_rows")
        .reset_index()
        .pivot(index="source", columns="data_quality", values="n_rows")
        .fillna(0)
        .astype(int)
    )

    # Ensure all three quality columns exist even if a category is absent
    for col in col_order:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[col_order]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_index()

    # Totals row
    totals = pivot.sum().rename("TOTAL")
    summary = pd.concat([pivot, totals.to_frame().T])

    print("\n=== Data Quality Summary ===")
    print(summary.to_string())
    print()

    total_all = int(totals["total"])
    for cat in col_order:
        n = int(totals[cat])
        pct = 100 * n / total_all if total_all > 0 else 0.0
        print(f"  {cat:<15} {n:>8,} rows  ({pct:.1f}%)")
    print(f"  {'TOTAL':<15} {total_all:>8,} rows")
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"source", "is_translated", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"add_data_quality_metadata() requires columns {required}. "
            f"Missing: {missing}"
        )


# ---------------------------------------------------------------------------
# Correctness assertions  (also used as a smoke-test when run directly)
# ---------------------------------------------------------------------------

def assert_quality_invariants(df: pd.DataFrame) -> None:
    """
    Assert four invariants that must hold after ``add_data_quality_metadata()``.

    Raises ``AssertionError`` with a descriptive message if any invariant
    is violated.  Intended for use in unit tests and pipeline smoke-tests.

    Parameters
    ----------
    df:
        DataFrame after ``add_data_quality_metadata()`` has been applied.

    Invariants checked
    ------------------
    1. Every row whose source is in GOLD_SOURCES has ``is_gold == True``.
    2. Every row whose source is NOT in GOLD_SOURCES has ``is_gold == False``.
    3. No row where ``is_translated == True`` still has ``language == 'en'``.
    4. Every row has a ``data_quality`` value in {'gold', 'silver', 'original_en'}.
    """
    # 1. All gold sources have is_gold=True
    gold_mask = df["source"].isin(GOLD_SOURCES)
    bad_gold = df.loc[gold_mask & ~df["is_gold"], "source"].unique()
    assert len(bad_gold) == 0, (
        f"is_gold=False for gold sources: {bad_gold}"
    )

    # 2. All non-gold sources have is_gold=False
    bad_silver = df.loc[~gold_mask & df["is_gold"], "source"].unique()
    assert len(bad_silver) == 0, (
        f"is_gold=True for non-gold sources: {bad_silver}"
    )

    # 3. No translated row still has language='en'
    translated_mask = df["is_translated"] == True  # noqa: E712
    still_english = df.loc[translated_mask & (df["language"] == "en")]
    assert len(still_english) == 0, (
        f"{len(still_english)} translated rows still have language='en'. "
        f"Sources affected: {still_english['source'].unique()}"
    )

    # 4. All rows have a valid data_quality value
    invalid_quality = df.loc[~df["data_quality"].isin(_VALID_QUALITY_VALUES)]
    assert len(invalid_quality) == 0, (
        f"{len(invalid_quality)} rows have invalid data_quality values: "
        f"{invalid_quality['data_quality'].unique()}"
    )