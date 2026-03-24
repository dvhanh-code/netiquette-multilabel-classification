"""
Loader for gutefrage.net moderation dataset.

Source: data/raw/gutefrage.net/dataset/datensatz_moderiert.numbers
Format: Apple Numbers file (19,045 rows — questions, answers, comments)
Language: de

Important design note
---------------------
gutefrage is a moderation dataset, not a pure abuse-annotation dataset.
Therefore, moderation decisions must not be naively converted into abuse labels.

This loader uses conservative mappings:
- only strong, semantically defensible moderation reasons are mapped
- rejection state alone is NOT used as a toxicity label
- defamation/libel reasons are NOT mapped to hate_speech

Columns used
------------
body             — main text (fallback: question_title if body is missing)
question_title   — fallback text source
moderation_state — kept as raw moderation metadata, not used directly as a label
deletion_reason  — moderator decision reason (primary signal)
notice_reason    — community report reason (secondary signal)

Universal label mapping
-----------------------
hate_speech:
    - "VolksverhetzungStGB130"
      Rationale: §130 StGB is the direct German statutory analog for hate speech
      (incitement to hatred against population groups). Strong mapping.

insult:
    - "Nutzervorführung / Persönlicher Angriff"
      Rationale: interpersonal targeting/attacking behaviour. The former "attack"
      label has been merged into "insult" under the reduced schema.
    - "BeleidigungStGB185"
      Rationale: §185 StGB is the direct German statutory analog for insult.

toxic:
    - "VolksverhetzungStGB130"
    - "Nutzervorführung / Persönlicher Angriff"
    - "BeleidigungStGB185"
    - "Feindseligkeit gegenüber Dritten"
      Rationale: hostility toward others is toxic by definition, even without
      a specific target group. Mapped conservatively to toxic only.

Explicitly NOT mapped to any universal label
--------------------------------------------
The following reasons are not mapped because they are closer to defamation/libel
(victim-specific false statements) than to group-targeted hate speech:
    - "UebleNachredeStGB186"  (§186 StGB: defamation — false fact about a person)
    - "VerleumdungStGB187"    (§187 StGB: slander — knowing false statement)
These are preserved in _UNMAPPED_PLATFORM_REASONS for future hybrid experiments.

Provenance
----------
Raw moderation fields (moderation_state, deletion_reason, notice_reason) are not
part of the unified SCHEMA_COLUMNS and are therefore not returned by load().

Use load_with_provenance() to obtain a wider DataFrame that includes:
    moderation_state_raw, deletion_reason_raw, notice_reason_raw

This wider frame is NOT compatible with UnifiedCorpusDataset and should only be
used for corpus-specific analysis and future hybrid/rule-based experiments.

TODO (future schema extension):
    If BaseCorpusLoader._make_frame() is extended to support an optional
    metadata_dict parameter, these three raw columns should be passed through
    directly so all loaders can expose provenance in a unified way.

All other universal labels are NaN (not annotated by this corpus).

Rows where both body and question_title are null/empty are dropped.
No predefined train/test split — split = None.
"""
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from ..base import BaseCorpusLoader

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

# Conservative universal-label mapping from gutefrage moderation reasons.
# A single reason may map to multiple universal labels.
_REASON_TO_LABELS = {
    "VolksverhetzungStGB130":                  {"hate_speech", "toxic"},
    "Nutzervorführung / Persönlicher Angriff": {"insult", "toxic"},
    "BeleidigungStGB185":                      {"insult", "toxic"},
    "Feindseligkeit gegenüber Dritten":        {"toxic"},
}

# Reasons intentionally kept out of universal abuse labels.
# Preserved here to make the deliberate non-mapping explicit and auditable.
# They may be used later as platform-specific defamation signals.
_UNMAPPED_PLATFORM_REASONS = {
    "UebleNachredeStGB186",  # §186 StGB: defamation — false fact about a person
    "VerleumdungStGB187",  # §187 StGB: slander — knowing false statement
}

# Only these labels are emitted from this corpus.
_SUPPORTED_LABELS = ("hate_speech", "insult", "toxic")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_numbers(path: Path) -> pd.DataFrame:
    """Read an Apple Numbers file into a DataFrame."""
    import numbers_parser

    doc = numbers_parser.Document(str(path))
    table = doc.sheets[0].tables[0]
    rows = [[cell.value for cell in row] for row in table.iter_rows()]
    return pd.DataFrame(rows[1:], columns=rows[0])


def _normalize_reason_series(series: pd.Series) -> pd.Series:
    """Normalize reason values for matching: fill NaN → '', strip whitespace."""
    return series.fillna("").astype(str).str.strip()


def _normalize_text_series(series: pd.Series) -> pd.Series:
    """Normalize text values and remove placeholder-like strings."""
    normalized = series.where(series.notna(), other="")
    normalized = normalized.astype(str).str.strip()
    placeholder_values = {"", "None", "<NA>", "nan", "NaN"}
    return normalized.where(~normalized.isin(placeholder_values), other="")


def _label_from_reasons(
    deletion_reason: pd.Series,
    notice_reason: pd.Series,
    label: str,
) -> pd.Series:
    """
    Return a float label column for the requested universal label.

    deletion_reason is treated as the primary moderation signal.
    notice_reason is treated as a secondary signal.
    A row is positive (1.0) if either reason maps to the label.

    Critically: moderation_state is NOT consulted here. Rejection alone
    does not constitute evidence of any specific abuse type.
    """
    matching_reasons = {
        reason
        for reason, labels in _REASON_TO_LABELS.items()
        if label in labels
    }
    matches_deletion = deletion_reason.isin(matching_reasons)
    matches_notice = notice_reason.isin(matching_reasons)
    return (matches_deletion | matches_notice).astype(float)


def _extract_fields(
    raw: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Extract and normalize all fields used for label derivation and text.

    Returns:
        text              — final text column (body with question_title fallback)
        deletion_reason   — normalized deletion reason
        notice_reason     — normalized notice reason
        moderation_state  — normalized moderation state (raw provenance only)

    Rows where both body and question_title are empty after normalization are
    dropped. The returned Series are all aligned to the same index.
    """
    body = _normalize_text_series(raw["body"])
    title = _normalize_text_series(raw["question_title"])
    
    # body is primary; fall back to question_title only if body is empty
    text = body.where(body != "", other=title)

    valid = text != ""
    raw = raw.loc[valid].reset_index(drop=True)
    text = text.loc[valid].reset_index(drop=True)

    deletion_reason = _normalize_reason_series(raw["deletion_reason"])
    notice_reason = _normalize_reason_series(raw["notice_reason"])
    moderation_state = _normalize_reason_series(raw["moderation_state"])

    return text, deletion_reason, notice_reason, moderation_state


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class GutefragLoader(BaseCorpusLoader):
    SOURCE = "gutefrage"
    LANGUAGE = "de"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        
        """
        Load gutefrage.net corpus.

        Returns a DataFrame with exactly SCHEMA_COLUMNS — compatible with
        UnifiedCorpusDataset. Raw moderation fields are not included here.
        Use load_with_provenance() for corpus-specific analysis.
        """
        raw = _load_numbers(self._numbers_path(data_dir))
        text, deletion_reason, notice_reason, _ = _extract_fields(raw)

        label_dict = {
            label: _label_from_reasons(deletion_reason, notice_reason, label)
            for label in _SUPPORTED_LABELS
        }
        return self._make_frame(texts=text, splits=None, label_dict=label_dict)

    def load_with_provenance(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Extended load that appends raw moderation columns to the standard frame.

        Extra columns (beyond SCHEMA_COLUMNS):
            moderation_state_raw  — raw moderation outcome (e.g. "Rejected (Final)")
            deletion_reason_raw   — reason assigned by the platform moderator
            notice_reason_raw     — reason flagged by the community

        WARNING: The returned DataFrame is wider than SCHEMA_COLUMNS and is NOT
        compatible with UnifiedCorpusDataset.load(). Use it only for:
            - corpus-specific analysis
            - hybrid / rule-based experiments
            - auditing label assignments
        """
        raw = _load_numbers(self._numbers_path(data_dir))
        text, deletion_reason, notice_reason, moderation_state = _extract_fields(raw)

        label_dict = {
            label: _label_from_reasons(deletion_reason, notice_reason, label)
            for label in _SUPPORTED_LABELS
        }
        df = self._make_frame(texts=text, splits=None, label_dict=label_dict)
        
        # Append raw provenance columns after _make_frame() so the schema
        # contract (SCHEMA_COLUMNS) is satisfied first.
        df = df.copy()
        df["moderation_state_raw"] = moderation_state.values
        df["deletion_reason_raw"] = deletion_reason.values
        df["notice_reason_raw"] = notice_reason.values
        return df

    # ── Private helpers ────────────────────────────────────────────────────
    
    @staticmethod
    def _numbers_path(data_dir: Union[str, Path]) -> Path:
        return (
            Path(data_dir)
            / "gutefrage.net"
            / "dataset"
            / "datensatz_moderiert.numbers"
        )
