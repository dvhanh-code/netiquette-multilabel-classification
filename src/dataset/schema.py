"""
Unified label schema shared across all corpora.

Each label is stored as float: 1.0 = positive, 0.0 = negative, NaN = not annotated.
NaN means the corpus simply does not provide that label — useful for multi-task
learning where loss is computed only on annotated labels.

Schema version: 4-label schema covering harmful/abusive language categories.

Merge decisions applied at the loader level:
    misogyny      → hate_speech  (GMHP7k: OR-merge hs | m_hs)
    identity_hate → hate_speech  (Jigsaw: identity_hate column mapped to hate_speech)
    severe_toxic  → toxic        (Jigsaw: OR-merge toxic | severe_toxic)
    attack        → insult       (Wikipedia Attacks: attack annotation used as insult)
    obscene       → dropped      (Jigsaw only; profanity ≠ harm; dropped entirely)
    impolite      → dropped      (Wikipedia Politeness; politeness is a different
                                  theoretical framework, not an abuse category)
"""
from typing import Dict, List

# Canonical label names (4-label schema: harmful/abusive language only)
ALL_LABELS: List[str] = [
    "hate_speech",  # group-targeted hate speech (absorbs: misogyny, identity_hate)
    "toxic",        # general toxicity (absorbs: severe_toxic)
    "threat",       # explicit threatening language
    "insult",       # insults and personal attacks (absorbs: attack)
]

# Which labels each corpus contributes after merge rules are applied (others are NaN)
CORPUS_LABELS: Dict[str, List[str]] = {
    "gmhp7k":            ["hate_speech"],
    "hocon34k":          ["hate_speech"],
    "detox":             ["toxic"],
    "jigsaw":            ["hate_speech", "toxic", "threat", "insult"],
    "wikipedia_attacks": ["insult"],
    "gutefrage":         ["hate_speech", "insult", "toxic"],
    "rp_mod":            ["hate_speech", "threat", "insult"],
}

# Full column order of the unified DataFrame (loader contract — do not add
# pipeline-level columns here, as loaders must satisfy this schema exactly)
SCHEMA_COLUMNS: List[str] = ["text", "source", "language", "split"] + ALL_LABELS

# Columns present in the final saved parquet after quality metadata is added.
# Extends SCHEMA_COLUMNS with the two quality columns produced by
# src.preprocessing.quality.add_data_quality_metadata().
OUTPUT_COLUMNS: List[str] = SCHEMA_COLUMNS + ["is_gold", "data_quality"]
