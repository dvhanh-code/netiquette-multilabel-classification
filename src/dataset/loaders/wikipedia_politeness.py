"""
Loader for Wikipedia Politeness corpus.

Source: data/raw/politeness/wikipedia-politeness-corpus/utterances.jsonl

Each line is a JSON object with:
    text            — utterance text
    meta.Binary     — politeness label: 1 = polite, -1 = impolite, 0 = neutral

Labels provided: impolite
    1.0  → Binary == -1  (impolite)
    0.0  → Binary ==  1  (polite)
    NaN  → Binary ==  0  (neutral / borderline — excluded from loss)

Language: en
No predefined train/test split in this corpus (split = None).
"""
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..base import BaseCorpusLoader

_BINARY_TO_IMPOLITE = {1: 0.0, -1: 1.0, 0: np.nan}


class WikipediaPolitenessLoader(BaseCorpusLoader):
    SOURCE = "wikipedia_politeness"
    LANGUAGE = "en"

    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        jsonl_path = (
            Path(data_dir)
            / "politeness"
            / "wikipedia-politeness-corpus"
            / "utterances.jsonl"
        )

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                binary = obj.get("meta", {}).get("Binary")
                records.append({
                    "text":     obj["text"],
                    "impolite": _BINARY_TO_IMPOLITE.get(binary, np.nan),
                })

        df = pd.DataFrame(records)

        return self._make_frame(
            texts=df["text"],
            splits=None,
            label_dict={"impolite": df["impolite"]},
        )
