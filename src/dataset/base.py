from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .schema import ALL_LABELS, SCHEMA_COLUMNS


class BaseCorpusLoader(ABC):
    SOURCE: str = ""
    LANGUAGE: str = ""

    @abstractmethod
    def load(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """Load corpus from data_dir and return a unified DataFrame."""

    def _make_frame(
        self,
        texts: pd.Series,
        splits: Optional[pd.Series],
        label_dict: dict,
    ) -> pd.DataFrame:
        """
        Assemble the unified DataFrame from raw arrays/Series.

        Args:
            texts:      Series of raw text strings.
            splits:     Series of split identifiers ('train'/'test'/'val') or None.
            label_dict: Mapping label_name -> Series | scalar.
                        Labels not present default to NaN (not annotated).
        """
        n = len(texts)

        splits_col = (
            pd.Series([None] * n, dtype=object)
            if splits is None
            else pd.Series(splits).reset_index(drop=True)
        )

        df = pd.DataFrame({
            "text":     pd.Series(texts).reset_index(drop=True),
            "source":   self.SOURCE,
            "language": self.LANGUAGE,
            "split":    splits_col,
        })

        for label in ALL_LABELS:
            val = label_dict.get(label, np.nan)
            df[label] = (
                pd.Series(val).reset_index(drop=True)
                if isinstance(val, pd.Series)
                else val
            )

        return df[SCHEMA_COLUMNS]
