"""
German → English translation using Helsinki-NLP/opus-mt-de-en (MarianMT).

Translates all rows with language == 'de' in the unified DataFrame,
updating the `text` and `language` columns in-place on a copy.

A disk cache (data/cache/translations_de_en.parquet) stores completed
translations keyed by MD5 hash, so repeated runs skip already-translated rows.

Usage
-----
    from src.preprocessing import GermanToEnglishTranslator

    translator = GermanToEnglishTranslator(cache_path="data/cache/translations_de_en.parquet")
    df_en = translator.translate(df)
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


class GermanToEnglishTranslator:
    """
    Translates German rows in a unified DataFrame to English.

    Args:
        cache_path: Path to a parquet file used to persist translations
                    across runs. Pass None to disable caching.
        batch_size: Number of texts per translation batch. Larger batches
                    are faster on GPU; reduce if you hit OOM errors.
        max_length: Max token length for both input and generated output.
                    MarianMT caps at 512; most short social-media texts are
                    well below this.
        device:     'cpu', 'cuda', or 'mps'. Defaults to auto-detect.
    """

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = device or self._default_device()
        self._cache: dict = {}
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of *df* with all German rows translated to English.

        The `text` column is replaced with the English translation;
        the `language` column is updated from 'de' to 'en'.
        """
        german_mask = df["language"] == "de"
        n_german = german_mask.sum()

        if n_german == 0:
            logger.info("No German rows found — nothing to translate.")
            return df.copy()

        self._load_cache()
        self._load_model()

        print(f"\nTranslating {n_german:,} German rows → English  (model: {MODEL_NAME})")

        german_texts: List[str] = df.loc[german_mask, "text"].tolist()
        hashes = [_md5(t) for t in german_texts]

        # Split into cache hits vs. texts that still need translation
        miss_positions = [i for i, h in enumerate(hashes) if h not in self._cache]
        n_hits = n_german - len(miss_positions)
        print(f"  Cache hits : {n_hits:,}  |  To translate : {len(miss_positions):,}")

        if miss_positions:
            texts_to_translate = [german_texts[i] for i in miss_positions]
            translations = self._translate_in_batches(texts_to_translate)
            for pos, translation in zip(miss_positions, translations):
                self._cache[hashes[pos]] = translation
            self._save_cache()

        # Write translations back into a copy of the DataFrame
        translated_texts = [self._cache[h] for h in hashes]
        df_out = df.copy()
        german_indices = df.index[german_mask]
        df_out.loc[german_indices, "text"] = translated_texts
        df_out.loc[german_indices, "language"] = "en"

        print(f"  Done. All {n_german:,} German rows are now in English.\n")
        return df_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_in_batches(self, texts: List[str]) -> List[str]:
        results = []
        batches = range(0, len(texts), self.batch_size)
        for start in tqdm(batches, desc="  Translating batches"):
            batch = texts[start : start + self.batch_size]
            results.extend(self._translate_batch(batch))
        return results

    def _translate_batch(self, texts: List[str]) -> List[str]:
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
            )

        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer

        print(f"  Loading model '{MODEL_NAME}' onto {self.device} …", flush=True)
        self._tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self._model = MarianMTModel.from_pretrained(MODEL_NAME).to(self.device)
        self._model.eval()
        print("  Model ready.")

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            cached = pd.read_parquet(self.cache_path)
            self._cache = dict(zip(cached["hash"], cached["translated"]))
            print(f"  Loaded {len(self._cache):,} cached translations from {self.cache_path}")

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "hash":       list(self._cache.keys()),
            "translated": list(self._cache.values()),
        }).to_parquet(self.cache_path, index=False)
        logger.debug("Cache saved (%d entries).", len(self._cache))

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
