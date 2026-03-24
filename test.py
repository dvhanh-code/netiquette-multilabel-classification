"""
Run all tests:
    python -m pytest test.py -v

Run a specific test:
    python -m pytest test.py::TestGermanToEnglishTranslator -v

Run without downloading the translation model (fast, offline):
    python -m pytest test.py -v -m "not slow"
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.dataset.schema import ALL_LABELS, CORPUS_LABELS, SCHEMA_COLUMNS
from src.dataset.loaders.gmhp7k import GMHP7kLoader
from src.dataset.loaders.jigsaw import JigsawLoader
from src.dataset.loaders.wikipedia_attacks import WikipediaAttacksLoader
from src.dataset.loaders.gutefrage import GutefragLoader
from src.preprocessing.translate import GermanToEnglishTranslator, _md5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal unified DataFrame from a list of row dicts."""
    base = {col: None for col in SCHEMA_COLUMNS}
    records = [{**base, **row} for row in rows]
    return pd.DataFrame(records)[SCHEMA_COLUMNS]


def _german_df() -> pd.DataFrame:
    """Minimal multilingual DataFrame for translator tests.

    Uses only labels that exist in the current 5-label schema.
    """
    return _make_df([
        {"text": "Das ist eine normale Aussage.", "language": "de", "source": "gmhp7k",  "hate_speech": 0.0},
        {"text": "Frauen gehören in die Küche.",  "language": "de", "source": "gmhp7k",  "hate_speech": 1.0},
        {"text": "Geh scheissen.",                "language": "de", "source": "hocon34k", "hate_speech": 0.0},
        {"text": "This is already in English.",   "language": "en", "source": "jigsaw",   "toxic": 0.0},
    ])


def _gutefrage_raw(**kwargs) -> pd.DataFrame:
    """Minimal raw gutefrage row with sensible defaults."""
    defaults = {
        "content_item_type": "Answer",
        "question_type":     None,
        "moderation_state":  "Accepted",
        "question_title":    None,
        "body":              "Normaler Beitrag.",
        "deletion_reason":   None,
        "notice_reason":     None,
    }
    return pd.DataFrame([{**defaults, **kwargs}])


def _jigsaw_dirs(tmp_path: Path, train_rows: list[dict]) -> Path:
    """
    Write minimal Jigsaw CSV files to tmp_path/jigsaw/ and return tmp_path.

    train_rows: list of dicts overriding the raw label defaults for train.csv.
    Creates empty (header-only) test.csv and test_labels.csv so the loader
    does not crash when it tries to open them.
    """
    jigsaw_dir = tmp_path / "jigsaw"
    jigsaw_dir.mkdir(parents=True, exist_ok=True)

    raw_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    defaults = {"comment_text": "sample text", **{c: 0 for c in raw_cols}}
    records  = [{**defaults, **r} for r in train_rows]
    pd.DataFrame(records).to_csv(jigsaw_dir / "train.csv", index=False)

    # Empty test files — loader requires them to exist
    pd.DataFrame(columns=["id", "comment_text"]).to_csv(
        jigsaw_dir / "test.csv", index=False
    )
    pd.DataFrame(columns=["id"] + raw_cols).to_csv(
        jigsaw_dir / "test_labels.csv", index=False
    )
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_all_labels_not_empty(self):
        assert len(ALL_LABELS) > 0

    def test_reduced_label_count(self):
        """Schema reduction: exactly 5 training labels."""
        assert len(ALL_LABELS) == 5

    def test_expected_labels_present(self):
        assert set(ALL_LABELS) == {"hate_speech", "toxic", "threat", "insult", "impolite"}

    def test_removed_labels_absent(self):
        """Labels removed in the schema reduction must not reappear."""
        removed = {"misogyny", "severe_toxic", "obscene", "identity_hate", "attack"}
        present = removed & set(ALL_LABELS)
        assert present == set(), f"Removed labels still in schema: {present}"

    def test_schema_columns_order(self):
        assert SCHEMA_COLUMNS[:4] == ["text", "source", "language", "split"]
        assert SCHEMA_COLUMNS[4:] == ALL_LABELS

    def test_corpus_labels_are_subset_of_all_labels(self):
        for corpus, labels in CORPUS_LABELS.items():
            unknown = set(labels) - set(ALL_LABELS)
            assert unknown == set(), f"{corpus} uses unknown labels: {unknown}"

    def test_all_corpora_present(self):
        expected = {
            "gmhp7k", "hocon34k", "detox", "jigsaw",
            "wikipedia_attacks", "wikipedia_politeness", "gutefrage",
        }
        assert expected == set(CORPUS_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Merge-rule tests — verify that each loader applies the reduction correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeRules:
    """
    Verify that every schema-reduction merge is correctly implemented
    in the affected loaders. Uses temporary files; no mock data leaks
    loader internals.
    """

    # ── GMHP7k: misogyny OR hate_speech → hate_speech ─────────────────────

    def test_gmhp7k_misogyny_only_produces_hate_speech(self, tmp_path):
        """hs=0 but m_hs=1 (majority): must produce hate_speech=1 via OR-merge."""
        csv_dir = tmp_path / "corpus-gmhp7k" / "datasets"
        csv_dir.mkdir(parents=True)
        pd.DataFrame([
            {"tweet_id": 1, "review_text": "Frauen sind dumm.", "hs": 0.0, "m_hs": 1.0},
        ]).to_csv(csv_dir / "misogynistic_hatespeech_phase1.csv", index=False)

        out = GMHP7kLoader().load(tmp_path)
        assert out["hate_speech"].iloc[0] == 1.0

    def test_gmhp7k_both_signals_produce_hate_speech(self, tmp_path):
        """hs=1 and m_hs=1: OR-merge still produces hate_speech=1."""
        csv_dir = tmp_path / "corpus-gmhp7k" / "datasets"
        csv_dir.mkdir(parents=True)
        pd.DataFrame([
            {"tweet_id": 1, "review_text": "Hass.", "hs": 1.0, "m_hs": 1.0},
        ]).to_csv(csv_dir / "misogynistic_hatespeech_phase1.csv", index=False)

        out = GMHP7kLoader().load(tmp_path)
        assert out["hate_speech"].iloc[0] == 1.0

    def test_gmhp7k_neither_signal_produces_hate_speech_zero(self, tmp_path):
        csv_dir = tmp_path / "corpus-gmhp7k" / "datasets"
        csv_dir.mkdir(parents=True)
        pd.DataFrame([
            {"tweet_id": 1, "review_text": "Normal.", "hs": 0.0, "m_hs": 0.0},
        ]).to_csv(csv_dir / "misogynistic_hatespeech_phase1.csv", index=False)

        out = GMHP7kLoader().load(tmp_path)
        assert out["hate_speech"].iloc[0] == 0.0

    def test_gmhp7k_misogyny_not_in_output_columns(self, tmp_path):
        """The 'misogyny' column must not appear in the loader output."""
        csv_dir = tmp_path / "corpus-gmhp7k" / "datasets"
        csv_dir.mkdir(parents=True)
        pd.DataFrame([
            {"tweet_id": 1, "review_text": "Text.", "hs": 0.0, "m_hs": 1.0},
        ]).to_csv(csv_dir / "misogynistic_hatespeech_phase1.csv", index=False)

        out = GMHP7kLoader().load(tmp_path)
        assert "misogyny" not in out.columns
        assert list(out.columns) == SCHEMA_COLUMNS

    # ── Jigsaw: identity_hate → hate_speech ───────────────────────────────

    def test_jigsaw_identity_hate_becomes_hate_speech(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "you people are inferior", "identity_hate": 1},
        ]))
        assert out["hate_speech"].iloc[0] == 1.0

    def test_jigsaw_no_identity_hate_gives_hate_speech_zero(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "normal text", "identity_hate": 0},
        ]))
        assert out["hate_speech"].iloc[0] == 0.0

    def test_jigsaw_identity_hate_not_in_output_columns(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "text", "identity_hate": 1},
        ]))
        assert "identity_hate" not in out.columns

    # ── Jigsaw: severe_toxic → toxic ──────────────────────────────────────

    def test_jigsaw_severe_toxic_contributes_to_toxic(self, tmp_path):
        """severe_toxic=1 with toxic=0: OR-merge must still give toxic=1."""
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "extremely bad", "toxic": 0, "severe_toxic": 1},
        ]))
        assert out["toxic"].iloc[0] == 1.0

    def test_jigsaw_severe_toxic_not_in_output_columns(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "text", "severe_toxic": 1},
        ]))
        assert "severe_toxic" not in out.columns

    # ── Jigsaw: obscene dropped ───────────────────────────────────────────

    def test_jigsaw_obscene_not_in_output_columns(self, tmp_path):
        """obscene is dropped from the schema; must not appear in output."""
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "profanity", "obscene": 1},
        ]))
        assert "obscene" not in out.columns
        assert list(out.columns) == SCHEMA_COLUMNS

    def test_jigsaw_threat_kept_as_is(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "I will hurt you", "threat": 1},
        ]))
        assert out["threat"].iloc[0] == 1.0

    def test_jigsaw_insult_kept_as_is(self, tmp_path):
        out = JigsawLoader().load(_jigsaw_dirs(tmp_path, [
            {"comment_text": "you idiot", "insult": 1},
        ]))
        assert out["insult"].iloc[0] == 1.0

    # ── Wikipedia Attacks: attack → insult ────────────────────────────────

    def _write_attacks(self, tmp_path: Path, attack_score: float) -> Path:
        d = tmp_path / "wikipedia_attacks"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"rev_id": 1, "comment": "some comment", "split": "train"}]).to_csv(
            d / "attack_annotated_comments.csv", index=False
        )
        pd.DataFrame([
            {"rev_id": 1, "worker_id": 1, "attack": attack_score},
            {"rev_id": 1, "worker_id": 2, "attack": attack_score},
        ]).to_csv(d / "attack_annotations.csv", index=False)
        return tmp_path

    def test_wikipedia_attacks_positive_becomes_insult(self, tmp_path):
        out = WikipediaAttacksLoader().load(self._write_attacks(tmp_path, 1.0))
        assert out["insult"].iloc[0] == 1.0

    def test_wikipedia_attacks_negative_gives_insult_zero(self, tmp_path):
        out = WikipediaAttacksLoader().load(self._write_attacks(tmp_path, 0.0))
        assert out["insult"].iloc[0] == 0.0

    def test_wikipedia_attacks_attack_not_in_output_columns(self, tmp_path):
        """'attack' column must not appear — it was merged into 'insult'."""
        out = WikipediaAttacksLoader().load(self._write_attacks(tmp_path, 1.0))
        assert "attack" not in out.columns
        assert list(out.columns) == SCHEMA_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Translator — unit tests (no model download)
# ─────────────────────────────────────────────────────────────────────────────

class TestGermanToEnglishTranslator:

    def test_no_german_rows_returns_copy(self):
        df = _make_df([
            {"text": "Hello world", "language": "en"},
            {"text": "Bonjour",     "language": "fr"},
        ])
        t = GermanToEnglishTranslator(cache_path=None)
        out = t.translate(df)
        pd.testing.assert_frame_equal(df, out)
        assert out is not df  # must be a copy

    def test_output_preserves_schema_columns(self):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = f"[TRANSLATED] {text}"
        out = t.translate(df)
        assert list(out.columns) == SCHEMA_COLUMNS

    def test_german_rows_language_updated_to_en(self):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = f"translated: {text}"
        out = t.translate(df)
        assert (out["language"] == "de").sum() == 0
        assert (out["language"] == "en").sum() == len(out)

    def test_non_german_text_unchanged(self):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = "some translation"
        out = t.translate(df)
        english_orig = df.loc[df["language"] == "en", "text"].iloc[0]
        english_out  = out.loc[out["source"] == "jigsaw", "text"].iloc[0]
        assert english_orig == english_out

    def test_labels_preserved_after_translation(self):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = "translated"
        out = t.translate(df)
        pd.testing.assert_series_equal(df["hate_speech"], out["hate_speech"])

    def test_row_count_unchanged(self):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = "translated"
        out = t.translate(df)
        assert len(out) == len(df)

    def test_cache_hit_skips_model_call(self, tmp_path):
        df = _german_df()
        t = GermanToEnglishTranslator(cache_path=None)
        for text in df.loc[df["language"] == "de", "text"]:
            t._cache[_md5(text)] = "cached translation"
        out = t.translate(df)
        assert (out["language"] == "en").all()

    def test_cache_persisted_and_reloaded(self, tmp_path):
        cache_file = tmp_path / "trans.parquet"
        df = _make_df([{"text": "Hallo Welt", "language": "de"}])

        t1 = GermanToEnglishTranslator(cache_path=cache_file)
        t1._cache[_md5("Hallo Welt")] = "Hello World"
        t1._save_cache()
        assert cache_file.exists()

        t2 = GermanToEnglishTranslator(cache_path=cache_file)
        t2._load_cache()
        assert _md5("Hallo Welt") in t2._cache
        assert t2._cache[_md5("Hallo Welt")] == "Hello World"

    def test_md5_deterministic(self):
        assert _md5("test") == _md5("test")
        assert _md5("a") != _md5("b")

    def test_default_device_returns_string(self):
        device = GermanToEnglishTranslator._default_device()
        assert device in ("cpu", "cuda", "mps")


# ─────────────────────────────────────────────────────────────────────────────
# Translator — integration test (downloads + runs the real model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestTranslatorIntegration:

    def test_real_translation_quality(self, tmp_path):
        samples = {
            "Das ist ein Test.":               "test",
            "Ich hasse dich.":                 "hate",
            "Das Wetter ist heute sehr schön.": "weather",
            "Frauen gehören in die Küche.":    "kitchen",
        }
        df = _make_df([{"text": t, "language": "de"} for t in samples])
        t = GermanToEnglishTranslator(cache_path=tmp_path / "cache.parquet", batch_size=4)
        out = t.translate(df)

        assert (out["language"] == "en").all()
        assert out["text"].str.len().min() > 0

        kitchen_row = out[df["text"] == "Frauen gehören in die Küche."]
        assert "kitchen" in kitchen_row["text"].iloc[0].lower()

    def test_cache_used_on_second_run(self, tmp_path):
        df = _make_df([{"text": "Guten Morgen.", "language": "de"}])
        cache = tmp_path / "cache.parquet"

        t1 = GermanToEnglishTranslator(cache_path=cache, batch_size=1)
        out1 = t1.translate(df)
        assert cache.exists()

        t2 = GermanToEnglishTranslator(cache_path=cache, batch_size=1)
        t2._load_cache()
        assert _md5("Guten Morgen.") in t2._cache
        out2 = t2.translate(df)
        assert out1["text"].iloc[0] == out2["text"].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Gutefrage loader — unit tests (no file I/O, monkeypatches _load_numbers)
# ─────────────────────────────────────────────────────────────────────────────

class TestGutefragLoader:
    """
    Unit tests for GutefragLoader.load().

    _load_numbers is monkeypatched so no file I/O occurs. The real public
    method GutefragLoader.load() is called, exercising the full loader path.
    """

    def _load(self, monkeypatch, raw: pd.DataFrame) -> pd.DataFrame:
        monkeypatch.setattr(
            "src.dataset.loaders.gutefrage._load_numbers",
            lambda _path: raw,
        )
        return GutefragLoader().load("dummy")

    def test_schema_columns_present(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw())
        assert list(out.columns) == SCHEMA_COLUMNS

    def test_no_extra_columns(self, monkeypatch):
        """load() must not leak raw provenance columns into the unified frame."""
        out = self._load(monkeypatch, _gutefrage_raw())
        for col in ("moderation_state_raw", "deletion_reason_raw", "notice_reason_raw"):
            assert col not in out.columns

    def test_source_and_language(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw())
        assert out["source"].iloc[0] == "gutefrage"
        assert out["language"].iloc[0] == "de"

    def test_split_is_none(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw())
        assert out["split"].iloc[0] is None

    def test_hate_speech_label_from_deletion_reason(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(deletion_reason="VolksverhetzungStGB130"))
        assert out["hate_speech"].iloc[0] == 1.0
        assert out["toxic"].iloc[0]       == 1.0

    def test_hate_speech_label_from_notice_reason(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(notice_reason="VolksverhetzungStGB130"))
        assert out["hate_speech"].iloc[0] == 1.0
        assert out["toxic"].iloc[0]       == 1.0

    def test_personal_attack_reason_maps_to_insult_and_toxic(self, monkeypatch):
        """Nutzervorführung maps to insult + toxic; 'attack' no longer a column."""
        out = self._load(monkeypatch, _gutefrage_raw(
            deletion_reason="Nutzervorführung / Persönlicher Angriff"
        ))
        assert out["insult"].iloc[0] == 1.0
        assert out["toxic"].iloc[0]  == 1.0
        assert "attack" not in out.columns

    def test_insult_label(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(notice_reason="BeleidigungStGB185"))
        assert out["insult"].iloc[0] == 1.0
        assert out["toxic"].iloc[0]  == 1.0

    def test_feindseligkeit_maps_to_toxic_not_hate_speech(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(
            deletion_reason="Feindseligkeit gegenüber Dritten"
        ))
        assert out["toxic"].iloc[0]       == 1.0
        assert out["hate_speech"].iloc[0] == 0.0

    def test_rejected_state_alone_is_not_toxic(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(moderation_state="Rejected"))
        assert out["toxic"].iloc[0] == 0.0

    def test_rejected_final_state_alone_is_not_toxic(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(moderation_state="Rejected (Final)"))
        assert out["toxic"].iloc[0] == 0.0

    def test_accepted_post_is_not_toxic(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(moderation_state="Accepted"))
        assert out["toxic"].iloc[0] == 0.0

    def test_defamation_reason_produces_no_labels(self, monkeypatch):
        """UebleNachredeStGB186 must not trigger any universal label."""
        out = self._load(monkeypatch, _gutefrage_raw(deletion_reason="UebleNachredeStGB186"))
        assert out["hate_speech"].iloc[0] == 0.0
        assert out["toxic"].iloc[0]       == 0.0
        assert out["insult"].iloc[0]      == 0.0

    def test_libel_reason_produces_no_labels(self, monkeypatch):
        """VerleumdungStGB187 must not trigger any universal label."""
        out = self._load(monkeypatch, _gutefrage_raw(deletion_reason="VerleumdungStGB187"))
        assert out["hate_speech"].iloc[0] == 0.0
        assert out["toxic"].iloc[0]       == 0.0
        assert out["insult"].iloc[0]      == 0.0

    def test_no_reason_means_no_labels(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(deletion_reason=None, notice_reason=None))
        for label in ("hate_speech", "insult", "toxic"):
            assert out[label].iloc[0] == 0.0

    def test_body_fallback_to_question_title(self, monkeypatch):
        out = self._load(monkeypatch, _gutefrage_raw(body=None, question_title="Frage ohne Body"))
        assert out["text"].iloc[0] == "Frage ohne Body"

    def test_rows_with_no_text_are_dropped(self, monkeypatch):
        raw = pd.concat([
            _gutefrage_raw(body="Gültiger Text"),
            _gutefrage_raw(body=None, question_title=None),
        ], ignore_index=True)
        out = self._load(monkeypatch, raw)
        assert len(out) == 1

    def test_placeholder_text_rows_are_dropped(self, monkeypatch):
        raw = pd.concat([
            _gutefrage_raw(body="None",          question_title=None),
            _gutefrage_raw(body="<NA>",          question_title=None),
            _gutefrage_raw(body="   ",           question_title=None),
            _gutefrage_raw(body="Gültiger Text", question_title=None),
        ], ignore_index=True)
        out = self._load(monkeypatch, raw)
        assert len(out) == 1
        assert out["text"].iloc[0] == "Gültiger Text"

    def test_multilabel_same_row(self, monkeypatch):
        """A post can be flagged for hate_speech and insult simultaneously."""
        out = self._load(monkeypatch, _gutefrage_raw(
            deletion_reason="VolksverhetzungStGB130",
            notice_reason="Nutzervorführung / Persönlicher Angriff",
            moderation_state="Accepted",
        ))
        assert out["hate_speech"].iloc[0] == 1.0
        assert out["insult"].iloc[0]      == 1.0
        assert out["toxic"].iloc[0]       == 1.0

    def test_non_gutefrage_labels_are_nan(self, monkeypatch):
        """Labels not annotated by gutefrage in the reduced schema must be NaN."""
        out = self._load(monkeypatch, _gutefrage_raw())
        # gutefrage annotates: hate_speech, insult, toxic
        # not annotated: threat, impolite
        for label in ("threat", "impolite"):
            assert np.isnan(out[label].iloc[0]), f"{label} should be NaN"


# ─────────────────────────────────────────────────────────────────────────────
# Gutefrage loader — load_with_provenance() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGutefragLoaderProvenance:
    """
    Unit tests for GutefragLoader.load_with_provenance().

    Uses the same monkeypatch strategy as TestGutefragLoader so the real
    public method is exercised end-to-end without any file I/O.
    """

    def _load_prov(self, monkeypatch, raw: pd.DataFrame) -> pd.DataFrame:
        monkeypatch.setattr(
            "src.dataset.loaders.gutefrage._load_numbers",
            lambda _path: raw,
        )
        return GutefragLoader().load_with_provenance("dummy")

    def test_schema_columns_still_present(self, monkeypatch):
        out = self._load_prov(monkeypatch, _gutefrage_raw())
        for col in SCHEMA_COLUMNS:
            assert col in out.columns, f"Missing schema column: {col}"

    def test_provenance_columns_present(self, monkeypatch):
        out = self._load_prov(monkeypatch, _gutefrage_raw())
        assert "moderation_state_raw" in out.columns
        assert "deletion_reason_raw"  in out.columns
        assert "notice_reason_raw"    in out.columns

    def test_moderation_state_raw_value(self, monkeypatch):
        out = self._load_prov(monkeypatch, _gutefrage_raw(moderation_state="Rejected (Final)"))
        assert out["moderation_state_raw"].iloc[0] == "Rejected (Final)"

    def test_deletion_reason_raw_value(self, monkeypatch):
        out = self._load_prov(monkeypatch, _gutefrage_raw(deletion_reason="UebleNachredeStGB186"))
        assert out["deletion_reason_raw"].iloc[0] == "UebleNachredeStGB186"

    def test_notice_reason_raw_value(self, monkeypatch):
        out = self._load_prov(monkeypatch, _gutefrage_raw(notice_reason="VerleumdungStGB187"))
        assert out["notice_reason_raw"].iloc[0] == "VerleumdungStGB187"

    def test_unmapped_reason_preserved_but_no_label(self, monkeypatch):
        """
        UebleNachredeStGB186 must appear in deletion_reason_raw but must not
        trigger any universal label.
        """
        out = self._load_prov(monkeypatch, _gutefrage_raw(deletion_reason="UebleNachredeStGB186"))
        assert out["deletion_reason_raw"].iloc[0] == "UebleNachredeStGB186"
        assert out["hate_speech"].iloc[0] == 0.0
        assert out["toxic"].iloc[0]       == 0.0
        assert out["insult"].iloc[0]      == 0.0

    def test_rejected_state_preserved_but_no_label(self, monkeypatch):
        """
        A Rejected post must not produce any abuse label, but the raw state
        must still appear in moderation_state_raw for future analysis.
        """
        out = self._load_prov(monkeypatch, _gutefrage_raw(moderation_state="Rejected"))
        assert out["moderation_state_raw"].iloc[0] == "Rejected"
        assert out["toxic"].iloc[0]                == 0.0
        assert out["hate_speech"].iloc[0]          == 0.0

    def test_null_reasons_stored_as_empty_string(self, monkeypatch):
        """Null reason fields must normalize to '' in provenance columns, not 'None'."""
        out = self._load_prov(monkeypatch, _gutefrage_raw(deletion_reason=None, notice_reason=None))
        assert out["deletion_reason_raw"].iloc[0] == ""
        assert out["notice_reason_raw"].iloc[0]   == ""


# ─────────────────────────────────────────────────────────────────────────────
# Gutefrage loader — integration tests (slow: reads real .numbers file)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestGutefragLoaderIntegration:

    def test_load_real_file(self):
        data_dir = Path(__file__).parent / "data" / "raw"
        df = GutefragLoader().load(data_dir)

        assert len(df) > 10_000
        assert list(df.columns) == SCHEMA_COLUMNS
        assert (df["source"] == "gutefrage").all()
        assert (df["language"] == "de").all()
        assert df["text"].notna().all()
        assert df["text"].str.strip().ne("").all()

        # gutefrage annotates these three labels; expect at least some positives
        for label in ("hate_speech", "insult", "toxic"):
            assert df[label].sum() > 0, f"Expected positives for {label}"

        # Labels not annotated by gutefrage in the reduced schema must be NaN
        for label in ("threat", "impolite"):
            assert df[label].isna().all(), f"{label} should be all NaN"

    def test_load_with_provenance_real_file(self):
        data_dir = Path(__file__).parent / "data" / "raw"
        df = GutefragLoader().load_with_provenance(data_dir)

        for col in SCHEMA_COLUMNS:
            assert col in df.columns, f"Missing schema column: {col}"

        for col in ("moderation_state_raw", "deletion_reason_raw", "notice_reason_raw"):
            assert col in df.columns, f"Missing provenance column: {col}"
            assert df[col].notna().all(), f"{col} must not contain NaN"

        # Rows where BOTH reasons are unmapped defamation reasons (or empty) must
        # produce no abuse labels. A row with an unmapped deletion_reason can still
        # get a positive label from notice_reason if that notice_reason is mapped.
        _unmapped_set = {"UebleNachredeStGB186", "VerleumdungStGB187", ""}
        unmapped = df[
            df["deletion_reason_raw"].isin(_unmapped_set)
            & df["notice_reason_raw"].isin(_unmapped_set)
        ]
        if len(unmapped) > 0:
            for label in ("hate_speech", "toxic", "insult"):
                assert (unmapped[label] == 0.0).all(), (
                    f"Row with no mapped reasons must not produce {label}=1"
                )