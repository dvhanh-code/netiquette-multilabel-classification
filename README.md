# Netiquette Multilabel Classification

This repository contains the implementation for the Master's thesis *"Hybride
Multilabel-Klassifikation von Online-Kommunikationsverhalten bezüglich der
Netiquette"* (Vu Hoang Anh Dang, Hochschule München). It covers the data
harmonization pipeline, BERT-based classifiers, LLM-based classifiers, and
exploratory as well as final hybrid BERT–LLM combination strategies for
detecting netiquette violations in German-language online comments.

## 1. Project Overview

The project addresses multilabel classification of online communication
behavior into four target labels: `hate_speech`, `toxic`, `threat`, and
`insult`. A comment may belong to zero, one, or several categories at once.

Seven heterogeneous corpora — four authentic German ("gold") sources and
three English-origin ("silver", machine-translated) sources — are
harmonized into one label schema and frozen into a single dataset used for
all reported experiments. On top of this dataset, the project implements
and compares:

- A classical TF-IDF + Linear SVM baseline
- BERT-based transformer classifiers (`bert-base-german-cased`,
  `deepset/gbert-large`) trained with different loss functions
- LLM-based classifiers, both locally via Ollama (Qwen) and via the Gemini
  API
- Hybrid BERT–LLM combination strategies, first as exploratory score-level
  fusion, then as a threshold-aware, per-label routing approach

Models are evaluated with Precision, Recall, F1, F2, Matthews Correlation
Coefficient (MCC), and a composite S-Score (mean of F2 and normalized MCC,
following the HSD-Detector formulation used by Ippen Digital), with 95%
bootstrap confidence intervals where applicable.

## 2. Repository Structure

```
.
├── src/
│   ├── dataset/          # schema, base loader, unified dataset, corpus loaders, splits, deduplication
│   ├── preprocessing/     # EN→DE translation, gold/silver quality metadata
│   ├── training/          # baseline SVM, transformer dataset/metrics/losses, generic training loop
│   ├── evaluation/        # bootstrap CI, translation-quality (LaBSE) evaluation, comparison utilities
│   ├── llm/                # Ollama backend/engine/prompts + Gemini client/prompts/parsing, platform rules
│   └── hybrid/            # exploratory BERT–LLM score-fusion strategies
├── experiments/           # CLI scripts for LLM eval, prompt ablation, hybrid fusion, per-label hybrid routing
├── notebooks/              # exploratory EDA and original Colab-based BERT experiment development
├── data/                   # raw corpora, processed intermediates, frozen final dataset
├── results/                 # experiment artifacts (metrics, thresholds, predictions, logs)
├── run_pipeline.py         # loads corpora, optionally translates + assigns splits, saves an intermediate parquet
├── requirements.txt
└── README.md
```

`src/llm/` contains two largely independent inference stacks (Ollama and
Gemini). `src/hybrid/` contains the exploratory fusion strategies; the
final threshold-aware hybrid routing logic lives directly in
`experiments/run_perlabel_hybrid.py`.

## 3. Data and Label Schema

Final label schema: `hate_speech`, `toxic`, `threat`, `insult`. Each label
column is a float with three possible values:

- `1.0` = positive (label applies)
- `0.0` = negative (label does not apply)
- `NaN` = not annotated by the source corpus — **not** a negative label

Because the seven source corpora annotate different subsets of the four
labels (`src/dataset/schema.py`, `CORPUS_LABELS`), most rows have NaN for
labels their source corpus does not provide. A boolean label mask (derived
from non-NaN positions) is used throughout so missing annotations never
contribute to training loss (`src/training/losses.py`) or to reported
metrics (`src/evaluation/bootstrap_ci.py`,
`experiments/run_perlabel_hybrid.py`).

## 4. Datasets

Seven corpora are actively loaded and merged by `src/dataset/loaders/`
(`ALL_LOADERS`, `src/dataset/loaders/__init__.py`):

**Gold German sources** (authentic, human-authored German text with human
annotations; used for final evaluation):

- `gmhp7k` — GMHP7k (German hate speech)
- `hocon34k` — HOCON34k
- `gutefrage` — Gutefrage.net
- `rp_mod` — RP-Mod/RP-Crowd

**Silver, English-origin sources** (machine-translated to German; training
data only):

- `jigsaw` — Jigsaw Toxic Comment
- `detox` — Detox / Wikipedia Toxicity
- `wikipedia_attacks` — Wikipedia Personal Attacks

A `WikipediaPolitenessLoader` (`src/dataset/loaders/wikipedia_politeness.py`)
also exists but is intentionally excluded from `ALL_LOADERS` and from the
final four-label pipeline; it targets a politeness-spectrum framework
outside the harmful-language schema and is kept only as an archived,
optional loader.

## 5. Data Processing Pipeline

The frozen dataset used for all reported experiments,
`data/final/unified_final_v1.parquet`, was produced through:

1. **Corpus-specific loaders and label harmonization** (`src/dataset/loaders/`)
   — map each corpus's native scheme onto the four-label schema (e.g.
   misogyny → `hate_speech`, `severe_toxic` → `toxic`, `attack` → `insult`).
2. **English-to-German translation** of English-origin rows via
   `Helsinki-NLP/opus-mt-en-de` (MarianMT), with a disk cache
   (`src/preprocessing/translate.py`).
3. **Gold/Silver quality metadata** (`src/preprocessing/quality.py`): the
   four gold sources are tagged `data_quality="gold"`; translated English
   rows become `data_quality="silver"`.
4. **LaBSE semantic similarity evaluation** using
   `sentence-transformers/LaBSE`
   (`src/evaluation/translation_quality.py`) to assess semantic agreement
   between translated German text and its English source.
5. **LaBSE-based filtering** retained translated rows with a similarity score
   of `>= 0.55`. This threshold is a project-specific heuristic quality
   cutoff and does not prove preservation of every pragmatic or tonal nuance.
6. **Exact-text deduplication** (`src/dataset/deduplicate.py`), keeping the
   best representative row per duplicate (gold over silver, more
   positive/annotated labels, longer text, as tiebreakers).
7. **Fresh split assignment** (`src/dataset/splits.py`,
   `assign_fresh_splits`): silver rows always go to `train`; gold rows are
   split ~70/15/15 via multilabel stratification (`iterstrat`, seed `42`).
   Validation and test splits are gold-only.

The preprocessing workflow evolved across several scripts and interactive
processing steps rather than one fully preserved end-to-end entry point.
`run_pipeline.py` covers corpus loading, translation and quality metadata,
but does not by itself reproduce the complete frozen dataset. After LaBSE-
based filtering at `>= 0.55`, the intermediate dataset
`data/processed/unified_with_quality_translated_filtered_labse055.parquet`
contained 573,366 rows.

The finalization step is implemented in `src/dataset/deduplicate.py`. It
normalizes texts using stripped text keys and removes duplicate texts while
prioritizing gold over silver rows, followed by rows with more positive
labels, more annotated labels, longer text, and finally the lower original
index as a deterministic tiebreaker. This reduces the dataset from 573,366
to 453,242 rows. The script then assigns fresh splits via
`assign_fresh_splits`: all silver rows are assigned to training, while gold
rows are multilabel-stratified into approximately 70/15/15 train,
validation, and test splits.

The resulting
`data/processed/unified_final_deduplicated.parquet` is identical to the
frozen dataset used for the reported experiments,
`data/final/unified_final_v1.parquet`.

Final dataset statistics: 453,242 rows total (88,330 gold / 364,912
silver); split into 426,742 train / 13,250 val / 13,250 test rows.

## 6. Dataset Loaders and Harmonization

`src/dataset/schema.py` defines the canonical schema and per-corpus label
map; `src/dataset/base.py` provides the abstract `BaseCorpusLoader`;
`src/dataset/unified.py` loads/concatenates the active corpora
(`UnifiedCorpusDataset`); `src/dataset/loaders/` holds one module per
corpus; `src/dataset/deduplicate.py` and `src/dataset/splits.py` implement
deduplication and split assignment (Section 5). Detailed per-corpus
mapping tables are documented in the thesis and not reproduced here.

## 7. Baseline and BERT Models

### TF-IDF + Linear SVM

Implemented in `src/training/baseline_tfidf_svm.py`: TF-IDF (unigrams +
bigrams, `sublinear_tf=True`) feeding a `LinearSVC` with
`class_weight="balanced"`. One classifier per label, trained only on rows
where that label is annotated.

### Transformer models

Generic transformer training machinery: `src/training/transformer_dataset.py`,
`src/training/transformer_metrics.py`, `src/training/losses.py`,
`src/training/train_transformer.py`.

`train_transformer.py` is a standalone, generic CLI implementation (default
base model `bert-base-german-cased`) and is not the sole provenance of
every reported BERT experiment — the original development, including the
final Focal Loss and ASL configurations, was carried out interactively in
`notebooks/masterarbeit02.ipynb` and `masterarbeit04.ipynb` (Section 14).

Models: `bert-base-german-cased`, `deepset/gbert-large`. Training
conditions: gold-only, gold+silver. Losses (`src/training/losses.py`, plus
an ASL class defined directly in `masterarbeit02.ipynb`): standard BCE,
Focal Loss, and Asymmetric Loss (Ben-Baruch et al. 2021). All losses mask
out NaN/unannotated labels.

## 8. Reported Final BERT Configuration

The final reported BERT system corresponds to
`results/gbert_large_gold_silver_128_focal_lr5e6/`:

- Model: `deepset/gbert-large`
- Training basis: Gold + Silver
- Loss: Focal Loss
- Max sequence length: 128

Final label-specific decision thresholds (`thresholds.json`, tuned on the
validation split):

| Label       | Threshold |
|-------------|-----------|
| hate_speech | 0.55      |
| toxic       | 0.30      |
| threat      | 0.80      |
| insult      | 0.45      |

Test-set macro metrics from `test_metrics.csv`: **Macro-F1 = 0.493**,
**Macro-S = 0.643**, matching the rounded values reported for this
configuration in the thesis. Full per-label Precision/Recall/F1/F2/MCC/
S-Score are available in `results/gbert_large_gold_silver_128_focal_lr5e6/test_metrics.csv`.

## 9. LLM Classification

Two separate, largely independent LLM inference pipelines exist in this
repository.

### 9.1 Ollama / Qwen

Core files: `src/llm/backends/base.py`, `src/llm/backends/ollama.py`,
`src/llm/model_configs.py`, `src/llm/engine.py`,
`src/llm/prompt_registry.py`, `src/llm/parsing.py`, `src/llm/checkpoint.py`.

The primary reported local model is **Qwen2.5:7B**. Relevant scripts:
`experiments/run_ollama_eval.py` (single-model, single-variant evaluation)
and `experiments/run_llm_prompt_ablation.py` (prompt-variant ablation).

Features: joint classification (single JSON call) and per-label
classification (one independent binary call per label), five prompt variants
(`joint_basic`, `joint_definitions`, `joint_fewshot`, `joint_rules`,
`joint_selfcheck`), resume-safe JSONL checkpointing, and multi-strategy JSON
parsing with reasoning-block stripping for chain-of-thought models.

`src/llm/model_configs.py` defines generation settings for a larger set of
models (Qwen2.5/3, Llama 3.1/3.2, Gemma 3, Mistral, DeepSeek-R1, Phi-4);
being listed there does not imply a model was used for a final experiment.

### 9.2 Gemini

Core files: `src/llm/gemini_client.py`, `src/llm/inference.py`,
`src/llm/prompts.py`, `src/llm/output_parser.py`,
`src/llm/platform_rules.py`.

The reported model is **Gemini 2.5 Flash** (`gemini-2.5-flash`), as
recorded directly in the corresponding result artifacts
(`results/llm_gemini_flash_full/summary.json`,
`results/llm_gemini_cot/summary.json`). The Gemini implementation and
experiment entry points use `gemini-2.5-flash` as their current default.
Experimental conditions: Zero-Shot, Few-Shot, and Chain-of-Thought (CoT),
run via `experiments/run_llm_eval.py`. CoT is an experimental condition
(`results/llm_gemini_cot/`), not a demonstrated improvement — see Section
11. The CoT run produced valid predictions for 12,049 of 13,250 test rows;
its comparison with the Few-Shot configuration without CoT is therefore
performed on the common-success subset.

## 10. Hybrid BERT–LLM Experiments

Two distinct stages of hybrid experimentation exist in this repository.

### Exploratory score-based fusion

Code: `src/hybrid/fusion.py`, `experiments/run_hybrid_eval.py`.

Strategies: `AverageFusion`, `WeightedFusion`, `ConfidenceGatedFusion`,
`StackingFusion`, `UnionFusion`, `IntersectionFusion`. This stage is
exploratory; its output directory (`results/hybrid_e7_gemini/`) documents
these comparisons but is **not** the final reported hybrid result.

### Final threshold-aware routing

Code: `experiments/run_perlabel_hybrid.py`. Combines the final BERT model
(`results/gbert_large_gold_silver_128_focal_lr5e6/`) with the Gemini 2.5
Flash run (`results/llm_gemini_flash_full/`) using:

- BERT's label-specific decision thresholds (from `thresholds.json`)
- **Per-label static routing**: for each label, compare the BERT validation
  S-Score against fixed LLM prior S-Scores stored in the experiment script.
  Because the provenance of these prior values cannot be reconstructed as
  validation-only from the retained execution artifacts, this strategy is
  treated as exploratory.
- **Uncertainty routing**: when BERT's sigmoid probability falls within a
  fixed band around its threshold, defer to the LLM prediction instead
- **Threat-only uncertainty routing**: the same uncertainty-band routing
  applied only to the `threat` label
- BERT fallback for any row where the LLM call failed
- Paired bootstrap comparison (1,000 resamples, seed 42) of each routing
  strategy's macro S-Score against the BERT-only baseline

Final result artifacts: `results/perlabel_hybrid/`. In the run recorded
there, per-label static routing selected BERT for all four labels, and
none of the routing strategies showed a statistically significant
improvement over BERT alone in the paired bootstrap comparison
(`results/perlabel_hybrid/summary.json`). `results/hybrid_e7_gemini/` is
exploratory score-fusion output and is not used as evidence for this final
threshold-aware hybrid result.

## 11. Statistical Evaluation

Core utilities: `src/evaluation/bootstrap_ci.py`,
`experiments/compare_llm_runs.py`, `experiments/run_perlabel_hybrid.py`.
All bootstrap CIs use 1,000 resamples with seed 42, and respect the label
mask so unannotated rows never enter a metric computation.

Two distinct comparison designs are used:

- **BERT vs. Gemini, and final hybrid routing vs. BERT**: paired bootstrap
  of the delta in macro S-Score, using identical resample indices for both
  systems (`paired_bootstrap_delta` in `run_perlabel_hybrid.py`);
  significance is judged by whether the delta's 95% CI excludes zero.
- **Gemini Few-Shot (joint) vs. Chain-of-Thought**: evaluated on the
  common-success subset only (12,049 of 13,250 rows where both runs
  produced a valid prediction; `experiments/compare_llm_runs.py`), with
  **separate** bootstrap CIs computed for each run's macro S-Score and
  compared by interval overlap (`results/llm_fair_comparison/summary.json`).
  This is **not** a paired-delta confidence interval — no such paired CI
  was computed for the CoT comparison.

## 12. Platform-Specific Adaptation

`src/llm/platform_rules.py` defines configurable prompt-side presets that
inject platform-specific guidelines into the LLM system prompt: `strict`,
`debate_forum`, `news_comments`, `social_media`, `annotation`. Custom rule
sets can also be built ad hoc via `custom_platform_rules()`.

This demonstrates technical configurability of the prompt layer for
different moderation contexts. Cross-platform effectiveness has **not**
been empirically validated on multiple real platform datasets in this
repository; the presets were exercised via prompt configuration only, not
validated against held-out data from each named platform.

## 13. Results Directory

`results/` contains one subdirectory per experiment run. The most relevant
for the thesis narrative:

- `results/baseline_tfidf_svm/` — TF-IDF + SVM baseline
- `results/bert_german_gold_only/`, `results/bert_german_gold_silver/` —
  early `bert-base-german-cased` runs
- `results/gbert_large_gold_only_128_focal_batch8/` — gbert-large,
  gold-only, Focal Loss
- `results/gbert_large_gold_silver_128_focal_lr5e6/` — **final reported
  BERT configuration** (Section 8)
- `results/gbert_large_gold_silver_128_asl_v3/` — gbert-large, gold+silver,
  ASL ablation
- `results/llm_prompt_ablation_500/` — Ollama prompt-variant ablation
- `results/ollama_qwen25_7b_joint/`, `results/ollama_qwen25_7b_per_label/` —
  Qwen2.5:7B joint vs. per-label
- `results/llm_gemini_flash_full/`, `results/llm_gemini_cot/` — Gemini 2.5
  Flash, full test set and CoT condition
- `results/llm_fair_comparison/` — common-success-subset comparison of the
  two Gemini runs
- `results/perlabel_hybrid/` — **final threshold-aware hybrid result**

The directory also contains exploratory, debug, and smoke-test runs (e.g.
`results/qwen3_smoke_test/`, `results/llm_gemini_debug/`,
`results/test_joint*/`) retained for provenance; directory names alone are
not claims about a configuration's final status.

## 14. Notebooks and Experimental Provenance

`notebooks/` contains the original exploratory and Colab-based experiment
development that preceded — and, for the BERT stage, substantially
replaced — generic script-based training.

- `00_masterarbeit_dataset.ipynb`, `masterarbeit00.ipynb`,
  `masterarbeit01.ipynb` — early Colab setup and dataset
  combination/download
- `01_eda.ipynb` — exploratory data analysis across the seven harmonized
  corpora
- `masterarbeit02.ipynb` — executed BERT experiment cells (`bert-base-german-cased`
  and `deepset/gbert-large`; BCE, Focal, and ASL loss runs, including the
  ASL loss class definition)
- `masterarbeit03.ipynb`, `masterarbeit04.ipynb` — further Colab BERT
  training notebooks, with an experiment tracking table (E1–E8)
- `validate_methodology.ipynb` — sanity-checks the frozen final dataset
  against expected per-source row counts

These notebooks — not `src/training/train_transformer.py` — are the
provenance for the Focal Loss and ASL BERT configurations; the generic
script and the notebooks coexist rather than one fully superseding the
other.

## 15. Installation

```bash
pip install -r requirements.txt
```

## 16. External Requirements

**Ollama** (local Qwen inference) must be installed and running separately,
with the target model pulled, e.g.:

```bash
ollama pull qwen2.5:7b
```

**Gemini API** access requires an API key set as an environment variable:

```bash
export GEMINI_API_KEY=your_key_here
```

Never commit a real API key to this repository.

## 17. Reproducibility Notes

- Raw third-party corpora are not included and must be obtained separately,
  placed at the paths expected by `src/dataset/loaders/`.
- Some Colab notebooks contain Colab-specific (`/content/drive/...`) or the
  author's local filesystem paths; not portable as-is.
- The generic training script and the original notebooks coexist and were
  not unified into one reproducible entry point (Section 14).
- Several `results/` subdirectories are exploratory/debug/smoke-test runs,
  not final configurations (Section 13).
- The LaBSE `>= 0.55` threshold is a project-specific heuristic, not a
  validated absolute quality standard.
- Gemini API behavior (model availability, rate limits, safety filtering)
  can change over time and is outside this repository's control.
- Final BERT evaluation uses the frozen gold test split of
  `data/final/unified_final_v1.parquet`.

## 18. Thesis Reference

Vu Hoang Anh Dang
Master's thesis, Hochschule München

*Hybride Multilabel-Klassifikation von Online-Kommunikationsverhalten
bezüglich der Netiquette*
