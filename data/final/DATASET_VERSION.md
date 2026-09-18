# unified_final_v1

Frozen dataset version for Master Thesis experiments.

Created: 2026-05-11

Pipeline:
1. Load corpora
2. Translate EN→DE
3. Assign quality labels
4. Filter LaBSE >= 0.55
5. Deduplicate exact texts
6. Fresh multilabel train/val/test split

Final statistics:
- Rows: 453,242
- Gold: 88,330
- Silver: 364,912

Split:
- Train: 426,742
- Val: 13,250
- Test: 13,250

Labels:
- hate_speech
- toxic
- threat
- insult