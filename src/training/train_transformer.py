"""
src/training/train_transformer.py
---------------------------------
Train a Transformer model for partially labeled multilabel
Netiquette classification.

Default model:
    bert-base-german-cased

Supported modes:
    gold_only    -> train only on gold train rows
    gold_silver  -> train on all train rows, evaluate on gold val/test

Usage:
    # quick debug run
    python3 src/training/train_transformer.py --mode gold_only --debug

    # normal gold-only training
    python3 src/training/train_transformer.py --mode gold_only

    # gold + silver training
    python3 src/training/train_transformer.py --mode gold_silver
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.transformer_dataset import (
    LABELS,
    NetiquetteTransformerDataset,
    load_dataset,
    print_dataset_summary,
)
from src.training.losses import MaskedBCEWithLogitsLoss
from src.training.transformer_metrics import (
    compute_multilabel_metrics,
    tune_thresholds,
    print_metrics_table,
)


DATA_PATH = Path("data/final/unified_final_v1.parquet")


class TransformerForMultilabelClassification(nn.Module):
    """
    Transformer encoder with a multilabel classification head.
    """

    def __init__(self, model_name: str, num_labels: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # BERT-style models provide pooler_output.
        # If unavailable, fall back to CLS token representation.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.classifier(self.dropout(pooled))
        return logits


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    """Move all tensor values in batch to device."""
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epoch: int,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )

        loss = loss_fn(
            logits=logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

        if step % 100 == 0:
            print(f"  Epoch {epoch} | step {step:,}/{len(dataloader):,} | loss={loss.item():.4f}")

    return total_loss / max(steps, 1)


@torch.no_grad()
def predict_logits(model, dataloader, device):
    """Collect logits, labels, and label masks."""
    model.eval()

    all_logits = []
    all_labels = []
    all_masks = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())
        all_masks.append(batch["label_mask"].detach().cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_masks, axis=0),
    )


def make_dataloader(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create PyTorch DataLoader."""
    dataset = NetiquetteTransformerDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def maybe_debug_subset(
    splits: Dict[str, pd.DataFrame],
    debug: bool,
) -> Dict[str, pd.DataFrame]:
    """Use small subsets for fast pipeline debugging."""
    if not debug:
        return splits

    print("\nDEBUG MODE ACTIVE: using small subsets")

    return {
        "train": splits["train"].sample(
            n=min(5_000, len(splits["train"])),
            random_state=42,
        ).reset_index(drop=True),
        "val": splits["val"].sample(
            n=min(1_000, len(splits["val"])),
            random_state=42,
        ).reset_index(drop=True),
        "test": splits["test"].sample(
            n=min(1_000, len(splits["test"])),
            random_state=42,
        ).reset_index(drop=True),
    }


def save_config(args, output_dir: Path) -> None:
    """Save training configuration."""
    config = vars(args).copy()
    config["data_path"] = str(DATA_PATH)
    config["labels"] = LABELS

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["gold_only", "gold_silver"],
        default="gold_only",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-german-cased",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    if args.output_dir is None:
        debug_suffix = "_debug" if args.debug else ""
        output_dir = Path(f"results/transformer_{args.mode}{debug_suffix}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_model").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRANSFORMER TRAINING")
    print("=" * 80)
    print(f"Model:      {args.model_name}")
    print(f"Mode:       {args.mode}")
    print(f"Device:     {device}")
    print(f"Output dir: {output_dir}")

    save_config(args, output_dir)

    print("\nLoading dataset...")
    splits = load_dataset(str(DATA_PATH), mode=args.mode)
    splits = maybe_debug_subset(splits, debug=args.debug)
    print_dataset_summary(splits)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
    )

    train_loader = make_dataloader(
        splits["train"],
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_dataloader(
        splits["val"],
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_dataloader(
        splits["test"],
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
    )

    print("\nLoading model...")
    model = TransformerForMultilabelClassification(
        model_name=args.model_name,
        num_labels=len(LABELS),
    )
    model.to(device)

    loss_fn = MaskedBCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print("\nTraining setup:")
    print(f"  epochs:              {args.epochs}")
    print(f"  batch_size:          {args.batch_size}")
    print(f"  train steps/epoch:   {len(train_loader):,}")
    print(f"  total steps:         {total_training_steps:,}")
    print(f"  warmup steps:        {warmup_steps:,}")
    print(f"  learning rate:       {args.learning_rate}")

    best_val_s = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        print(f"\nEpoch {epoch} train loss: {train_loss:.4f}")

        print("\nEvaluating on validation set...")
        val_logits, val_labels, val_mask = predict_logits(model, val_loader, device)

        val_metrics_05 = compute_multilabel_metrics(
            logits=val_logits,
            labels=val_labels,
            label_mask=val_mask,
            split_name="val",
        )

        print_metrics_table(f"VAL METRICS epoch {epoch} threshold=0.5", val_metrics_05)

        macro_s = float(
            val_metrics_05[val_metrics_05["label"] == "MACRO"]["s_score"].iloc[0]
        )

        val_metrics_05.to_csv(
            output_dir / f"val_metrics_epoch_{epoch}.csv",
            index=False,
        )

        if macro_s > best_val_s:
            best_val_s = macro_s
            best_epoch = epoch

            print(f"\nNew best model: epoch {epoch}, val macro S={macro_s:.4f}")

            torch.save(
                model.state_dict(),
                output_dir / "best_model" / "pytorch_model.bin",
            )

            tokenizer.save_pretrained(output_dir / "best_model")

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro S: {best_val_s:.4f}")

    print("\nLoading best model...")
    model.load_state_dict(torch.load(output_dir / "best_model" / "pytorch_model.bin", map_location=device, weights_only=True))
    model.to(device)

    print("\nPredicting validation set for threshold tuning...")
    val_logits, val_labels, val_mask = predict_logits(model, val_loader, device)

    best_thresholds = tune_thresholds(
        logits=val_logits,
        labels=val_labels,
        label_mask=val_mask,
        metric="s_score",
    )

    print("\nBest thresholds tuned on validation set:")
    print(best_thresholds)

    (output_dir / "thresholds.json").write_text(
        json.dumps(best_thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    val_metrics_tuned = compute_multilabel_metrics(
        logits=val_logits,
        labels=val_labels,
        label_mask=val_mask,
        thresholds=best_thresholds,
        split_name="val",
    )

    print_metrics_table("FINAL VAL METRICS tuned thresholds", val_metrics_tuned)

    print("\nPredicting test set...")
    test_logits, test_labels, test_mask = predict_logits(model, test_loader, device)

    test_metrics_tuned = compute_multilabel_metrics(
        logits=test_logits,
        labels=test_labels,
        label_mask=test_mask,
        thresholds=best_thresholds,
        split_name="test",
    )

    print_metrics_table("FINAL TEST METRICS tuned thresholds", test_metrics_tuned)

    val_metrics_tuned.to_csv(output_dir / "val_metrics.csv", index=False)
    test_metrics_tuned.to_csv(output_dir / "test_metrics.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_macro_s_05": best_val_s,
        "thresholds": best_thresholds,
        "val_macro": val_metrics_tuned[val_metrics_tuned["label"] == "MACRO"]
        .iloc[0]
        .to_dict(),
        "test_macro": test_metrics_tuned[test_metrics_tuned["label"] == "MACRO"]
        .iloc[0]
        .to_dict(),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  {output_dir / 'config.json'}")
    print(f"  {output_dir / 'thresholds.json'}")
    print(f"  {output_dir / 'val_metrics.csv'}")
    print(f"  {output_dir / 'test_metrics.csv'}")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()