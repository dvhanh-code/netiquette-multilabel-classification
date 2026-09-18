"""
experiments/compare_llm_runs.py
--------------------------------
Fairer Vergleich zweier LLM-Läufe auf der Schnittmenge erfolgreicher Vorhersagen.

Hintergrund:
    Der CoT-Lauf konnte aufgrund von API-Kontingentbeschränkungen nur 12.049 von
    13.250 Testinstanzen abschließen. Die fehlgeschlagenen Instanzen liegen als
    NaN in predictions.npz vor und werden von der Standard-Evaluation implizit
    als negative Vorhersagen gewertet. Dies unterschätzt Recall und F1 des
    CoT-Laufs systematisch.

    Dieses Skript evaluiert beide Läufe ausschließlich auf jenen Instanzen,
    für die BEIDE Läufe eine gültige Vorhersage geliefert haben.

Verwendung:
    python3 experiments/compare_llm_runs.py \\
        --run-a results/llm_gemini_flash_full \\
        --run-b results/llm_gemini_cot \\
        --labels-from results/gbert_large_gold_silver_128_focal_lr5e6 \\
        --name-a "Gemini (Joint)" \\
        --name-b "Gemini (CoT)"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["hate_speech", "toxic", "threat", "insult"]


# ── Metrics ────────────────────────────────────────────────────────────────

def confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def f_beta(tp: int, fp: int, fn: int, beta: float) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if prec == 0.0 and rec == 0.0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


def s_score(tp: int, fp: int, fn: int, tn: int) -> float:
    """S-Score = arithmetisches Mittel aus F2 und normiertem MCC (Ippen Digital)."""
    f2       = f_beta(tp, fp, fn, beta=2.0)
    mcc_norm = (mcc(tp, fp, fn, tn) + 1.0) / 2.0
    return (f2 + mcc_norm) / 2.0


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             label_mask: np.ndarray,
             eval_mask: np.ndarray,
             system_name: str) -> pd.DataFrame:
    """
    y_true, y_pred, label_mask : (n_rows, n_labels)
    eval_mask                  : (n_rows,)  — Zeilen, die evaluiert werden
    """
    rows = []
    for j, label in enumerate(LABELS):
        keep = eval_mask & label_mask[:, j]
        yt = y_true[keep, j].astype(int)
        yp = y_pred[keep, j].astype(int)

        tp, fp, fn, tn = confusion(yt, yp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append({
            "system":       system_name,
            "label":        label,
            "n_evaluated":  int(keep.sum()),
            "n_positives":  int(yt.sum()),
            "precision":    prec,
            "recall":       rec,
            "f1":           f_beta(tp, fp, fn, 1.0),
            "f2":           f_beta(tp, fp, fn, 2.0),
            "mcc":          mcc(tp, fp, fn, tn),
            "s_score":      s_score(tp, fp, fn, tn),
        })

    df = pd.DataFrame(rows)
    macro = {
        "system":      system_name,
        "label":       "MACRO",
        "n_evaluated": int(eval_mask.sum()),
        "n_positives": int(df["n_positives"].sum()),
    }
    for col in ["precision", "recall", "f1", "f2", "mcc", "s_score"]:
        macro[col] = float(df[col].mean())

    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


# ── Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_macro_s(y_true, y_pred, label_mask, eval_mask,
                      n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx_pool = np.flatnonzero(eval_mask)
    scores = []

    for _ in range(n_resamples):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        per_label = []
        for j in range(len(LABELS)):
            sel = idx[label_mask[idx, j]]
            if len(sel) == 0:
                continue
            yt = y_true[sel, j].astype(int)
            yp = y_pred[sel, j].astype(int)
            per_label.append(s_score(*confusion(yt, yp)))
        if per_label:
            scores.append(np.mean(per_label))

    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ── IO ─────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    path = run_dir / "predictions.npz"
    if not path.exists():
        sys.exit(f"ERROR: {path} nicht gefunden.")
    d = np.load(path)
    return d["predictions"], d["success_mask"].astype(bool)


def load_ground_truth(bert_dir: Path):
    path = bert_dir / "test_logits.npz"
    if not path.exists():
        sys.exit(
            f"ERROR: {path} nicht gefunden.\n"
            "Ground-Truth-Labels werden aus dem BERT-Testlauf gelesen."
        )
    d = np.load(path)
    return d["labels"].astype(int), d["label_mask"].astype(bool)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--labels-from", required=True,
                    help="BERT-Verzeichnis mit test_logits.npz (Ground Truth)")
    ap.add_argument("--name-a", default="Run A")
    ap.add_argument("--name-b", default="Run B")
    ap.add_argument("--output-dir", default="results/llm_fair_comparison")
    ap.add_argument("--n-resamples", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_a, ok_a = load_run(Path(args.run_a))
    preds_b, ok_b = load_run(Path(args.run_b))
    y_true, label_mask = load_ground_truth(Path(args.labels_from))

    assert preds_a.shape == preds_b.shape == y_true.shape, \
        f"Shape mismatch: {preds_a.shape} / {preds_b.shape} / {y_true.shape}"

    common = ok_a & ok_b

    print("=" * 74)
    print("EVALUATIONSBASIS")
    print("=" * 74)
    print(f"  Gesamtinstanzen           : {len(common):>7,}")
    print(f"  {args.name_a:<24}: {ok_a.sum():>7,}  ({ok_a.mean():.1%})")
    print(f"  {args.name_b:<24}: {ok_b.sum():>7,}  ({ok_b.mean():.1%})")
    print(f"  Schnittmenge (beide gültig): {common.sum():>7,}  ({common.mean():.1%})")
    print(f"  Ausgeschlossen             : {(~common).sum():>7,}")
    print("=" * 74)

    # NaN → 0 nur zur Sicherheit; auf `common` gibt es keine NaN mehr
    pa = np.nan_to_num(preds_a, nan=0.0)
    pb = np.nan_to_num(preds_b, nan=0.0)

    df_a = evaluate(y_true, pa, label_mask, common, args.name_a)
    df_b = evaluate(y_true, pb, label_mask, common, args.name_b)
    table = pd.concat([df_a, df_b], ignore_index=True)

    # ── Ausgabe pro Label ──────────────────────────────────────────────────
    for label in LABELS + ["MACRO"]:
        sub = table[table["label"] == label]
        print(f"\n  ── {label} ──")
        print(f"  {'System':<24} {'P':>7} {'R':>7} {'F1':>7} {'F2':>7} "
              f"{'MCC':>7} {'S':>7}")
        print(f"  {'-'*70}")
        for _, r in sub.iterrows():
            print(f"  {r['system']:<24} {r['precision']:>7.4f} {r['recall']:>7.4f} "
                  f"{r['f1']:>7.4f} {r['f2']:>7.4f} {r['mcc']:>7.4f} "
                  f"{r['s_score']:>7.4f}")

        if label != "MACRO":
            a = sub.iloc[0]["s_score"]
            b = sub.iloc[1]["s_score"]
            arrow = "↑" if b > a else ("↓" if b < a else "=")
            print(f"  {'Δ S-Score (B − A)':<24} {b - a:>+7.4f}  {arrow}")

    # ── Bootstrap CI für Macro-S ───────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"BOOTSTRAP 95%-KONFIDENZINTERVALLE (Macro S, n={args.n_resamples})")
    print("=" * 74)

    ci_a = bootstrap_macro_s(y_true, pa, label_mask, common, args.n_resamples)
    ci_b = bootstrap_macro_s(y_true, pb, label_mask, common, args.n_resamples)

    s_a = float(df_a[df_a["label"] == "MACRO"]["s_score"].iloc[0])
    s_b = float(df_b[df_b["label"] == "MACRO"]["s_score"].iloc[0])

    print(f"  {args.name_a:<24} S = {s_a:.4f}  [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"  {args.name_b:<24} S = {s_b:.4f}  [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print(f"  {'Differenz (B − A)':<24}     {s_b - s_a:+.4f}")

    overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
    if overlap:
        print(f"\n  → Konfidenzintervalle überlappen: kein statistisch "
              f"signifikanter Unterschied.")
    else:
        print(f"\n  → Konfidenzintervalle überlappen nicht: signifikanter "
              f"Unterschied.")
    print("=" * 74)

    # ── Speichern ──────────────────────────────────────────────────────────
    table.to_csv(out_dir / "fair_comparison.csv", index=False)

    summary = {
        "run_a":            {"dir": args.run_a, "name": args.name_a,
                             "n_successful": int(ok_a.sum())},
        "run_b":            {"dir": args.run_b, "name": args.name_b,
                             "n_successful": int(ok_b.sum())},
        "n_total":          int(len(common)),
        "n_common":         int(common.sum()),
        "n_excluded":       int((~common).sum()),
        "macro_s_a":        s_a,
        "macro_s_a_ci":     list(ci_a),
        "macro_s_b":        s_b,
        "macro_s_b_ci":     list(ci_b),
        "delta_macro_s":    s_b - s_a,
        "ci_overlap":       bool(overlap),
        "per_label_delta_s": {
            lbl: float(
                df_b[df_b["label"] == lbl]["s_score"].iloc[0]
                - df_a[df_a["label"] == lbl]["s_score"].iloc[0]
            ) for lbl in LABELS
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nErgebnisse gespeichert → {out_dir}")


if __name__ == "__main__":
    main()