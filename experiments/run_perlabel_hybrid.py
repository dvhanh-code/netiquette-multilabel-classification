"""
experiments/run_perlabel_hybrid.py
-----------------------------------
E12: Per-Label Hybrid Routing — BERT und LLM pro Label kombiniert.

Motivation:
    Die einfachen Fusionsstrategien aus E11 (Average, Union, ...) behandeln
    alle Labels gleich. Die Einzelergebnisse zeigen jedoch komplementäre
    Stärken: BERT dominiert bei hate_speech, das LLM bei threat.
    Dieses Skript evaluiert Routing-Strategien, die pro Label das jeweils
    stärkere System wählen.

Strategien:
    1. Per-label static routing : pro Label das laut Vorwissen stärkere
                                  System (Entscheidung: BERT-val-S vs.
                                  publizierte E10-Werte — kein Test-Leakage)
    2. Uncertainty routing      : BERT entscheidet; liegt seine Sigmoid-
                                  Wahrscheinlichkeit im Band [t−r, t+r],
                                  übernimmt das LLM. Festes r, kein Tuning.
    3. Uncertainty threat-only  : wie 2, aber nur für das Label threat.

Voraussetzungen:
    BERT-Dir:  val_logits.npz, test_logits.npz, thresholds.json
    LLM-Dir:   predictions.npz (test)

Usage:
    python3 experiments/run_perlabel_hybrid.py \\
        --bert-dir results/gbert_large_gold_silver_128_focal_lr5e6 \\
        --llm-dir  results/llm_gemini_flash_full \\
        --output-dir results/perlabel_hybrid
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = ["hate_speech", "toxic", "threat", "insult"]


# ── Metrics ────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def f_beta(tp, fp, fn, beta):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if prec == 0.0 and rec == 0.0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def mcc(tp, fp, fn, tn):
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return num / den if den > 0 else 0.0


def s_score(tp, fp, fn, tn):
    f2 = f_beta(tp, fp, fn, 2.0)
    mcc_norm = (mcc(tp, fp, fn, tn) + 1.0) / 2.0
    return (f2 + mcc_norm) / 2.0


def label_s(y_true, y_pred, label_mask, j):
    m = label_mask[:, j]
    return s_score(*confusion(y_true[m, j].astype(int),
                              y_pred[m, j].astype(int)))


def full_eval(y_true, y_pred, label_mask, system):
    rows = []
    for j, lbl in enumerate(LABELS):
        m = label_mask[:, j]
        yt, yp = y_true[m, j].astype(int), y_pred[m, j].astype(int)
        tp, fp, fn, tn = confusion(yt, yp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "system": system, "label": lbl,
            "n": int(m.sum()), "pos": int(yt.sum()),
            "precision": prec, "recall": rec,
            "f1": f_beta(tp, fp, fn, 1.0), "f2": f_beta(tp, fp, fn, 2.0),
            "mcc": mcc(tp, fp, fn, tn), "s_score": s_score(tp, fp, fn, tn),
        })
    df = pd.DataFrame(rows)
    macro = {"system": system, "label": "MACRO",
             "n": int(df["n"].sum()), "pos": int(df["pos"].sum())}
    for c in ["precision", "recall", "f1", "f2", "mcc", "s_score"]:
        macro[c] = float(df[c].mean())
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


# ── Paired bootstrap ───────────────────────────────────────────────────────

def paired_bootstrap_delta(y_true, pred_a, pred_b, label_mask,
                           n_resamples=1000, seed=42):
    """CI of (macro_S_b − macro_S_a) using identical resample indices."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sa, sb = [], []
        for j in range(len(LABELS)):
            m = label_mask[idx, j]
            if m.sum() == 0:
                continue
            ytj = y_true[idx, j][m].astype(int)
            sa.append(s_score(*confusion(ytj, pred_a[idx, j][m].astype(int))))
            sb.append(s_score(*confusion(ytj, pred_b[idx, j][m].astype(int))))
        if sa:
            deltas.append(np.mean(sb) - np.mean(sa))
    d = np.array(deltas)
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-dir", required=True)
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--output-dir", default="results/perlabel_hybrid")
    ap.add_argument("--uncertainty-radius", type=float, default=0.15,
                    help="festes r für die uncertainty-Strategie")
    ap.add_argument("--n-resamples", type=int, default=1000)
    args = ap.parse_args()

    bert_dir = Path(args.bert_dir)
    llm_dir  = Path(args.llm_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    bert_val  = np.load(bert_dir / "val_logits.npz")
    bert_test = np.load(bert_dir / "test_logits.npz")
    llm_test  = np.load(llm_dir / "predictions.npz")
    thresholds = json.load(open(bert_dir / "thresholds.json"))

    thr = np.array([thresholds[l] for l in LABELS])

    yv = bert_val["labels"].astype(int)
    mv = bert_val["label_mask"].astype(bool)
    pv = sigmoid(bert_val["logits"])
    bert_val_pred = (pv >= thr).astype(int)

    yt = bert_test["labels"].astype(int)
    mt = bert_test["label_mask"].astype(bool)
    pt = sigmoid(bert_test["logits"])
    bert_test_pred = (pt >= thr).astype(int)

    llm_pred_raw = llm_test["predictions"]
    llm_ok       = llm_test["success_mask"].astype(bool)
    # LLM-Ausfälle: auf BERT zurückfallen (dokumentiert im Summary)
    llm_test_pred = np.where(np.isnan(llm_pred_raw),
                             bert_test_pred, llm_pred_raw).astype(int)
    n_fallback = int((~llm_ok).sum())

    print(f"Loaded. Test rows: {len(yt):,} | "
          f"LLM fallback rows (→BERT): {n_fallback}")

    # ── Strategy 1: per-label static routing ───────────────────────────────
    # Entscheidung: BERT-val-S vs. publizierte E10-Werte als Prior.
    # Kein Zugriff auf Test-Ground-Truth bei der Auswahl.
    E10_PRIOR_S = {"hate_speech": 0.504, "toxic": 0.728,
                   "threat": 0.474, "insult": 0.663}

    routing = {}
    print("\nPer-Label-Routing (BERT-val-S vs. E10-Prior):")
    for j, lbl in enumerate(LABELS):
        s_bert_val = label_s(yv, bert_val_pred, mv, j)
        s_llm_prior = E10_PRIOR_S[lbl]
        choice = "bert" if s_bert_val >= s_llm_prior else "llm"
        routing[lbl] = choice
        print(f"  {lbl:<12} BERT(val)={s_bert_val:.4f}  "
              f"LLM(prior)={s_llm_prior:.4f}  → {choice.upper()}")

    static_pred = bert_test_pred.copy()
    for j, lbl in enumerate(LABELS):
        if routing[lbl] == "llm":
            static_pred[:, j] = llm_test_pred[:, j]

    # ── Strategy 2: uncertainty routing (festes r) ─────────────────────────
    r = args.uncertainty_radius
    uncertain = np.abs(pt - thr) <= r
    unc_pred = np.where(uncertain, llm_test_pred, bert_test_pred).astype(int)
    frac_routed = uncertain.mean(axis=0)
    print(f"\nUncertainty-Routing (r={r}) — Anteil an LLM delegiert:")
    for j, lbl in enumerate(LABELS):
        print(f"  {lbl:<12} {frac_routed[j]:.1%}")

    # ── Strategy 3: uncertainty nur für threat ─────────────────────────────
    thr_only_pred = bert_test_pred.copy()
    j_threat = LABELS.index("threat")
    thr_only_pred[:, j_threat] = np.where(
        uncertain[:, j_threat],
        llm_test_pred[:, j_threat],
        bert_test_pred[:, j_threat],
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    systems = {
        "BERT only (E7)":                     bert_test_pred,
        "LLM only (E10, BERT-fallback)":      llm_test_pred,
        "Per-label static routing":           static_pred,
        f"Uncertainty routing (r={r})":       unc_pred,
        f"Uncertainty threat-only (r={r})":   thr_only_pred,
    }

    tables = [full_eval(yt, p, mt, name) for name, p in systems.items()]
    table = pd.concat(tables, ignore_index=True)

    for lbl in LABELS + ["MACRO"]:
        sub = table[table["label"] == lbl]
        print(f"\n  ── {lbl} ──")
        print(f"  {'System':<36} {'P':>7} {'R':>7} {'F1':>7} {'S':>7}")
        print(f"  {'-'*68}")
        for _, row in sub.iterrows():
            print(f"  {row['system']:<36} {row['precision']:>7.4f} "
                  f"{row['recall']:>7.4f} {row['f1']:>7.4f} "
                  f"{row['s_score']:>7.4f}")

    # ── Paired bootstrap vs BERT ───────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"PAIRED BOOTSTRAP — ΔS gegenüber BERT (n={args.n_resamples})")
    print(f"{'='*74}")
    results_ci = {}
    for name, p in systems.items():
        if name.startswith("BERT"):
            continue
        d, lo, hi = paired_bootstrap_delta(
            yt, bert_test_pred, p, mt, args.n_resamples)
        sig = "SIGNIFIKANT" if (lo > 0 or hi < 0) else "n.s."
        results_ci[name] = {"delta": d, "ci": [lo, hi],
                            "significant": bool(lo > 0 or hi < 0)}
        print(f"  {name:<38} ΔS = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

    # ── Save ───────────────────────────────────────────────────────────────
    table.to_csv(out_dir / "perlabel_hybrid_results.csv", index=False)
    summary = {
        "routing_static": routing,
        "uncertainty_radius": r,
        "llm_fallback_rows": n_fallback,
        "frac_routed_to_llm": {l: float(frac_routed[j])
                               for j, l in enumerate(LABELS)},
        "paired_bootstrap_vs_bert": results_ci,
        "note": (
            "Static routing uses BERT val-S vs published E10 scores as "
            "prior (no test leakage). LLM-failed rows fall back to BERT."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()