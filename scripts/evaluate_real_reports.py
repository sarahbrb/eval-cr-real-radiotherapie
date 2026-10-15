"""
Linguistic quality evaluation for a set of medical reports (.txt files)

Designed to evaluate real clinical reports and compare them to synthetic ones
produced by the CAG pipeline (see evaluate_cag.py for the synthetic side)

Metrics (per report, computed on ALL reports):
  n_mots          word count
  msttr_100       lexical richness (length-corrected TTR, 100-word windows)
  flesch_re_fr    readability (French Flesch Reading Ease)
  gunning_fog_fr  readability (Gunning Fog index; higher = more complex)
  perplexite      surface fluency via a French causal LM

Metrics (corpus-level, computed on --sample_n reports if set):
  self_bleu_4          diversity; higher = reports look too similar
  distinct_1/2         unique unigram/bigram ratio (higher = more diverse)
  mean_pairwise_cosine mean TF-IDF cosine similarity (lower = more diverse)

  ⚠ Corpus-level metrics depend on N. When comparing to synthetic results
    (typically N=700), pass --sample_n 700 so the corpus sizes match.

Usage:
    python scripts/evaluate_real_reports.py \\
        --input_dir /path/to/reports/ \\
        --output results.csv \\
        --lm_model asi/gpt-fr-cased-base \\
        --device cuda \\
        --sample_n 700
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def load_reports(input_dir: Path) -> dict:
    """Load all .txt files from a directory. Returns {filename: text}."""
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in: {input_dir}")
    reports = {}
    for path in files:
        # Text passed as is (no strip), like cr_text in evaluate_cag.py
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"Skipping empty file: {path.name}")
            continue
        reports[path.name] = text
    print(f"{len(reports)} report(s) loaded from {input_dir}")
    return reports


def tokenize_words(text: str) -> list:
    return [w.lower() for w in WORD_RE.findall(text)]


def ttr(tokens: list) -> float:
    return len(set(tokens)) / len(tokens) if tokens else float("nan")


def msttr_100(tokens: list, window: int = 100) -> float:
    """Mean Segmental TTR over non-overlapping 100-word windows.
    Avoids the length bias of raw TTR (Covington & McFall, 2010)."""
    if not tokens:
        return float("nan")
    if len(tokens) < window:
        return ttr(tokens)
    chunks = [tokens[i : i + window]
              for i in range(0, len(tokens), window)
              if i + window <= len(tokens)]
    return float(np.mean([ttr(c) for c in chunks]))


# Readability: same textstat calls as evaluate_cag.py (synthetic side), so that
# real and synthetic scores are computed identically.
def flesch_re_fr(text: str) -> float:
    try:
        import textstat
        textstat.set_lang("fr")
        return float(textstat.flesch_reading_ease(text))
    except Exception as e:
        print(f"[warn] flesch_re_fr failed: {e}")
        return float("nan")


def gunning_fog_fr(text: str) -> float:
    try:
        import textstat
        textstat.set_lang("fr")
        return float(textstat.gunning_fog(text))
    except Exception as e:
        print(f"[warn] gunning_fog_fr failed: {e}")
        return float("nan")


def distinct_n(texts: list, n: int) -> float:
    """Corpus-level Distinct-n: unique n-grams / total n-grams (Li et al., 2016).
    Higher = more lexically diverse."""
    total, unique = 0, set()
    for text in texts:
        tokens = tokenize_words(text)
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(ngrams)
        unique.update(ngrams)
    if total == 0:
        return float("nan")
    return len(unique) / total


def mean_pairwise_cosine(texts: list) -> float:
    """Mean pairwise TF-IDF cosine similarity over all pairs in the corpus.
    Lower = more semantically diverse."""
    if len(texts) < 2:
        return float("nan")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("scikit-learn not installed — run: pip install scikit-learn")
        return float("nan")

    vecs = TfidfVectorizer().fit_transform(texts)
    sim  = cosine_similarity(vecs)
    n    = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    return float(upper.mean())


class FluencyScorer:
    """Perplexity scoring using a French causal language model."""

    def __init__(self, model_name: str, device: str):
        print(f"Loading fluency model: {model_name}")
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """Texts over 1024 tokens are truncated (consistent with evaluate_cag.py)."""
        enc      = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return float("nan")
        out = self.model(input_ids, labels=input_ids)
        return float(np.exp(out.loss.item()))


def self_bleu_4(texts: list) -> float:
    """Mean corpus Self-BLEU-4: each report scored against all others as references.
    Higher = less diverse corpus (Zhu et al., 2018)."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        print("sacrebleu not installed — run: pip install sacrebleu")
        return float("nan")

    if len(texts) < 2:
        print("Self-BLEU requires at least 2 reports.")
        return float("nan")

    bleu   = BLEU(max_ngram_order=4, effective_order=True)
    scores = [
        bleu.sentence_score(hyp, texts[:i] + texts[i + 1:]).score
        for i, hyp in enumerate(texts)
    ]
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(
        description="Linguistic quality evaluation of medical reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/evaluate_real_reports.py"
            " --input_dir ./reports/ --output results.csv"
            " --lm_model asi/gpt-fr-cased-base --device cuda --sample_n 700"
        ),
    )
    parser.add_argument("--input_dir", required=True,
                        help="directory of .txt reports (UTF-8, one file per report)")
    parser.add_argument("--output", required=True,
                        help="output CSV path; a *_summary.csv is also written alongside it")
    parser.add_argument("--lm_model", default="asi/gpt-fr-cased-base",
                        help="HuggingFace causal LM for perplexity (default: asi/gpt-fr-cased-base)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="'cuda' (recommended) or 'cpu'")
    parser.add_argument("--sample_n", type=int, default=None,
                        help=(
                            "If set, randomly sample this many reports for corpus-level "
                            "diversity metrics (self_bleu_4, distinct_1/2, mean_pairwise_cosine). "
                            "Per-report metrics (n_mots, msttr, flesch, fog, perplexity) are "
                            "always computed on ALL reports. "
                            "Use the same N as your synthetic corpora for a fair comparison "
                            "(e.g. --sample_n 700)."
                        ))
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample_n subsampling (default: 42)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Directory not found: {input_dir}")

    output_path  = Path(args.output)
    summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)

    reports = load_reports(input_dir)
    texts   = list(reports.values())
    n_total = len(texts)

    # ── Per-report surface metrics (no GPU needed) ────────────────────────────
    rows = []
    for fname, text in reports.items():
        tokens = tokenize_words(text)
        rows.append({
            "file":           fname,
            "n_mots":         len(tokens),
            "msttr_100":      msttr_100(tokens),
            "flesch_re_fr":   flesch_re_fr(text),
            "gunning_fog_fr": gunning_fog_fr(text),
        })
    df = pd.DataFrame(rows)

    # ── Perplexity (GPU) — computed on all reports ────────────────────────────
    print(f"Computing perplexity on {n_total} reports ({args.device}) ...")
    scorer   = FluencyScorer(args.lm_model, args.device)
    ppl_vals = []
    for i, (fname, text) in enumerate(reports.items()):
        ppl = scorer.perplexity(text)
        ppl_vals.append(ppl)
        print(f"  [{i+1}/{n_total}] {fname}  PPL={ppl}")
    df["perplexite"] = ppl_vals

    # ── Corpus-level diversity metrics (optionally subsampled) ────────────────
    if args.sample_n is not None and args.sample_n < n_total:
        random.seed(args.seed)
        corpus_texts = random.sample(texts, args.sample_n)
        corpus_n     = args.sample_n
        print(f"\nCorpus-level metrics on {corpus_n} randomly sampled reports "
              f"(seed={args.seed}) out of {n_total} ...")
    else:
        corpus_texts = texts
        corpus_n     = n_total
        if args.sample_n is not None:
            print(f"\n[info] --sample_n={args.sample_n} >= total ({n_total}), using all reports.")
        print(f"\nComputing corpus-level diversity metrics on all {corpus_n} reports ...")

    sb4 = self_bleu_4(corpus_texts)
    d1  = distinct_n(corpus_texts, 1)
    d2  = distinct_n(corpus_texts, 2)
    mpc = mean_pairwise_cosine(corpus_texts)

    # ── Save results ──────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nPer-report detail  →  {output_path}  (N={n_total})")

    metric_cols = ["n_mots", "msttr_100", "flesch_re_fr", "gunning_fog_fr", "perplexite"]
    summary = df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary.index.name = f"metric (N_reports={n_total})"
    # Corpus-level metrics: single value, no std (corpus-wide, not per-report)
    corpus_label = f"N_corpus={corpus_n}"
    summary.loc[f"self_bleu_4 ({corpus_label})"]          = [sb4,  float("nan")]
    summary.loc[f"distinct_1 ({corpus_label})"]           = [d1,   float("nan")]
    summary.loc[f"distinct_2 ({corpus_label})"]           = [d2,   float("nan")]
    summary.loc[f"mean_pairwise_cosine ({corpus_label})"] = [mpc,  float("nan")]
    summary.to_csv(summary_path, encoding="utf-8")
    print(f"Corpus summary     →  {summary_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"RÉSULTATS — {n_total} rapports réels")
    print(f"{'='*65}")
    print(f"\n  Métriques per-rapport (N={n_total}) :")
    labels = {
        "n_mots":         "Longueur           (n_mots)",
        "msttr_100":      "Richesse lexicale  (MSTTR-100)",
        "flesch_re_fr":   "Lisibilité         (Flesch RE)",
        "gunning_fog_fr": "Complexité         (Gunning Fog)",
        "perplexite":     "Fluidité           (Perplexité, ↓ = meilleur)",
    }
    means = df[metric_cols].mean()
    stds  = df[metric_cols].std()
    for col, label in labels.items():
        if pd.isna(means[col]):
            print(f"    {label:<45}  n/a")
        else:
            print(f"    {label:<45}  {means[col]:>8.3f}  ± {stds[col]:.3f}")

    print(f"\n  Métriques de diversité corpus (N={corpus_n}"
          + (f", sous-échantillon seed={args.seed}" if corpus_n < n_total else "") + ") :")
    print(f"    {'Diversité (Self-BLEU-4,  ↓ = plus divers)':<45}  {sb4:>8.2f}")
    print(f"    {'Diversité (Distinct-1,   ↑ = plus divers)':<45}  {d1:>8.4f}")
    print(f"    {'Diversité (Distinct-2,   ↑ = plus divers)':<45}  {d2:>8.4f}")
    print(f"    {'Similarité (Cosine TF-IDF, ↓ = plus divers)':<45}  {mpc:>8.4f}")

    if corpus_n < n_total:
        print(f"\n  ⚠  Les métriques corpus ont été calculées sur {corpus_n}/{n_total} rapports.")
        print(f"     Passer --sample_n identique à N de vos corpus synthétiques pour comparer.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
