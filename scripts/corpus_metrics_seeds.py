"""
Corpus-level diversity metrics over several random subsamples

Self-BLEU-4, Distinct-1/2 and mean pairwise TF-IDF cosine depend on the corpus
size, so real reports are subsampled to the size of the synthetic groups
(N=700). A single subsample depends on which reports were drawn: this script
repeats it and reports mean ± std across subsamples. Two modes:

  independent  one draw without replacement per seed; subsamples may share
               reports. Sampling is identical to evaluate_real_reports.py
               (random.seed(seed) then random.sample over the sorted file
               list), so seed 42 gives back the values of the main evaluation.
  disjoint     the sorted file list is shuffled once (--split_seed), then cut
               into --n_folds consecutive folds of N: no report is in two folds
               (5 × 700 = 3500 of 3687 reports used).

In both modes the same seed selects the same reports in every input_dir when
they hold the same file names (e.g. raw vs punctuated reports).

Usage:
    python scripts/corpus_metrics_seeds.py \\
        --input_dirs "2. Corps Extraits" "3. CRs ponctues" \\
        --seeds 42 43 44 45 46 --sample_n 700 --output results/corpus_seeds.csv
    python scripts/corpus_metrics_seeds.py --mode disjoint \\
        --input_dirs "2. Corps Extraits" "3. CRs ponctues" \\
        --split_seed 42 --n_folds 5 --sample_n 700 --output results/corpus_folds.csv
"""

import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from evaluate_real_reports import (distinct_n, load_reports,
                                   mean_pairwise_cosine, self_bleu_4)

METRICS = ["self_bleu_4", "distinct_1", "distinct_2", "mean_pairwise_cosine"]


def corpus_metrics(job):
    input_dir, label, texts = job
    tag = "  ".join(f"{k}={v}" for k, v in label.items())
    print(f"  start  {input_dir}  {tag}", flush=True)
    row = {
        "input_dir":            input_dir,
        **label,
        "n_corpus":             len(texts),
        "self_bleu_4":          self_bleu_4(texts),
        "distinct_1":           distinct_n(texts, 1),
        "distinct_2":           distinct_n(texts, 2),
        "mean_pairwise_cosine": mean_pairwise_cosine(texts),
    }
    print(f"  done   {input_dir}  {tag}  self_bleu_4={row['self_bleu_4']:.4f}", flush=True)
    return row


def independent_subsamples(texts, seeds, n):
    """One independent draw without replacement per seed (subsamples may overlap)."""
    for seed in seeds:
        random.seed(seed)
        yield {"seed": seed}, random.sample(texts, n)


def disjoint_subsamples(texts, split_seed, n_folds, n):
    """Shuffle once, then cut into consecutive folds of n: no report in two folds."""
    if n_folds * n > len(texts):
        raise SystemExit(f"{n_folds} folds × {n} > {len(texts)} reports")
    random.seed(split_seed)
    order = random.sample(range(len(texts)), len(texts))
    for k in range(n_folds):
        yield ({"split_seed": split_seed, "fold": k + 1},
               [texts[i] for i in order[k * n:(k + 1) * n]])


def main():
    ap = argparse.ArgumentParser(description="Corpus metrics over several subsamples.")
    ap.add_argument("--input_dirs", nargs="+", required=True)
    ap.add_argument("--mode", choices=["independent", "disjoint"], default="independent",
                    help="independent: one draw per seed (may overlap); "
                         "disjoint: one shuffle cut into non-overlapping folds")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                    help="independent mode: one subsample per seed")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="disjoint mode: seed of the single shuffle")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="disjoint mode: number of folds")
    ap.add_argument("--sample_n", type=int, default=700)
    ap.add_argument("--output", required=True,
                    help="per-subsample CSV; a *_summary.csv (mean, std) is written alongside it")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = []
    for input_dir in args.input_dirs:
        texts = list(load_reports(Path(input_dir)).values())
        if args.mode == "independent":
            subsamples = independent_subsamples(texts, args.seeds, args.sample_n)
        else:
            subsamples = disjoint_subsamples(texts, args.split_seed, args.n_folds, args.sample_n)
        jobs += [(input_dir, label, sub) for label, sub in subsamples]

    print(f"{len(jobs)} {args.mode} subsample(s) of N={args.sample_n}, {args.workers} worker(s)")
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(corpus_metrics, jobs)

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8")

    summary = df.groupby("input_dir", sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    n_col = "n_seeds" if args.mode == "independent" else "n_folds"
    summary.insert(0, n_col, df.groupby("input_dir", sort=False).size())
    summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)
    summary.to_csv(summary_path, encoding="utf-8")

    print(f"\nPer-subsample values  →  {output_path}")
    print(f"Mean ± std            →  {summary_path}\n")
    for input_dir, r in summary.iterrows():
        print(f"  {input_dir}  (N={args.sample_n}, {int(r[n_col])} {args.mode} subsamples)")
        for m in METRICS:
            print(f"    {m:<22} {r[m + '_mean']:>9.4f}  ± {r[m + '_std']:.4f}")


if __name__ == "__main__":
    main()
