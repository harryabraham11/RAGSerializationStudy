import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("results/evaluation_results.csv")  # adjust path as needed

print("=" * 70)
print("WILCOXON SIGNED-RANK TEST: naive vs structured, per retriever")
print("=" * 70)

for retriever in df["retriever"].unique():
    sub = df[df["retriever"] == retriever]

    naive = sub[sub["corpus_condition"] == "naive"].set_index("question_id")["reciprocal_rank"]
    structured = sub[sub["corpus_condition"] == "structured"].set_index("question_id")["reciprocal_rank"]

    # align on question_id so pairs are guaranteed to match correctly
    paired = pd.concat([naive, structured], axis=1, keys=["naive", "structured"]).dropna()

    diffs = paired["naive"] - paired["structured"]
    n_pairs = len(paired)
    n_nonzero = (diffs != 0).sum()

    print(f"\n--- Retriever: {retriever} ---")
    print(f"n pairs = {n_pairs}, n nonzero differences = {n_nonzero}")
    print(f"mean naive MRR = {paired['naive'].mean():.4f}, mean structured MRR = {paired['structured'].mean():.4f}")

    if n_nonzero < 1:
        print("All differences are zero -- Wilcoxon test cannot be run.")
        continue

    stat, p_value = wilcoxon(paired["naive"], paired["structured"])
    print(f"Wilcoxon statistic = {stat:.4f}, p-value = {p_value:.6f}")

    # simple effect-size proxy: matched-pairs rank-biserial correlation
    r = 1 - (2 * stat) / (n_nonzero * (n_nonzero + 1))
    print(f"Matched-pairs rank-biserial correlation (effect size) = {r:.4f}")