import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Load the DataFrame
df = pl.read_parquet("results/evaluation/player_blunders_detailed.parquet")

# Define the columns of interest for evaluation
cols = [
    "is_blunder_actual",
    "is_blunder_pred_maia2",
    "delta_pred_maia2",
    "is_blunder_pred_maia2_ft",
    "delta_pred_maia2_ft",
    "is_blunder_pred_maia2_ft_mcts",
    "delta_pred_maia2_ft_mcts",
]

# Group by player and compute the mean (rate) for each column
df_rates = df.group_by("player_name").agg(
    [pl.col(c).mean().alias(f"{c}_rate") for c in cols]
)

# Convert the Polars DataFrame to Pandas for LaTeX and plotting compatibility
df_pandas = df_rates.to_pandas()

# Generate the LaTeX table source code with a 3-decimal float format
latex_table = df_pandas.to_latex(index=False, float_format="%.3f")

print("Generated LaTeX source code:")
print(latex_table)

# Define X-axis parameters (player names and bar widths)
players = df_pandas["player_name"]
x = np.arange(len(players))
width = 0.2

# ==========================================
# 1. Bar Chart: Blunder Rates (is_blunder)
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(x - 1.5 * width, df_pandas["is_blunder_actual_rate"], width, label="Actual")
ax1.bar(x - 0.5 * width, df_pandas["is_blunder_pred_maia2_rate"], width, label="Maia2")
ax1.bar(
    x + 0.5 * width, df_pandas["is_blunder_pred_maia2_ft_rate"], width, label="Maia2 FT"
)
ax1.bar(
    x + 1.5 * width,
    df_pandas["is_blunder_pred_maia2_ft_mcts_rate"],
    width,
    label="Maia2 FT MCTS",
)

ax1.set_ylabel("Blunder Rate", fontsize=12)
ax1.set_title("Comparison of Blunder Rates across Models and Players", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(players, rotation=45, ha="right")
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("blunder_rates.pdf", dpi=300)
plt.show()

# ==========================================
# 2. Bar Chart: Evaluation Deltas (delta)
# ==========================================
fig, ax2 = plt.subplots(figsize=(12, 6))

ax2.bar(x - width, df_pandas["delta_pred_maia2_rate"], width, label="Maia2")
ax2.bar(x, df_pandas["delta_pred_maia2_ft_rate"], width, label="Maia2 FT")
ax2.bar(
    x + width, df_pandas["delta_pred_maia2_ft_mcts_rate"], width, label="Maia2 FT MCTS"
)

ax2.set_ylabel("Mean Evaluation Delta (Centipawns)", fontsize=12)
ax2.set_title(
    "Comparison of Mean Evaluation Deltas across Models and Players", fontsize=14
)
ax2.set_xticks(x)
ax2.set_xticklabels(players, rotation=45, ha="right")
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("delta_rates.pdf", dpi=300)
plt.show()
