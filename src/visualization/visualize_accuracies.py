import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.config import ProjectConfig


def visualize_player_accuracies(config: ProjectConfig):
    """
    Generates academic visualizations of the model's predictive accuracy
    across the evaluated chess champions.
    """
    input_path = os.path.join(
        config.result_folder, "player_accuracies_comparison.parquet"
    )

    if not os.path.exists(input_path):
        print(f"Error: The accuracy results file was not found at {input_path}")
        return

    df = pd.read_parquet(input_path)

    # Ensure the dataframe is sorted by accuracy for the bar chart
    df = df.sort_values("custom_accuracy", ascending=False)

    # Set the visual style for academic plotting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ---------------------------------------------------------
    # Plot 1: Ranked Bar Chart of Prediction Accuracy
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(data=df, x="player", y="custom_accuracy", palette="viridis")

    plt.title(
        "Move Prediction Accuracy per Historical Champion", pad=20, fontweight="bold"
    )
    plt.xlabel("Chess Champion", fontweight="bold")
    plt.ylabel("Accuracy (Ratio)", fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    # Add exact value labels on top of each bar for clarity
    for p in barplot.patches:
        barplot.annotate(
            format(p.get_height(), ".3f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=10,
        )

    plt.tight_layout()
    bar_output_path = os.path.join(config.result_folder, "bar_prediction_accuracy.pdf")
    plt.savefig(bar_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Scatter Plot (Accuracy vs. Number of Positions)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="n_positions",
        y="custom_accuracy",
        hue="custom_accuracy",
        palette="viridis",
        s=150,
        legend=False,
    )

    plt.title(
        "Impact of Dataset Volume on Prediction Accuracy", pad=20, fontweight="bold"
    )
    plt.xlabel("Number of Evaluated Positions", fontweight="bold")
    plt.ylabel("Prediction Accuracy", fontweight="bold")

    # Annotate points with the player's name
    for i in range(df.shape[0]):
        plt.text(
            df["n_positions"].iloc[i] + (df["n_positions"].max() * 0.01),
            df["custom_accuracy"].iloc[i],
            df["player"].iloc[i],
            fontsize=9,
        )

    plt.tight_layout()
    scatter_output_path = os.path.join(
        config.result_folder, "scatter_accuracy_vs_volume.pdf"
    )
    plt.savefig(scatter_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualizations successfully saved to {config.result_folder}")
