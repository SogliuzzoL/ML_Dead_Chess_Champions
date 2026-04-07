import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()

sns.set_theme(
    context="paper",
    style="ticks",
    font="STIXGeneral",
    palette="colorblind",
    rc={
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)
target_width = 3.5
target_height = 2.5


def jsd_heatmap(config: Config):
    df = pl.read_parquet(
        config.paths.get_distances_path(
            method=config.jsd.method, is_test=True, kde=True
        )
    )
    print(df)
    pdf = df.to_pandas()

    distance_col = "distance" if "distance" in pdf.columns else "jsd"
    matrix_df = pdf.pivot(index="p1", columns="p2", values=distance_col)

    fig, ax = plt.subplots(figsize=(target_width, target_height))

    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap="cividis",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Distance JSD"},
    )

    ax.set_title("Jensen,Shannon Divergence")
    ax.set_xlabel("Player")
    ax.set_ylabel("Player")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.savefig(
        config.paths.jsd_heatmap_path, format="pdf", dpi=300, bbox_inches="tight"
    )

    plt.close(fig)


def moves_distribution(config: Config):
    df = pl.read_parquet(config.paths.dataset_path)

    graph = sns.catplot(
        data=df.to_pandas(),
        x="player_name",
        kind="count",
        height=target_height,
        aspect=target_width / target_height,
    )

    graph.figure.suptitle("Distribution of Moves per Player")
    graph.set_axis_labels("Player", "Number of Moves")

    plt.setp(graph.ax.get_xticklabels(), rotation=45, ha="right")

    graph.savefig(
        config.paths.moves_distribution_graph_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def generate_all_graphics(config: Config):
    logger.info("Generating moves distribution graph...")
    moves_distribution(config)

    logger.info("Generating JSD heatmap...")
    jsd_heatmap(config)
