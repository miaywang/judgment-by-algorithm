import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def risk_score_distribution(claiborne_df, copiah_df, warren_df):
    claiborne_df["County"] = "Claiborne"
    copiah_df["County"] = "Copiah"
    warren_df["County"] = "Warren"

    combined_df = pd.concat([claiborne_df, copiah_df, warren_df], ignore_index=True)

    summary_df = (
        combined_df.groupby(["County", "Race", "Gender"])["Risk Score"]
        .mean()
        .reset_index()
        .rename(columns={"Risk Score": "Average Risk Score"})
    )

    g = sns.catplot(
        data=summary_df,
        kind="bar",
        x="County",
        y="Average Risk Score",
        hue="Gender",
        col="Race",
        palette="pastel",
        height=5,
        aspect=1,
        ci=None
    )

    for ax in g.axes.flat:
        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle("Average Risk Score by Race and Gender in Each County")
    g.set_axis_labels("County", "Average Risk Score")
    g.set_titles(col_template="{col_name}")
    g.tight_layout()
    g.savefig("risk_score_distribution_by_county_facet.png", dpi=300, bbox_inches='tight')
    plt.close()

    return summary_df