import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height * 100:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black')


def demographic_profiles(claiborne_df, copiah_df, warren_df):
    claiborne_df["County"] = "Claiborne"
    copiah_df["County"] = "Copiah"
    warren_df["County"] = "Warren"

    combined_df = pd.concat([claiborne_df, copiah_df, warren_df], ignore_index=True)

    ### Gender ###
    gender_pivot = (
        combined_df.pivot_table(index="County", columns="Gender", aggfunc="size", fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
        .reset_index()
    )
    gender_melted = gender_pivot.melt(id_vars="County", var_name="Gender", value_name="Proportion")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=gender_melted, x="County", y="Proportion", hue="Gender")
    annotate_bars(ax)
    plt.title("Gender Distribution by County")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig("gender_distribution_by_county.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Race ###
    race_pivot = (
        combined_df.pivot_table(index="County", columns="Race", aggfunc="size", fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
        .reset_index()
    )
    race_melted = race_pivot.melt(id_vars="County", var_name="Race", value_name="Proportion")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=race_melted, x="County", y="Proportion", hue="Race")
    annotate_bars(ax)
    plt.title("Racial Distribution by County")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend(title="Race")
    plt.tight_layout()
    plt.savefig("racial_distribution_by_county.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Education Level ###
    edu_pivot = (
        combined_df.pivot_table(index="County", columns="Education Level", aggfunc="size", fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
        .reset_index()
    )
    edu_melted = edu_pivot.melt(id_vars="County", var_name="Education Level", value_name="Proportion")

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(data=edu_melted, x="County", y="Proportion", hue="Education Level")
    annotate_bars(ax)
    plt.title("Education Level Distribution by County")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend(title="Education Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("education_distribution_by_county.png", dpi=300, bbox_inches='tight')
    plt.close()

    return gender_pivot, race_pivot, edu_pivot