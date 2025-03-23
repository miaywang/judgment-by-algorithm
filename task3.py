import pandas as pd
import matplotlib.pyplot as plt

def plot_ai_vs_judge_distribution_by_group(claiborne_df, copiah_df, warren_df):
    df = pd.concat([claiborne_df, copiah_df, warren_df], ignore_index=True)
    df['Risk Category'] = df['Risk Score'].apply(categorize_risk)
    df['Risk Category'] = pd.Categorical(df['Risk Category'], categories=['Low', 'Medium', 'High'], ordered=True)
    for county in ['Claiborne County', 'Copiah County', 'Warren County']:
        plot_combined_race_gender_for_county(county, df)
    

def categorize_risk(score):
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'


def plot_combined_race_gender_for_county(county_name, df):
    county_df = df[df['County'] == county_name]
    races = county_df['Race'].unique()
    genders = county_df['Gender'].unique()

    total_cols = max(len(races), len(genders))

    fig, axes = plt.subplots(2, total_cols, figsize=(6 * total_cols, 10), sharey=True)

    if total_cols == 1:
        axes = axes.reshape(2, 1)

    # Race
    for i, race in enumerate(races):
        ax = axes[0, i]
        sub_df = county_df[county_df['Race'] == race]
        grouped = sub_df.groupby(['Risk Category', 'Judge Decision']).size().unstack(fill_value=0).reindex(['Low', 'Medium', 'High'])
        grouped = grouped.reset_index()

        grouped['Total'] = grouped[0] + grouped[1]
        grouped['Release Proportion'] = grouped[0] / grouped['Total']
        grouped['Detain Proportion'] = grouped[1] / grouped['Total']

        ax.bar(grouped['Risk Category'], grouped['Release Proportion'], label='Release (0)', color='tab:blue')
        ax.bar(grouped['Risk Category'], grouped['Detain Proportion'], bottom=grouped['Release Proportion'], label='Detain (1)', color='tab:orange')
        ax.set_title(f"{county_name} - Race: {race}")
        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Proportion")

        for idx, row in grouped.iterrows():
            release_pct = row['Release Proportion'] * 100
            detain_pct = row['Detain Proportion'] * 100

            ax.text(row['Risk Category'], row['Release Proportion'] / 2,
                    f"{int(row[0])} ({release_pct:.1f}%)",
                    ha='center', va='center', color='white', fontsize=9)

            ax.text(row['Risk Category'], row['Release Proportion'] + row['Detain Proportion'] / 2,
                    f"{int(row[1])} ({detain_pct:.1f}%)",
                    ha='center', va='center', color='white', fontsize=9)

    # Gender
    for i, gender in enumerate(genders):
        ax = axes[1, i]
        sub_df = county_df[county_df['Gender'] == gender]
        grouped = sub_df.groupby(['Risk Category', 'Judge Decision']).size().unstack(fill_value=0).reindex(['Low', 'Medium', 'High'])
        grouped = grouped.reset_index()

        grouped['Total'] = grouped[0] + grouped[1]
        grouped['Release Proportion'] = grouped[0] / grouped['Total']
        grouped['Detain Proportion'] = grouped[1] / grouped['Total']

        ax.bar(grouped['Risk Category'], grouped['Release Proportion'], label='Release (0)', color='tab:blue')
        ax.bar(grouped['Risk Category'], grouped['Detain Proportion'], bottom=grouped['Release Proportion'], label='Detain (1)', color='tab:orange')
        ax.set_title(f"{county_name} - Gender: {gender}")
        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Proportion")

        for idx, row in grouped.iterrows():
            release_pct = row['Release Proportion'] * 100
            detain_pct = row['Detain Proportion'] * 100

            ax.text(row['Risk Category'], row['Release Proportion'] / 2,
                    f"{int(row[0])} ({release_pct:.1f}%)",
                    ha='center', va='center', color='white', fontsize=9)

            ax.text(row['Risk Category'], row['Release Proportion'] + row['Detain Proportion'] / 2,
                    f"{int(row[1])} ({detain_pct:.1f}%)",
                    ha='center', va='center', color='white', fontsize=9)

    if len(races) < total_cols:
        for i in range(len(races), total_cols):
            fig.delaxes(axes[0, i])
    if len(genders) < total_cols:
        for i in range(len(genders), total_cols):
            fig.delaxes(axes[1, i])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Judge Decisions vs AI Risk Score by Race and Gender in {county_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig(f"{county_name.replace(' ', '_').lower()}_race_gender_decision_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()