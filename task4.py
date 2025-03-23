import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_outcome(row):
    if row['AI Prediction'] == 1 and row['Re-offense'] == 1:
        return 'TP'
    elif row['AI Prediction'] == 1 and row['Re-offense'] == 0:
        return 'FP'
    elif row['AI Prediction'] == 0 and row['Re-offense'] == 1:
        return 'FN'
    else:
        return 'TN'

def reoffense_rates_fairness(claiborne_df, copiah_df, warren_df):
    df = pd.concat([claiborne_df, copiah_df, warren_df], ignore_index=True)

    threshold = 5
    df['AI Prediction'] = df['Risk Score'].apply(lambda x: 1 if x >= threshold else 0)
    df['Outcome'] = df.apply(get_outcome, axis=1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))
    fig.subplots_adjust(hspace=0.4)

    for idx, county in enumerate(df['County'].unique()):
        county_df = df[df['County'] == county]
        results = []

        for race in county_df['Race'].unique():
            sub_df = county_df[county_df['Race'] == race]
            if sub_df.empty:
                continue

            TP = len(sub_df[sub_df['Outcome'] == 'TP'])
            FP = len(sub_df[sub_df['Outcome'] == 'FP'])
            FN = len(sub_df[sub_df['Outcome'] == 'FN'])
            TN = len(sub_df[sub_df['Outcome'] == 'TN'])

            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

            results.append({'Race': race, 'FPR': FPR, 'FNR': FNR})

        metrics_df = pd.DataFrame(results)
        metrics_df['Race'] = pd.Categorical(metrics_df['Race'], categories=['Black', 'White', 'Other'], ordered=True)
        metrics_df = metrics_df.sort_values('Race')
        x = np.arange(len(metrics_df))
        width = 0.35

        ax = axes[idx]
        ax.bar(x - width/2, metrics_df['FPR'], width, label='FPR', color='tomato')
        ax.bar(x + width/2, metrics_df['FNR'], width, label='FNR', color='steelblue')

        for i, val in enumerate(metrics_df['FPR']):
            ax.text(x[i] - width/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
        for i, val in enumerate(metrics_df['FNR']):
            ax.text(x[i] + width/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Rate')
        ax.set_title(f"{county} - FPR and FNR by Race", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Race'], rotation=45)
        ax.legend()

    plt.suptitle("False Positive Rate (FPR) and False Negative Rate (FNR) by Race Across Counties", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("all_counties_fpr_fnr_by_race.png", dpi=300)
    plt.close()