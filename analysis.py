import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from task1 import demographic_profiles
from task2 import risk_score_distribution
from task3 import plot_ai_vs_judge_distribution_by_group
from task4 import reoffense_rates_fairness

claiborne_df = pd.read_csv("Claiborne_county_synthetic_data.csv")
copiah_df = pd.read_csv("Copiah_county_synthetic_data.csv")
warren_df = pd.read_csv("Warren_county_synthetic_data.csv")

### Task 1: Analyze the demographic profiles of three counties
gender_data, race_data, education_data = demographic_profiles(claiborne_df, copiah_df, warren_df)

### Task 2: Evaluate risk scores across demographic groups
summary_df = risk_score_distribution(claiborne_df, copiah_df, warren_df)
print(summary_df)

### Task 3: Compare judges’ bail decisions to AI risk scores
plot_ai_vs_judge_distribution_by_group(claiborne_df, copiah_df, warren_df)


### Task 4: Analyze re-offense rates and fairness metrics
reoffense_rates_fairness(claiborne_df, copiah_df, warren_df)