import matplotlib.pyplot as plt
import seaborn as sns
# It seems the code execution environment has been reset, so I need to re-import pandas
import pandas as pd

# Re-load the Excel file to examine its structure
file_path = 'Case07-data.xlsx'
data = pd.read_excel(file_path)

# Cleaning and restructuring the data

# Removing unnecessary rows and columns
data_cleaned = data.drop(index=[0, 1, 2, 3, 4], columns=["Unnamed: 0", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"]).reset_index(drop=True)

# Renaming columns
column_names = ["Participant",
                "Hand_Mental_Demand", "Hand_Physical_Demand", "Hand_Frustration", "Hand_Perceived_Performance", "Hand_Overall_Effort",
                "Headset_Mental_Demand", "Headset_Physical_Demand", "Headset_Frustration", "Headset_Perceived_Performance", "Headset_Overall_Effort"]
data_cleaned.columns = column_names

# Convert data to appropriate types
data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')
data_cleaned.head()

# Creating side-by-side box plots for each TLX factor
tlx_factors = ['Mental_Demand', 'Physical_Demand', 'Frustration', 'Perceived_Performance', 'Overall_Effort']

# Creating a single plot with all TLX factors and both modalities on one axis

# Creating a DataFrame for all factors with a 'Modality' and 'Factor' column
all_factors_data = pd.DataFrame()

for factor in tlx_factors:
    temp_df = pd.DataFrame({
        'Score': pd.concat([data_cleaned[f'Hand_{factor}'], data_cleaned[f'Headset_{factor}']]),
        'Modality': ['Hand Controller'] * len(data_cleaned) + ['Headset'] * len(data_cleaned),
        'Factor': [factor.replace('_', ' ')] * len(data_cleaned) * 2
    })
    all_factors_data = pd.concat([all_factors_data, temp_df])

# Creating a single box plot for all factors
plt.figure(figsize=(12, 8))
sns.boxplot(x='Factor', y='Score', hue='Modality', data=all_factors_data, palette='Set1')
plt.title('Comparison of TLX Factor Scores by Modality')
plt.xlabel('Factor')
plt.ylabel('Score')
plt.legend(title='Modality')
plt.show()

# Adjusting the previous code to include interpretation of the results

from scipy.stats import ttest_rel, wilcoxon

# Lists to store p-values and interpretations
ttest_results = []
wilcoxon_results = []

for factor in tlx_factors:
    # Extracting scores for each modality
    hand_scores = data_cleaned[f'Hand_{factor}']
    headset_scores = data_cleaned[f'Headset_{factor}']

    # Performing the paired t-test
    t_stat, t_pval = ttest_rel(hand_scores, headset_scores)
    ttest_interpretation = "Significant difference" if t_pval < 0.05 else "No significant difference"
    ttest_results.append((factor, t_pval, ttest_interpretation))

    # Performing the Wilcoxon signed-rank test
    w_stat, w_pval = wilcoxon(hand_scores, headset_scores)
    wilcoxon_interpretation = "Significant difference" if w_pval < 0.05 else "No significant difference"
    wilcoxon_results.append((factor, w_pval, wilcoxon_interpretation))

# Displaying the results
print("Paired t-test results:")
for factor, pval, interpretation in ttest_results:
    print(f"{factor}: p-value = {pval}, Interpretation: {interpretation}")

print("\nWilcoxon test results:")
for factor, pval, interpretation in wilcoxon_results:
    print(f"{factor}: p-value = {pval}, Interpretation: {interpretation}")

# Creating a single plot with the distribution for each TLX factor
plt.figure(figsize=(15, 10))

# Loop through each TLX factor and add a subplot for its distribution
for i, factor in enumerate(tlx_factors, 1):
    plt.subplot(2, 3, i)  # Adjusting the layout for 5 subplots (2 rows, 3 columns)
    sns.histplot(data_cleaned, x=f'Hand_{factor}', kde=True, color='blue', label='Hand Controller', alpha=0.6)
    sns.histplot(data_cleaned, x=f'Headset_{factor}', kde=True, color='red', label='Headset', alpha=0.6)
    plt.title(f'Distribution of {factor.replace("_", " ")}')
    plt.legend()

plt.tight_layout()
plt.show()
