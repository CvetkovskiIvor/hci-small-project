import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

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


# Function to check normality for each TLX factor
def check_normality(data, factor_names):
    for factor in factor_names:
        print(f"\n--- Checking Normality for {factor} ---")

        # Combining scores from both modalities
        combined_scores = pd.concat([data[f'Hand_{factor}'], data[f'Headset_{factor}']])

        # Histogram
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(combined_scores, bins=20, alpha=0.7, color='blue')
        plt.title(f'Histogram of {factor}')

        # Q-Q Plot
        plt.subplot(1, 2, 2)
        stats.probplot(combined_scores, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {factor}')
        plt.show()

        # Shapiro-Wilk Test
        shapiro_test = stats.shapiro(combined_scores)
        print(f"\nShapiro-Wilk Test for {factor}: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
        print("Interpretation:", "Data is likely normal" if shapiro_test[1] > 0.05 else "Data is likely not normal")

        # D'Agostino's K^2 Test
        dagostino_test = stats.normaltest(combined_scores)
        print(f"\nD'Agostino's K^2 Test for {factor}: Statistic={dagostino_test[0]}, p-value={dagostino_test[1]}")
        print("Interpretation:", "Data is likely normal" if dagostino_test[1] > 0.05 else "Data is likely not normal")

        # Anderson-Darling Test
        anderson_test = stats.anderson(combined_scores, dist='norm')
        print(f"\nAnderson-Darling Test for {factor}: Statistic={anderson_test.statistic}")
        for i in range(len(anderson_test.critical_values)):
            sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
            if anderson_test.statistic > cv:
                print(f"At the {sl}% significance level, the data is likely not normal (statistic > critical value).")
            else:
                print(f"At the {sl}% significance level, the data is likely normal (statistic <= critical value).")


# Checking normality for each TLX factor
check_normality(data_cleaned, tlx_factors)

# Continuing from the provided code to include Z value calculation in Wilcoxon Signed-Rank Test

# Function to calculate the Z value for the Wilcoxon Signed-Rank Test
def calculate_wilcoxon_z(hand_scores, headset_scores):
    # Calculate the differences and their absolute values
    differences = hand_scores - headset_scores
    abs_diff = np.abs(differences)

    # Assign ranks to absolute differences
    ranks = abs_diff.rank()

    # Calculate the sum of ranks for positive and negative differences
    pos_rank_sum = np.sum(ranks[differences > 0])
    neg_rank_sum = np.sum(ranks[differences < 0])

    # Wilcoxon statistic (smaller of the two rank sums)
    W = min(pos_rank_sum, neg_rank_sum)

    # Number of non-zero differences
    n = np.count_nonzero(differences)

    # Expected value and standard deviation for W
    E_W = n * (n + 1) / 4
    StdDev_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    # Z value calculation
    Z = (W - E_W) / StdDev_W
    return Z

# Adding Z value calculation to the results
for factor in tlx_factors:
    # Extracting scores for each modality
    hand_scores = data_cleaned[f'Hand_{factor}']
    headset_scores = data_cleaned[f'Headset_{factor}']

    # Calculating the Z value for Wilcoxon Signed-Rank Test
    Z = calculate_wilcoxon_z(hand_scores, headset_scores)
    print(f"Wilcoxon Z value for {factor}: {Z}")

