"""
Lab 3 - ANOVA Test using WorldEnergy.csv
Course: KQC7016 Data Analytics

Purpose:
This program demonstrates a one-way ANOVA test using the WorldEnergy.csv dataset.
The selected dependent variable is renewables_share_elec, which measures the
percentage share of electricity generation from renewable sources.

Research question:
Is the mean renewable share of electricity generation significantly different
among China, United States, India, Malaysia, and Germany from 2000 to 2024?

Teacher-provided statistical references:
- F-table.pdf: used as a manual reference for ANOVA significance checking.
- T-table.pdf: not used in the main one-way ANOVA decision because ANOVA uses
  the F distribution. It is kept as a reference for understanding t tests or
  pairwise comparisons, but the main Lab 3 test is based on the F statistic.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. File path and output folder
# ------------------------------------------------------------
DATA_FILE = "WorldEnergy.csv"          # Put WorldEnergy.csv in the same folder as this file
OUTPUT_DIR = "Lab3_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset and basic inspection
# ------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("===== BASIC DATASET INFORMATION =====")
print("Dataset shape:", df.shape)
print("First 10 columns:", list(df.columns[:10]))
print("Year range:", df["year"].min(), "to", df["year"].max())
print("Number of unique countries/regions:", df["country"].nunique())
print()

# ------------------------------------------------------------
# 3. Select data for ANOVA
# ------------------------------------------------------------
selected_countries = ["China", "United States", "India", "Malaysia", "Germany"]
selected_years = (2000, 2024)
selected_variable = "renewables_share_elec"

anova_df = df[
    (df["country"].isin(selected_countries))
    & (df["year"].between(selected_years[0], selected_years[1]))
][["country", "year", selected_variable, "fossil_share_elec", "carbon_intensity_elec"]].copy()

anova_df = anova_df.dropna(subset=[selected_variable])

print("===== SELECTED DATA FOR ANOVA =====")
print("Selected countries:", selected_countries)
print("Selected period:", selected_years[0], "to", selected_years[1])
print("Dependent variable:", selected_variable)
print("Number of observations after cleaning:", len(anova_df))
print()

anova_df.to_csv(os.path.join(OUTPUT_DIR, "anova_selected_data.csv"), index=False)

# ------------------------------------------------------------
# 4. Descriptive statistics
# ------------------------------------------------------------
summary_table = anova_df.groupby("country")[selected_variable].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
).round(3)

print("===== DESCRIPTIVE STATISTICS =====")
print(summary_table)
print()
summary_table.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

# ------------------------------------------------------------
# 5. Assumption checks
# ------------------------------------------------------------
groups = [
    group[selected_variable].values
    for country, group in anova_df.groupby("country")
]

levene_stat, levene_p = stats.levene(*groups)

print("===== ASSUMPTION CHECKS =====")
print("Levene test for equal variances:")
print(f"Statistic = {levene_stat:.4f}, p-value = {levene_p:.6f}")
print("Interpretation: p < 0.05 suggests unequal variances, so results should be discussed carefully.")
print()

print("Shapiro-Wilk normality test by country:")
for country, group in anova_df.groupby("country"):
    shapiro_stat, shapiro_p = stats.shapiro(group[selected_variable])
    print(f"{country:15s}: statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.6f}")
print()

# ------------------------------------------------------------
# 6. One-way ANOVA
# ------------------------------------------------------------
model = ols(f"{selected_variable} ~ C(country)", data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2).round(6)

f_stat, p_value = stats.f_oneway(*groups)

# Degrees of freedom for one-way ANOVA
number_of_groups = len(selected_countries)
number_of_observations = len(anova_df)
df_between = number_of_groups - 1
df_within = number_of_observations - number_of_groups

# Critical F values. These are the programmatic equivalent of checking F-table.pdf.
f_critical_005 = stats.f.ppf(1 - 0.05, df_between, df_within)
f_critical_001 = stats.f.ppf(1 - 0.01, df_between, df_within)

print("===== ONE-WAY ANOVA RESULT =====")
print(anova_table)
print()
print(f"SciPy ANOVA check: F-statistic = {f_stat:.4f}, p-value = {p_value:.10f}")
print(f"Degrees of freedom: df_between = {df_between}, df_within = {df_within}")
print(f"Critical F at alpha = 0.05: {f_critical_005:.4f}")
print(f"Critical F at alpha = 0.01: {f_critical_001:.4f}")

if f_stat >= f_critical_005:
    print("F-table decision at 0.05 level: F obtained is larger than F critical, so the result is significant.")
else:
    print("F-table decision at 0.05 level: F obtained is smaller than F critical, so the result is not significant.")

if p_value < 0.05:
    print("p-value decision: Reject H0. There is a statistically significant difference among group means.")
else:
    print("p-value decision: Fail to reject H0. There is no statistically significant difference among group means.")
print()

anova_table.to_csv(os.path.join(OUTPUT_DIR, "anova_table.csv"))

critical_values = pd.DataFrame({
    "alpha": [0.05, 0.01],
    "df_between": [df_between, df_between],
    "df_within": [df_within, df_within],
    "critical_F": [f_critical_005, f_critical_001],
    "obtained_F": [f_stat, f_stat],
    "significant_by_F_table": [f_stat >= f_critical_005, f_stat >= f_critical_001]
})
critical_values.to_csv(os.path.join(OUTPUT_DIR, "f_critical_values.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "statistical_table_notes.txt"), "w", encoding="utf-8") as f:
    f.write("F-table.pdf is relevant to this lab because one-way ANOVA uses the F distribution.\n")
    f.write("The code calculates the same kind of critical F values using scipy.stats.f.ppf.\n")
    f.write("T-table.pdf is not used for the main ANOVA decision because t-tests use the t distribution, while ANOVA uses the F distribution.\n")
    f.write("The T-table can be used as a general reference for t-tests or pairwise comparison concepts, but the main Lab 3 conclusion should be based on F and p-value.\n")

# ------------------------------------------------------------
# 7. Post-hoc Tukey HSD test
# ------------------------------------------------------------
tukey = pairwise_tukeyhsd(
    endog=anova_df[selected_variable],
    groups=anova_df["country"],
    alpha=0.05
)

print("===== TUKEY HSD POST-HOC TEST =====")
print(tukey)
print()

with open(os.path.join(OUTPUT_DIR, "tukey_hsd_result.txt"), "w", encoding="utf-8") as f:
    f.write(str(tukey))

# ------------------------------------------------------------
# 8. Visualization 1: Boxplot
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
anova_df.boxplot(column=selected_variable, by="country", grid=False)
plt.title("Renewable Share of Electricity Generation by Country (2000-2024)")
plt.suptitle("")
plt.xlabel("Country")
plt.ylabel("Renewables Share of Electricity (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure1_boxplot.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. Visualization 2: Line chart
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
for country in selected_countries:
    country_data = anova_df[anova_df["country"] == country]
    plt.plot(country_data["year"], country_data[selected_variable], marker="o", label=country)
plt.title("Renewable Share of Electricity Generation Trend (2000-2024)")
plt.xlabel("Year")
plt.ylabel("Renewables Share of Electricity (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure2_trend.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Visualization 3: Mean bar chart with standard deviation
# ------------------------------------------------------------
means = anova_df.groupby("country")[selected_variable].mean().reindex(selected_countries)
stds = anova_df.groupby("country")[selected_variable].std().reindex(selected_countries)

plt.figure(figsize=(9, 5))
plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
plt.title("Mean Renewable Electricity Share with Standard Deviation")
plt.xlabel("Country")
plt.ylabel("Mean Renewables Share of Electricity (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure3_mean_bar.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. Visualization 4: Histogram by group
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
for country in selected_countries:
    country_data = anova_df[anova_df["country"] == country][selected_variable]
    plt.hist(country_data, bins=8, alpha=0.45, label=country)
plt.title("Distribution of Renewable Electricity Share by Country")
plt.xlabel("Renewables Share of Electricity (%)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure4_histogram.png"), dpi=300)
plt.close()

print("===== OUTPUT FILES CREATED =====")
print(f"All result tables and figures are saved in: {OUTPUT_DIR}")
print("Figures created:")
print("1. figure1_boxplot.png")
print("2. figure2_trend.png")
print("3. figure3_mean_bar.png")
print("4. figure4_histogram.png")
print("Additional statistical table files created:")
print("5. f_critical_values.csv")
print("6. statistical_table_notes.txt")
