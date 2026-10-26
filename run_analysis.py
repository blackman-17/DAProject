"""
WHAT CAUSES STRESS?  Data Analytics project, MobileWell100+ dataset
====================================================================
Run:   python run_analysis.py

PART 1  Data preprocessing
PART 2  Exploratory data analysis (EDA)
PART 3  Confirmatory data analysis (CDA): 12 hypothesis tests using
        t-tests, z-tests, one-way ANOVA and the chi-square test

Stress = 'tense_arousal' from the daily survey (0 = very calm, 100 = very restless).
All charts and tables are saved in the 'results' folder.
"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")        # so symbols like χ² print on every computer

# install any missing library automatically (pandas, numpy, scipy, matplotlib, seaborn)
for library in ["pandas", "numpy", "scipy", "matplotlib", "seaborn"]:
    try:
        __import__(library)
    except ImportError:
        print(f"Installing {library} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                            # save charts to files, do not open windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
(RESULTS / "eda").mkdir(parents=True, exist_ok=True)
(RESULTS / "hypotheses").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
COLORS = ["#2A9D8F", "#E76F51", "#6D597A"]


def save_chart(path):
    plt.tight_layout()
    plt.savefig(RESULTS / path, dpi=150)
    plt.close()


# =====================================================================================
# PART 1: DATA PREPROCESSING
# =====================================================================================
print("PART 1: Preprocessing ...")

# ---- 1.1 Daily survey: one row per stress report --------------------------------------
reports = pd.read_csv(DATA / "daily_survey.csv")
reports = reports[["participant", "timestamp", "occasion", "valence", "energetic_arousal",
                   "tense_arousal", "emotional_event"]]
print("  missing values:", reports.isna().sum().sum(), "| duplicates:", reports.duplicated().sum())

reports["time"] = pd.to_datetime(reports["timestamp"], format="ISO8601")
# phone data is taken from the full clock hour BEFORE the report (report at 14:37 -> 13:00-14:00)
reports["hour"] = reports["time"].dt.floor("h") - pd.Timedelta(hours=1)


# ---- 1.2 Labels: compare each report with the SAME person's median -----------------------
def compare_with_own_median(column, above, below):
    own_median = reports.groupby("participant")[column].transform("median")
    label = pd.Series(None, index=reports.index, dtype="object")
    label[reports[column] > own_median] = above
    label[reports[column] < own_median] = below
    return label                                        # reports equal to the median stay empty


reports["stress_level"] = compare_with_own_median("tense_arousal", "High stress", "Low stress")
reports["energy_level"] = compare_with_own_median("energetic_arousal", "Awake", "Tired")
reports["mood"] = compare_with_own_median("valence", "Good mood", "Bad mood")

reports["event"] = "Pleasant event"
reports.loc[reports["emotional_event"] == 0, "event"] = "No event"
reports.loc[reports["emotional_event"] < 0, "event"] = "Unpleasant event"

reports["day_type"] = "Weekday"
reports.loc[reports["time"].dt.dayofweek >= 5, "day_type"] = "Weekend"

# ---- 1.3 Phone sensors, summarised per person per hour -----------------------------------
screen = pd.read_csv(DATA / "screen.csv")
screen["hour"] = pd.to_datetime(screen["timestamp"], format="ISO8601").dt.floor("h")
screen["screen_on"] = screen["screen_event"] == "SCREEN_ON"
screen_hourly = screen.groupby(["participant", "hour"])["screen_on"].sum().reset_index()

# activity is already summarised per hour (see data_preparation/make_activity_hourly.py)
activity_hourly = pd.read_csv(DATA / "activity_hourly.csv", parse_dates=["hour"])

reports = reports.merge(screen_hourly, on=["participant", "hour"], how="left")
reports = reports.merge(activity_hourly, on=["participant", "hour"], how="left")

# no screen event in an hour = screen turned on 0 times (only for phones that recorded screen data)
has_screen_data = reports["participant"].isin(screen["participant"])
reports.loc[has_screen_data, "screen_on"] = reports.loc[has_screen_data, "screen_on"].fillna(0)

# ---- 1.4 Initial survey: one row per participant ---------------------------------------
people = pd.read_csv(DATA / "initial_survey.csv")
people["anxiety_score"] = people[[f"GAD_{i}" for i in range(1, 8)]].sum(axis=1)       # GAD-7, 0-21
people["depression_score"] = people[[f"PHQ_{i}" for i in range(1, 10)]].sum(axis=1)   # PHQ-9, 0-27
# Brief Resilience Scale: items 2, 4, 6 are reverse-scored (6 - answer), then averaged (1-5)
people["resilience_score"] = (people["BRS_1"] + people["BRS_3"] + people["BRS_5"]
                              + (6 - people["BRS_2"]) + (6 - people["BRS_4"]) + (6 - people["BRS_6"])) / 6
# negative feelings (PANAS): upset, hostile, ashamed, nervous, afraid (5-25)
people["negative_feelings"] = people[["IPSF_1", "IPSF_2", "IPSF_4", "IPSF_6", "IPSF_9"]].sum(axis=1)

average_stress = reports.groupby("participant")["tense_arousal"].mean().rename("average_stress")
people = people.merge(average_stress, on="participant")

# groups used by the tests
people["anxiety_group"] = np.where(people["anxiety_score"] >= 10, "Anxiety (GAD-7 ≥ 10)", "No anxiety (GAD-7 < 10)")
people["depression_group"] = np.where(people["depression_score"] >= 10, "Depression (PHQ-9 ≥ 10)", "No depression (PHQ-9 < 10)")
people["resilience_group"] = np.where(people["resilience_score"] < 3, "Low resilience (< 3)", "Normal / high resilience")
people["mental_health_group"] = people["MENT_LS"].map({1: "Dissatisfied", 2: "Dissatisfied",
                                                       3: "Satisfied", 4: "Very satisfied"})
people["negative_feelings_group"] = pd.cut(people["negative_feelings"], bins=[4, 7, 11, 25],
                                           labels=["Low (5-7)", "Medium (8-11)", "High (12-25)"]).astype(str)
# stress level of each PERSON: above / below the median of all participants, and in thirds
people["high_stress_person"] = people["average_stress"] > people["average_stress"].median()
people["stress_third"] = pd.qcut(people["average_stress"], 3,
                                 labels=["Low stress", "Medium stress", "High stress"]).astype(str)

reports.to_csv(RESULTS / "clean_reports.csv", index=False, encoding="utf-8-sig")
people.to_csv(RESULTS / "clean_participants.csv", index=False, encoding="utf-8-sig")
print(f"  {len(reports)} reports from {reports['participant'].nunique()} participants")


# =====================================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# =====================================================================================
print("PART 2: EDA ...")

columns = ["tense_arousal", "valence", "energetic_arousal", "screen_on", "pct_in_vehicle", "pct_still"]
reports[columns].describe().round(2).to_csv(RESULTS / "descriptive_statistics.csv", encoding="utf-8-sig")
print(reports[columns].describe().round(1).to_string())

plt.figure(figsize=(8, 5))
sns.histplot(reports["tense_arousal"], bins=30, color="#264653")
plt.title("Stress scores of all reports")
plt.xlabel("Tense arousal (0 = very calm, 100 = very restless)")
save_chart("eda/1_stress_histogram.png")

plt.figure(figsize=(8, 5))
sns.histplot(people["average_stress"], bins=20, color="#264653")
plt.title("Average stress per participant")
plt.xlabel("Average tense arousal")
save_chart("eda/2_average_stress_per_person.png")

times = {1: "08:00", 2: "10:00", 3: "13:00", 4: "16:00", 5: "19:00", 6: "22:00"}
reports["survey_time"] = reports["occasion"].map(times)
plt.figure(figsize=(8, 5))
sns.pointplot(data=reports, x="survey_time", y="tense_arousal", order=list(times.values()), color=COLORS[1])
plt.title("Average stress at each survey time (95% CI)")
plt.xlabel("Survey time")
plt.ylabel("Average tense arousal")
save_chart("eda/3_stress_by_time_of_day.png")

names = {"tense_arousal": "Stress", "valence": "Mood", "energetic_arousal": "Energy",
         "emotional_event": "Emotional event", "screen_on": "Screen on", "pct_in_vehicle": "% in vehicle",
         "pct_still": "% still"}
plt.figure(figsize=(8, 6.5))
sns.heatmap(reports[list(names)].rename(columns=names).corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1)
plt.title("Correlation between report variables")
save_chart("eda/4_correlation_heatmap.png")

plt.figure(figsize=(7, 4))
missing = reports[["screen_on", "pct_in_vehicle"]].isna().mean() * 100
sns.barplot(x=missing.values, y=["Screen", "Activity"], color="#E9C46A")
plt.title("Reports without phone data in the hour before")
plt.xlabel("% of reports missing")
save_chart("eda/5_missing_phone_data.png")


# =====================================================================================
# PART 3: CONFIRMATORY DATA ANALYSIS (12 hypotheses)
# =====================================================================================
print("PART 3: Hypothesis tests ...")
results = []


def save_result(hid, title, test, n, statistic, p, effect, ci=(np.nan, np.nan)):
    results.append({"id": hid, "hypothesis": title, "test": test, "n": n, "statistic": statistic,
                    "p_value": p, "effect": effect, "ci_low": ci[0], "ci_high": ci[1],
                    "decision": "Reject H0" if p < 0.05 else "Fail to reject H0"})


def per_person_means(value, label_column, a, b):
    """Average of `value` for each participant in situation a and in situation b (>= 3 reports each)."""
    data = reports.dropna(subset=[value, label_column])
    data = data[data[label_column].isin([a, b])]
    counts = data.groupby(["participant", label_column]).size().unstack()
    keep = counts[(counts[a] >= 3) & (counts[b] >= 3)].index
    return data[data["participant"].isin(keep)].groupby(["participant", label_column])[value].mean().unstack()


def paired_chart(hid, title, table, a, b, y_label, subtitle):
    long = table.reset_index().melt(id_vars="participant", value_name="value")
    plt.figure(figsize=(7, 5.5))
    sns.boxplot(data=long, x=long.columns[1], y="value", order=[a, b], hue=long.columns[1], legend=False,
                palette={a: COLORS[0], b: COLORS[1]}, showfliers=False, showmeans=True, width=0.5)
    sns.stripplot(data=long, x=long.columns[1], y="value", order=[a, b], color="black", alpha=0.35)
    plt.title(f"{hid}: {title}\n{subtitle}")
    plt.xlabel("")
    plt.ylabel(y_label)
    save_chart(f"hypotheses/{hid}.png")


# ---- Paired t-test: the SAME people in two situations ------------------------------------
def paired_t_test(hid, title, value, label_column, a, b, y_label):
    table = per_person_means(value, label_column, a, b)
    test = stats.ttest_rel(table[b], table[a])               # difference = b - a
    ci = test.confidence_interval()
    difference = (table[b] - table[a]).mean()
    paired_chart(hid, title, table, a, b, y_label,
                 f"Paired t-test: t({test.df}) = {test.statistic:.2f}, p = {test.pvalue:.3g}")
    save_result(hid, title, "Paired t-test", len(table), f"t({test.df}) = {test.statistic:.2f}",
                test.pvalue, f"mean difference = {difference:.2f}", (ci.low, ci.high))


# ---- A. Phone sensors: the hour before a HIGH- vs LOW-stress report ------------------------
paired_t_test("H1", "Screen turned on more before high stress", "screen_on", "stress_level",
              "Low stress", "High stress", "Times screen turned on (hour before)")
paired_t_test("H2", "More time in a vehicle before high stress", "pct_in_vehicle", "stress_level",
              "Low stress", "High stress", "% of the hour before in a vehicle")
paired_t_test("H3", "Less time sitting still before high stress", "pct_still", "stress_level",
              "Low stress", "High stress", "% of the hour before sitting still")

# ---- B. Daily life: the same person in two situations -------------------------------------
paired_t_test("H4", "Unpleasant events raise stress", "tense_arousal", "event",
              "No event", "Unpleasant event", "Average stress")

# H5: z-test for the mean difference (large sample, n > 30)
table = per_person_means("tense_arousal", "day_type", "Weekend", "Weekday")
d = table["Weekday"] - table["Weekend"]
n = len(d)
standard_error = d.std() / np.sqrt(n)
z = d.mean() / standard_error
p = 2 * stats.norm.sf(abs(z))
paired_chart("H5", "Weekdays are more stressful than weekends", table, "Weekend", "Weekday", "Average stress",
             f"z-test (n = {n}): z = {z:.2f}, p = {p:.3g}")
save_result("H5", "Weekdays are more stressful than weekends", "z-test (mean difference)", n, f"z = {z:.2f}", p,
            f"mean difference = {d.mean():.2f}", (d.mean() - 1.96 * standard_error, d.mean() + 1.96 * standard_error))

paired_t_test("H6", "More stress when tired", "tense_arousal", "energy_level", "Awake", "Tired", "Average stress")
paired_t_test("H7", "More stress when in a bad mood", "tense_arousal", "mood", "Good mood", "Bad mood",
              "Average stress")

# ---- C. Personal factors -----------------------------------------------------------------
# H8: two-proportion z-test. Share of people with above-median stress, anxiety vs no anxiety
with_anxiety = people[people["anxiety_group"] == "Anxiety (GAD-7 ≥ 10)"]["high_stress_person"]
without_anxiety = people[people["anxiety_group"] == "No anxiety (GAD-7 < 10)"]["high_stress_person"]
p1, n1 = with_anxiety.mean(), len(with_anxiety)
p2, n2 = without_anxiety.mean(), len(without_anxiety)
p_pooled = (with_anxiety.sum() + without_anxiety.sum()) / (n1 + n2)
z = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
p = 2 * stats.norm.sf(abs(z))
se_difference = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

plt.figure(figsize=(7, 5.5))
shares = people.groupby("anxiety_group")["high_stress_person"].mean().mul(100).reindex(
    ["No anxiety (GAD-7 < 10)", "Anxiety (GAD-7 ≥ 10)"])
sns.barplot(x=shares.index, y=shares.values, hue=shares.index, palette=COLORS[:2], legend=False)
for i, value in enumerate(shares.values):
    plt.text(i, value + 1, f"{value:.0f}%", ha="center")
plt.title(f"H8: People with anxiety are more often highly stressed\nTwo-proportion z-test: z = {z:.2f}, p = {p:.3g}")
plt.xlabel("")
plt.ylabel("% of people with above-median stress")
plt.ylim(0, 100)
save_chart("hypotheses/H8.png")
save_result("H8", "People with anxiety are more often highly stressed", "Two-proportion z-test", n1 + n2,
            f"z = {z:.2f}", p, f"{p1:.0%} vs {p2:.0%}", (p1 - p2 - 1.96 * se_difference, p1 - p2 + 1.96 * se_difference))

# H9: chi-square test of independence. Depression group x stress level (thirds)
crosstab = pd.crosstab(people["depression_group"], people["stress_third"])[["Low stress", "Medium stress", "High stress"]]
chi2, p, dof, expected = stats.chi2_contingency(crosstab)
cramers_v = np.sqrt(chi2 / (crosstab.values.sum() * (min(crosstab.shape) - 1)))
crosstab.to_csv(RESULTS / "H9_crosstab.csv", encoding="utf-8-sig")

plt.figure(figsize=(7, 5.5))
sns.countplot(data=people, x="depression_group", hue="stress_third", palette=COLORS,
              hue_order=["Low stress", "Medium stress", "High stress"],
              order=["No depression (PHQ-9 < 10)", "Depression (PHQ-9 ≥ 10)"])
plt.title(f"H9: Depression symptoms and stress level\nChi-square: χ²({dof}) = {chi2:.2f}, p = {p:.3g}")
plt.xlabel("")
plt.ylabel("Number of participants")
plt.legend(title="Stress level (thirds)")
save_chart("hypotheses/H9.png")
save_result("H9", "Depression symptoms are linked to stress level", "Chi-square test", int(crosstab.values.sum()),
            f"χ²({dof}) = {chi2:.2f}", p, f"Cramér's V = {cramers_v:.2f}")

# H10: independent-samples t-test (Welch: does not assume equal variances)
low = people[people["resilience_group"] == "Low resilience (< 3)"]["average_stress"]
normal = people[people["resilience_group"] == "Normal / high resilience"]["average_stress"]
test = stats.ttest_ind(low, normal, equal_var=False)
ci = test.confidence_interval()
plt.figure(figsize=(7, 5.5))
order = ["Normal / high resilience", "Low resilience (< 3)"]
sns.boxplot(data=people, x="resilience_group", y="average_stress", order=order, hue="resilience_group",
            legend=False, palette={order[0]: COLORS[0], order[1]: COLORS[1]}, showfliers=False, showmeans=True, width=0.5)
sns.stripplot(data=people, x="resilience_group", y="average_stress", order=order, color="black", alpha=0.4)
plt.title(f"H10: People with low resilience are more stressed\n"
          f"Independent t-test: t({test.df:.0f}) = {test.statistic:.2f}, p = {test.pvalue:.3g}")
plt.xlabel("")
plt.ylabel("Participant's average stress")
save_chart("hypotheses/H10.png")
save_result("H10", "People with low resilience are more stressed", "Independent t-test (Welch)", len(low) + len(normal),
            f"t({test.df:.1f}) = {test.statistic:.2f}", test.pvalue,
            f"mean difference = {low.mean() - normal.mean():.2f}", (ci.low, ci.high))


# H11 and H12: one-way ANOVA, average stress across three groups of people
def anova_test(hid, title, group_column, order, x_label):
    groups = [people.loc[people[group_column] == g, "average_stress"] for g in order]
    test = stats.f_oneway(*groups)
    df_between, df_within = len(groups) - 1, len(people) - len(groups)
    grand_mean = people["average_stress"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((people["average_stress"] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total

    plt.figure(figsize=(7, 5.5))
    sns.boxplot(data=people, x=group_column, y="average_stress", order=order, hue=group_column, legend=False,
                palette=dict(zip(order, COLORS)), showfliers=False, showmeans=True, width=0.5)
    sns.stripplot(data=people, x=group_column, y="average_stress", order=order, color="black", alpha=0.4)
    plt.title(f"{hid}: {title}\nOne-way ANOVA: F({df_between}, {df_within}) = {test.statistic:.2f}, p = {test.pvalue:.3g}")
    plt.xlabel(x_label)
    plt.ylabel("Participant's average stress")
    save_chart(f"hypotheses/{hid}.png")
    save_result(hid, title, "One-way ANOVA", len(people), f"F({df_between}, {df_within}) = {test.statistic:.2f}",
                test.pvalue, f"η² = {eta_squared:.2f}")


anova_test("H11", "Stress differs by satisfaction with mental health", "mental_health_group",
           ["Very satisfied", "Satisfied", "Dissatisfied"], "Satisfaction with own mental health")
anova_test("H12", "Stress rises with general negative feelings", "negative_feelings_group",
           ["Low (5-7)", "Medium (8-11)", "High (12-25)"], "General negative feelings (PANAS)")

# ---- Results table and summary chart ---------------------------------------------------------
table = pd.DataFrame(results)
table.round(4).to_csv(RESULTS / "hypothesis_results.csv", index=False, encoding="utf-8-sig")

table["evidence"] = -np.log10(table["p_value"])
table["label"] = table["id"] + ": " + table["hypothesis"]
plt.figure(figsize=(13, 7))
sns.barplot(data=table, x="evidence", y="label", hue="test", dodge=False)
plt.axvline(-np.log10(0.05), color="red", linestyle="--", label="p = 0.05")
plt.title("All 12 hypotheses together:\nhow strong is the evidence against H0?")
plt.xlabel("Strength of evidence = −log10(p)\n(bars past the red line have p < 0.05)")
plt.ylabel("")
plt.legend(loc="lower right")
save_chart("summary_all_hypotheses.png")

print()
print(table[["id", "hypothesis", "test", "n", "statistic", "p_value", "effect", "decision"]]
      .round(4).to_string(index=False))
print("\nDone. Charts and tables are in the 'results' folder.")
