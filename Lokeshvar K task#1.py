# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option('display.max_columns', None)

# 1. LOAD DATASET

df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Unemployment in India.csv")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print("Dataset Shape:", df.shape)
print(df.head())

# 2. DATA CLEANING

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Rename columns
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Create additional date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()

print("\nCleaned Dataset:")
print(df.head())

# 3. EXPLORATORY DATA ANALYSIS

print("\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)

print(df.describe())

# Average unemployment by region
region_avg = (
    df.groupby('Region')['Unemployment_Rate']
      .mean()
      .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by Region")
print(region_avg)

# 4. UNEMPLOYMENT TREND ANALYSIS

monthly_trend = (
    df.groupby('Date')['Unemployment_Rate']
      .mean()
)

plt.figure(figsize=(14,6))
plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker='o'
)

plt.title("India Unemployment Trend")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 5. COVID-19 IMPACT ANALYSIS

pre_covid = df[df['Date'] < '2020-03-01']
covid_period = df[df['Date'] >= '2020-03-01']

print("\nAverage Unemployment Before COVID:",
      round(pre_covid['Unemployment_Rate'].mean(), 2))

print("Average Unemployment During COVID:",
      round(covid_period['Unemployment_Rate'].mean(), 2))

covid_comparison = pd.DataFrame({
    'Period': ['Pre-COVID', 'COVID Period'],
    'Rate': [
        pre_covid['Unemployment_Rate'].mean(),
        covid_period['Unemployment_Rate'].mean()
    ]
})

plt.figure(figsize=(8,5))
sns.barplot(
    data=covid_comparison,
    x='Period',
    y='Rate'
)

plt.title("Impact of COVID-19 on Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
plt.show()

# 6. STATE-WISE ANALYSIS

top_states = (
    df.groupby('Region')['Unemployment_Rate']
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States with Highest Unemployment")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.show()

# 7. SEASONAL TREND ANALYSIS

monthly_avg = (
    df.groupby('Month_Name')['Unemployment_Rate']
      .mean()
)

month_order = [
    'January','February','March','April',
    'May','June','July','August',
    'September','October','November','December'
]

monthly_avg = monthly_avg.reindex(month_order)

plt.figure(figsize=(12,5))
sns.lineplot(
    x=monthly_avg.index,
    y=monthly_avg.values,
    marker='o'
)

plt.title("Seasonal Pattern of Unemployment")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 8. CORRELATION ANALYSIS

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='Labour_Participation_Rate',
    y='Unemployment_Rate'
)

plt.title("Labour Participation vs Unemployment")
plt.show()

corr = df[
    ['Unemployment_Rate',
     'Labour_Participation_Rate',
     'Employed']
].corr()

plt.figure(figsize=(8,5))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")
plt.show()

# 9. KEY INSIGHTS

print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)

print("""
1. COVID-19 significantly increased unemployment rates.

2. Several states consistently experienced
   higher unemployment than the national average.

3. Seasonal fluctuations indicate changing
   labour market conditions throughout the year.

4. Labour participation and unemployment
   exhibit varying relationships across regions.

5. Lockdowns and economic disruptions
   heavily affected employment opportunities.
""")

# 10. POLICY RECOMMENDATIONS

print("\n" + "="*50)
print("POLICY RECOMMENDATIONS")
print("="*50)

print("""
• Expand employment guarantee schemes.

• Strengthen skill development programs.

• Support MSMEs with financial incentives.

• Develop region-specific employment policies.

• Improve labour market monitoring systems
  for future economic crises.
""")