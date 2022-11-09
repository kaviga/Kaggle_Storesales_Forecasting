#For research and learning purposes
#Credits to:

#https: // www.kaggle.com / code / ekrembayar / store - sales - ts - forecasting - a - comprehensive - guide / notebook
#https: // www.kaggle.com / code / howoojang / first - kaggle - notebook - following - ts - tutorial
#https: // www.kaggle.com / competitions / store - sales - time - series - forecasting / data?select = train.csv
#https: // www.kaggle.com / code / kashishrastogi / store - sales - analysis - time - serie / notebook


# 0.IMPORTS AND PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import os
import gc

import statsmodels.api as sm


path = '/Users/*********/PycharmProjects/store_sale_time_series_forecasting/'
os.listdir(path)
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
transactions = pd.read_csv(path + 'transactions.csv')
oil = pd.read_csv(path + 'oil.csv', index_col=False)
holidays = pd.read_csv(path + 'holidays_events.csv')
stores = pd.read_csv(path + 'stores.csv')
sample = pd.read_csv(path + "sample_submission.csv")

print(f'Train Shape: {train.shape}')
print("-" * 75)
print(train.head())
print("-" * 75)
print(train.info())
print("-" * 75)
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)



# 1. EXPLORATORY DATA ANALYSIS(EDA)

# 1.1 TRANSACTIONS

transactions.head()
temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
temp.head(10)


# Spearman's rank correlation coefficient (p):In statistics, Spearman's p, means statistical dependence between the rankings of two variables. It assesses how well the relationship between two variables can be described using a monotonic function. Similar to pearson's p but with ranks. Pearson's is more sensitive to outliers. That is because Spearman's p limits the outlier to the value of its rank.

print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(
    temp.corr("spearman").sales.loc["transactions"]))
px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr',
        title="Yearly transactions per store")

# First pattern discovered --> December increases sales
# Let's check yearly & monthly average transactions

avg_transactions = transactions.groupby('date').agg({'transactions': 'mean'}).reset_index()
print(avg_transactions)

a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year
px.line(a, x='date', y='transactions', color='year', title="Monthly Average Transactions")

c = temp.copy()
c['year'] = c.date.dt.year
c['dayofweek'] = c.date.dt.dayofweek + 1
c = c.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(c, x='dayofweek', y='transactions', color='year', title='Yearly Transactions per day of Week')

# --> More sales in days 6 and 7 of week.
# 1.2 Sales

train.head(10)
px.line(train.sort_values(["store_nbr", "date"]), x='date', y='sales', color='store_nbr',
        title="Yearly sales per store")

b = train.copy()
b['year'] = b.date.dt.year
b['dayofweek'] = b.date.dt.dayofweek

b = b.set_index("date").resample("M").sales.mean().reset_index()
b["year"] = b.date.dt.year
px.line(b, x='date', y='sales', title="Monthly Average Sales")

px.scatter(temp, x="transactions", y="sales", trendline="ols", trendline_color_override="red")

# --> High correlation between transactions and sales

train['date'] = pd.to_datetime(train['date'])
train['day_of_week'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
data_grouped_day = train.groupby(['day_of_week']).mean()['sales']
data_grouped_month = train.groupby(['month']).mean()['sales']
data_grouped_year = train.groupby(['year']).mean()['sales']

plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(131)
plt.title('avg sales per day')
data_grouped_day.plot(kind='bar', stacked=True)
plt.subplot(132)
plt.title('avg sales per month')
data_grouped_month.plot(kind='bar', stacked=True)
plt.subplot(133)
plt.title('avg sales per year')
data_grouped_year.plot(kind='bar', stacked=True)

# --> Sat & sunday
# show the highest values
# --> december show the highest values
# --> growin yearly

# 1.3 Oil Prices

oil.head()
px.line(oil, x='date', y='dcoilwtico')

# As we can see, we are missing data points.
# There are various imputation methods we can apply.
# However, the most simple solution for that is Linear Interpolation.
# The linear interpolant is the straight line between points.

oil['date'] = pd.to_datetime(oil.date)

# Interpolate:
oil['dcoilwtico'] = np.where(oil['dcoilwtico'] == 0, np.nan, oil['dcoilwtico'])
oil['dcoilwtico_interpolated'] = oil.dcoilwtico.interpolate()
px.line(oil, x='date', y='dcoilwtico_interpolated', title='Daily Oil Price Interpolated')

# Theoricaly, if daily oil price is high, price of product increases and sales decreases.
# We then expect a negative relationship here between the interpolation of daily oil price and sales.
# Let's check both visually and with Spearman's p value:


temp = pd.merge(temp, oil, how="left")
print("Correlation with Daily Oil Prices")
print(temp.drop(["store_nbr", "dcoilwtico"], axis=1).corr("spearman").dcoilwtico_interpolated.loc[
          ["sales", "transactions"]], "\n")

# drop nan values to calculate trendlines:
temp.dropna(inplace=True)

# calculate trendlines for both sales and transactions vs oil prices:
# Numpy's polyfit function helps us draw those trendline in matplotlib scatter plots
t1 = np.polyfit(temp.dcoilwtico_interpolated, temp.sales, 1)
t2 = np.polyfit(temp.dcoilwtico_interpolated, temp.transactions, 1)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
temp.plot.scatter(ax=axes[0], x="dcoilwtico_interpolated", y='transactions', c="orange")
temp.plot.scatter(ax=axes[1], x="dcoilwtico_interpolated", y='sales', c="blue")

# Even though Spearman's p value means there is no particular correlation between oil price and grocery consumption, as we can see in the graph above there is a clear pattern.
# --> Pattern between daily oil prices & Transactions, Sales.
# We can clearly see two clusters here: above and below 70.
# When daily oil price is under 70, there are more sales in the data.

# 1.4 Family
train.family.unique()

train['family'] = train['family'].astype('category')
train['family_category'] = train['family'].cat.codes

family_category = dict(zip(train['family'].cat.codes, train['family']))
family_category

data_grouped_family_types = train.groupby(['family']).mean()[['sales', 'onpromotion']]

data_grouped_family_types['%_s'] = 100 * data_grouped_family_types['sales'] / data_grouped_family_types['sales'].sum()
data_grouped_family_types['%_s'] = data_grouped_family_types['%_s'].round(decimals=3)

percent = 100 * data_grouped_family_types['sales'] / data_grouped_family_types['sales'].sum()
percent = percent.round(decimals=3)
patches, texts = plt.pie(data_grouped_family_types['%_s'], startangle=90, radius=1.5)

lables_2 = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(family_category.values(), percent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy = zip(*sorted(zip(patches, lables_2, data_grouped_family_types['%_s']),
                                         key=lambda x: x[2],
                                         reverse=True))

plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)

# Top products:
# 1. Grocery
# 2. Beverages
# 3. Produce
# 4.Cleaning
# 5. Dairy.

# --> Grocery + Beverages > 50 %


# 1.5 Holidays

holidays.head()
# Merge Holidays(date + type of holiday day) & copied train table(average sales) --> "b" table


holiday_table = holidays[['date', 'type']]
# b.head()

holiday_table['date'] = pd.to_datetime(holiday_table['date'])
df = pd.merge_asof(holiday_table, b, on='date')

# df.head(-100)

# Drop NaN Values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()

sales_per_type = df.groupby(['type']).mean()['sales']
sales_per_type.head()

a_h_s = sales_per_type.mean()

sales_per_type.plot(kind='bar', figsize=(15, 9)).set_title('Sales per holiday type')
print(f'Average holiday Sales is:  {a_h_s}')

# Zero Forecasting

# Some stores don 't sell certain products --> forecast next 15 days will be 0.
# Create new dataframe for those products. Combine at submission part.

zero_table = train.groupby(['store_nbr', 'family']).sales.sum().reset_index().sort_values(['family', 'store_nbr'])
zero_table = zero_table[zero_table.sales == 0]
zero_table

print(train.shape)
# Anti Join
outer_join = train.merge(zero_table[zero_table.sales == 0].drop("sales", axis=1), how='outer', indicator=True)
train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis=1)
del outer_join
gc.collect()
train.shape

zero_prediction = []
for i in range(0, len(zero_table)):
    zero_prediction.append(
        pd.DataFrame({
            "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
            "store_nbr": zero_table.store_nbr.iloc[i],
            "family": zero_table.family.iloc[i],
            "sales": 0
        })
    )
zero_prediction = pd.concat(zero_prediction)
del zero_table
gc.collect()


# LINEAR REGRESSION


avg_sales = train.groupby('date').agg({'sales': 'mean'}).reset_index()
avg_sales['Time'] = np.arange(len(avg_sales.index))
avg_sales.head()

import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(12, 6),
    titlesize=18,
    titleweight='bold',
)

plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

# Use it for the Lag_1 plot later.
plot_params = dict(
    color='0.75',
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

fig, ax = plt.subplots()
ax.plot('Time', 'sales', data=avg_sales, color='0.75')
ax = sns.regplot(x='Time', y='sales', data=avg_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of sales');


avg_sales['Lag_1'] = avg_sales['sales'].shift(1)
avg_sales = avg_sales.reindex(columns=['date', 'sales', 'Lag_1', 'Time'])

fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='sales', data=avg_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of sales')

from sklearn.linear_model import LinearRegression

# Training data
X = avg_sales.loc[:, ['Time']]  # features
y = avg_sales.loc[:, 'sales']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)
y_pred

ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of sales')

from sklearn.linear_model import LinearRegression

X = avg_sales.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = avg_sales.loc[:, 'sales']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_pred

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('sales')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of sales');


ax = y.plot(**plot_params)
ax = y_pred.plot()

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(), )
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# Load the sales dataset
avg_sales = train.groupby('date').agg({'sales': 'mean'}).reset_index()
avg_sales = avg_sales.set_index('date').to_period("D")
avg_sales.head()

# %%
X = avg_sales.copy()

# days within a week
X['day'] = X.index.dayofweek  # the x-axis (freq)
X['week'] = X.index.week  # the seasonal period (period)

# days within a year
X['dayofyear'] = X.index.dayofyear
X['year'] = X.index.year

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
seasonal_plot(X, y="sales", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="sales", period="year", freq="dayofyear", ax=ax1);


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=avg_sales.index,
    constant=True,  # dummy feature for bias (y-intercept)
    order=1,  # trend ( order 1 means linear)
    seasonal=True,  # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality
    drop=True,  # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index
# X.head()



y = avg_sales["sales"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=180)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="sales - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()


from pathlib import Path
from warnings import simplefilter

comp_dir = Path('/Users/gorkagamo/PycharmProjects/store_sale_time_series_forecasting/')

holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

# National and regional holidays in the training set
holidays = (
    holidays_events
        .query("locale in ['National', 'Regional']")
        .loc['2017':'2017-08-15', ['description']]
        .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

X_holidays = pd.DataFrame(
    ohe.fit_transform(holidays),
    index=holidays.index,
    columns=holidays.description.unique(),
)

# Pandas solution
X_holidays = pd.get_dummies(holidays)

# Join to training data
X2 = X.join(X_holidays, on='date').fillna(0.0)

model = LinearRegression().fit(X2, y)
y_pred = pd.Series(
    model.predict(X2),
    index=X2.index,
    name='Fitted',
)

y_pred = pd.Series(model.predict(X2), index=X2.index)
ax = y.plot(**plot_params, alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax.legend();

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]

# Create training data
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()
X['NewYear'] = (X.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

df_test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create features for test set
X_test = dp.out_of_sample(steps=16)
X_test.index.name = 'date'
X_test['NewYear'] = (X_test.index.dayofyear == 1)

y_submit = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])
y_submit.to_csv('submission.csv', index=False)
