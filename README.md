# Power Outage Duration Analysis

Final Project for DSC 80 @ UC San Diego

**Name(s)**: Srinivas Sriram

## Introduction

The dataset I will be analyzing in this project is a collection of major power outage records in the United States from January 2000 to July 2016. According to the dataset’s source paper, a major outage impacts at least 50,000 customers or causes an unplanned firm load loss of at least 300 MW ([dataset publication](https://www.sciencedirect.com/science/article/pii/S2352340918307182)).

This project centers around one central question:

How do various known factors at the onset of an outage (such as location, climate, and outage causes) influence outage duration, and can we build a predictive model to estimate outage duration before restoration?

Understanding this question has immense real-world impact. If communities could obtain a reliable estimate of outage duration using only information known at the start of the outage, they could allocate resources more effectively and improve overall disaster response.

The dataset contains 1534 rows, each representing a single major outage event. Below are the key columns relevant to this analysis:

| Column Name                  | Description                                                                                      |
|------------------------------|--------------------------------------------------------------------------------------------------|
| **YEAR**                     | Indicates the year when the outage event occurred                                                |
| **MONTH**                    | Indicates the month when the outage event occurred                                               |
| **U.S._STATE**               | Represents all the states in the continental U.S.                                                |
| **POSTAL.CODE**              | Represents the postal code of the U.S. states                                                    |
| **NERC.REGION**              | North American Electric Reliability Corporation (NERC) regions involved in the outage            |
| **CLIMATE.REGION**           | U.S. climate regions as defined by the National Centers for Environmental Information            |
| **ANOMALY.LEVEL**            | Oceanic Niño/La Niña index (ONI), a 3-month running mean of SST anomalies in the Niño 3.4 region |
| **CLIMATE.CATEGORY**         | Climate episode classification: “Warm”, “Cold”, or “Normal” based on ONI thresholds              |
| **OUTAGE.START.DATE**        | Day of the year when the outage event started1                                                   |
| **OUTAGE.START.TIME**        | Time of day when the outage event started1                                                       |
| **OUTAGE.RESTORATION.DATE**  | Day of the year when power was restored to all the customers                                     |
| **OUTAGE.RESTORATION.TIME**	 | Time of day when power was restored to all the customers                                         |
| **CAUSE.CATEGORY**           | Broader categories of events causing the major power outages                                     |
| **CAUSE.CATEGORY.DETAIL**    | Detailed description of the cause category                                                       |
| **HURRICANE.NAMES**          | Name of the hurricane if the outage was hurricane-related                                        |
| **CUSTOMERS.AFFECTED**       | Number of customers affected by the power outage event                                           | 
| **RES.PRICE**                | Monthly residential electricity price (cents/kWh)                                                |
| **COM.PRICE**                | Monthly commercial electricity price (cents/kWh)                                                 |
| **IND.PRICE**                | Monthly industrial electricity price (cents/kWh)                                                 |
| **TOTAL.PRICE**              | Average monthly electricity price for the state (cents/kWh)                                      |
| **RES.SALES**                | Electricity consumption in the residential sector (MWh)                                          |
| **COM.SALES**                | Electricity consumption in the commercial sector (MWh)                                           |
| **IND.SALES**                | Electricity consumption in the industrial sector (MWh)                                           |
| **TOTAL.SALES**              | Total electricity consumption in the state (MWh)                                                 |
| **RES.PERCEN**               | Percent of total electricity consumption from the residential sector                             |
| **COM.PERCEN**               | Percent of total consumption from the commercial sector                                          |
| **IND.PERCEN**               | Percent of total consumption from the industrial sector                                          |
| **RES.CUSTOMERS**            | Annual number of residential electricity customers                                               |
| **COM.CUSTOMERS**            | Annual number of commercial electricity customers                                                |
| **IND.CUSTOMERS**            | Annual number of industrial electricity customers                                                |
| **TOTAL.CUSTOMERS**          | Total annual number of customers served in the state                                             |
| **RES.CUST.PCT**             | Percent of residential customers                                                                 |
| **COM.CUST.PCT**             | Percent of commercial customers                                                                  |
| **IND.CUST.PCT**             | Percent of industrial customers                                                                  |
| **PC.REALGSP.STATE**         | Per-capita real gross state product (2009 dollars)                                               |
| **PC.REALGSP.USA**           | Per-capita U.S. real gross domestic product (2009 dollars)                                       |
| **PC.REALGSP.REL**           | Relative per-capita real GSP compared to the U.S.                                                |
| **PC.REALGSP.CHANGE**        | Percent change in per-capita real GSP from previous year                                         |
| **UTIL.REALGSP**             | Real GSP contributed by the utility industry                                                     |
| **TOTAL.REALGSP**            | Real GSP contributed by all industries                                                           |
| **UTIL.CONTRI**              | Utility sector’s percent contribution to total real GDP                                          |
| **PI.UTIL.OFUSA**            | Utility sector’s income as a percent of total U.S. utility-sector income                         |
| **POPULATION**               | Population of the U.S. state in that year                                                        |
| **POPPCT_URBAN**             | Percent of population living in urban areas                                                      |
| **POPPCT_UC**                | Percent of population living in urban clusters                                                   |
| **POPDEN_URBAN**             | Population density of urban areas (persons per sq. mile)                                         |
| **POPDEN_UC**                | Population density of urban clusters                                                             |
| **POPDEN_RURAL**             | Population density of rural areas                                                                |
| **AREAPCT_URBAN**            | Percent of land area that is urban                                                               |
| **AREAPCT_UC**               | Percent of land area that is urban clusters                                                      |
| **PCT_LAND**                 | State land area as a percent of total U.S. land area                                             |
| **PCT_WATER_TOT**            | State total water area as a percent of U.S. water area                                           |
| **PCT_WATER_INLAND**         | State inland water area as a percent of U.S. inland water area                                   |

## Data Cleaning and Exploratory Data Analysis

To prepare the outage dataset for analysis, I applied the following cleaning steps:

1. The original data was presented as an Excel file. I used Google Sheets to remove the first 3 filler rows as well as correctly format the table by removing the variables and units headers.
2. I examined the null distribution across columns, to see how which columns have missing values. I also examined for any non-trivial missingness (a large proportion of missing rows in a column).
3. Checked for null-like placeholder values (e.g., "None" or empty strings). No placeholders were found.
4. Created new time-based columns: Using OUTAGE.START.DATE, OUTAGE.START.TIME, OUTAGE.RESTORATION.DATE, and OUTAGE.RESTORATION.TIME, I constructed datetime columns OUTAGE.START and OUTAGE.RESTORATION (all in hours).
5. Using OUTAGE.START and OUTAGE.RESTORATION, I recalculated the OUTAGE.DURATION column (in hours). The reason behind this recalculation was to ensure consistency between the previous timestamp columns and the duration column. This column is now our target for future analysis. 

Below is the head of the cleaned DataFrame (subset of the columns selected):

| OBS | YEAR | MONTH | U.S._STATE | NERC.REGION | CLIMATE.REGION     | CLIMATE.CATEGORY | CAUSE.CATEGORY     | OUTAGE.START        | OUTAGE.RESTORATION  | OUTAGE.DURATION |
|-----|------|-------|------------|-------------|--------------------|------------------|--------------------|---------------------|---------------------|-----------------|
| 1   | 2011 | 7.0   | Minnesota  | MRO         | East North Central | normal           | severe weather     | 2011-07-01 17:00:00 | 2011-07-03 20:00:00 | 51.000000       |
| 2   | 2014 | 5.0   | Minnesota  | MRO         | East North Central | normal           | intentional attack | 2014-05-11 18:38:00 | 2014-05-11 18:39:00 | 0.016667        |
| 3   | 2010 | 10.0  | Minnesota  | MRO         | East North Central | cold             | severe weather     | 2010-10-26 20:00:00 | 2010-10-28 22:00:00 | 50.000000       |
| 4   | 2012 | 6.0   | Minnesota  | MRO         | East North Central | normal           | severe weather     | 2012-06-19 04:30:00 | 2012-06-20 23:00:00 | 42.500000       |
| 5   | 2015 | 7.0   | Minnesota  | MRO         | East North Central | warm             | severe weather     | 2015-07-18 02:00:00 | 2015-07-19 07:00:00 | 29.000000       |

### Univariate Analysis

For my univariate plot, I chose to analyze the distribution of outage duration to examine skewness, modality, and the spread of the duration column, since this is the column I will be using for future analysis.

<iframe

  src="assets/duration_distribution.html"

  width="800"

  height="600"

  frameborder="0"

></iframe>

This plot shows the distribution of OUTAGE.DURATION is extremely right-skewed, and longer durations are less likely based on our dataset.

### Bivariate Analysis

For my bivariate plot, I examined a linear relationship between the number of CUSTOMERS.AFFECTED and OUTAGE.DURATION, as longer durations should affect more customers.

<iframe
  src="assets/customers_affectedvsoutage_duration.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows a very weak positive association between the two variables, indicating that the number of customers affected isn't a good linear predictor of outage duration.

I also plotted the mean OUTAGE.DURATION by US._STATE using folium, to see which states have longer average durations. 

<iframe

  src="assets/map.html"

  width="800"

  height="600"

  frameborder="0"

></iframe>

### Interesting Aggregates

One aggregate that I performed was a pivot table of the mean OUTAGE.DURATION per U.S._STATE by CAUSE.CATEGORY. This table allows for a deeper analysis for how different causes affect outage durations in different states, and also reflects each state's capabilities to handle with outages caused by these different categories. 

Here is the head of this aggregate DataFrame:

| U.S. State | Equipment Failure | Fuel Supply Emergency | Intentional Attack | Islanding | Public Appeal | Severe Weather | System Operability Disruption |
|------------|-------------------|-----------------------|--------------------|-----------|---------------|----------------|-------------------------------|
| Alabama    | NaN               | NaN                   | 1.283333           | NaN       | NaN           | 23.695833      | NaN                           |
| Arizona    | 2.308333          | NaN                   | 10.660000          | NaN       | NaN           | 428.775000     | 6.408333                      |
| Arkansas   | 1.750000          | NaN                   | 9.130556           | 0.050000  | 17.728571     | 45.030000      | NaN                           |
| California | 8.746825          | 102.676667            | 15.774306          | 3.580952  | 33.801852     | 48.806219      | 6.061111                      |
| Colorado   | NaN               | NaN                   | 1.950000           | 0.033333  | NaN           | 45.454167      | 4.662500                      |

## Assessment of Missingness

### NMAR Analysis

One column that is NMAR in this dataset is HURRICANE.NAMES. This is because if the outage is not caused by a hurricane, the column will be empty, meaning that the value's missingness is dependent on the value itself (whether or not this is a hurricane-related entry). 

We could make this missingness MAR by having a column that simply indicates whether or not this is a hurricane-related outage, thus making the missingness of the name column dependent on the value of this indicator column. 

### Missingness Dependency

The column CLIMATE.CATEGORY represents the "climate episode" of the year the entry corresponds. This was the first identified column in our null distribution analysis from earlier that appeared to have a small amount of missing values (9 to be exact). Let's see what columns are associated with the missingness of this column.

#### U.S_STATE

Let's see if this column's missingness is MAR with the U.S_STATE in its entry by running a permutation test (significance level of 0.05).

**H0**: The distribution of U.S_STATE when CLIMATE.CATEGORY is missing is equal to the distribution of U.S_STATE when CLIMATE.CATEGORY is not missing.

**H1**: The two distributions are not equal to each other.

From this test, I found an observed TVD of 0.8032786885245903, and a p-value of 0.04346. Since our p-value of 0.04346 is less than a significance level of 0.05, we **reject** H0. It's likely that the distribution of state when the climate category is missing is different than the distribution of state when the climate category is not missing.

Below is the simulated null distribution along with our observed test statistic:

<iframe

  src="assets/us_state_climate_category_missing.html"

  width="800"

  height="600"

  frameborder="0"

></iframe>

### CUSTOMERS.AFFECTED

Now, let's see if climate category's missingness is dependent on the customers affected column (CUSTOMERS.AFFECTED) (significance level of 0.05).

**H0**: Mean CUSTOMERS.AFFECTED when CLIMATE.CATEGORY is missing is equal to the mean CUSTOMERS.AFFECTED when CLIMATE.CATEGORY is not missing.

**H1**: Mean CUSTOMERS.AFFECTED when CLIMATE.CATEGORY is missing is not equal to the mean CUSTOMERS.AFFECTED when CLIMATE.CATEGORY is not missing.

From this test, I found an observed mean difference of 60232.25224037956, and a p-value of 0.50028. Since our p-value of 0.50028 is greater than a significance level of 0.05, we **fail to reject** H0. It's likely that the mean customers affected when the climate category is missing is equal to the mean customers when the climate category is not missing. 

Below is the simulated null distribution along with our observed test statistic:

<iframe

  src="assets/customers_affected_missing_climate_category.html"

  width="800"

  height="600"

  frameborder="0"

></iframe>

## Hypothesis Testing

Let's analyze our target column, OUTAGE.DURATION, and hypothesize about its severity for different groups. 

Specifically, let's analyze the average outage duration for outages caused by severe weather compared to the average outage duration for outages not caused by severe weather.

**H0**: The average outage duration for outages caused by severe weather is the same as the average outage duration for outages not caused by severe weather (Dbar for severe weather - Dbar for non-severe weather = 0)

**H1**: The average outage duration for outages caused by severe weather is greater than the average outage duration for outages not caused by severe weather (Dbar for severe weather - Dbar for non-severe weather > 0)

We will use the difference in means as our test statistic, with a significance level of 0.05. 

From this test, I got an observed difference of 42.27376718667372, and a p-value of 0.0. 

Since this p-value of 0.0 is below a standard significance level of 0.05, we **reject** H0. It's likely that the average outage duration for outages caused by severe weather is greater than the average outage duration for outages not caused by severe weather.

These results make sense, as for outages caused by severe weather it's expected that it would take longer for the power to return than for other smaller-scale reasons.

Below is the simulated null distribution along with our test statistic. 

<iframe

  src="assets/hyp_test.html"

  width="800"

  height="600"

  frameborder="0"

></iframe>

## Framing a Prediction Problem

**Prediction Problem:** Can we predict the duration of a outage using only meaningful information known at the onset of the outage?

This would be a regression problem, as OUTAGE.DURATION is a quantiative and continuous variable. I chose this variable because this would be the most useful variable to predict for real-world situations, if people could have a quality estimate of how long their power outages would last based on current situational data, they could prepare and establish the necessary safety measures in their communities. The metric I'll be mainly using to evaluate my model is RMSE, because regression problems require continuous measures of error and RMSE penalizes larger errors more heavily and offers a more math-friendly approach to optimizing parameters. I'll also compare this performance with MAE, to ensure that the model consistently performs across different metrics.

Only data known at the onset of the outage, such as regional data and outage causes, will be used to make predictions. Data determined after the outage will not be used to train the model. 

## Baseline Model

My baseline model is a linear regression model that used U.S._STATE (nominal, one-hot encoded), CAUSE.CATEGORY (nominal, one-hot encoded), and TOTAL.CUSTOMERS (quantitative) as features to predict OUTAGE.DURATION. 

I used U.S._STATE because each state has their own unique climate, infrastructure, and governance that may dictate how quickly they are able to respond and fix outages. CAUSE.CATEGORY is a useful feature as well since the cause of the outage, as shown earlier with the severe weather test, can drastically change the expected time to restore power. Finally, TOTAL.CUSTOMERS is a good measure of how populated a certain area is, and there might be hidden patterns how areas with larger customer bases respond to power outages.

Below is the baseline model's initial performance:

```angular2html
Baseline R^2 train = 0.2241636662121359
Baseline R^2 test = 0.1494085239683144
Baseline RMSE on train = 78.70499777006609
Baseline RMSE on test = 120.44510635777436
```
The model achieves a train R² of 0.224 and test R² of 0.149, indicating that it explains only a small portion of the variance in outage duration. RMSE values of 78.705 hours (train) and 120.445 hours (test) show that predictions are typically off by several days. These low scores are expected for this simple baseline model.

## Final Model

For my final model, I trained a RandomForestRegressor on a wide subset of onset-known features, listed below:

```python
# feature list for final model
categorical_features = [
    'YEAR', 'MONTH', 'U.S._STATE', 'POSTAL.CODE', 'NERC.REGION',
    'CLIMATE.REGION', 'ANOMALY.LEVEL', 'CLIMATE.CATEGORY',
    'OUTAGE.START.DATE', 'OUTAGE.START.TIME',
    'CAUSE.CATEGORY', 'CAUSE.CATEGORY.DETAIL'
]

numeric_features = [
    'RES.PRICE', 'COM.PRICE', 'IND.PRICE', 'TOTAL.PRICE',
    'RES.SALES', 'COM.SALES', 'IND.SALES', 'TOTAL.SALES',
    'RES.PERCEN', 'COM.PERCEN', 'IND.PERCEN',
    'RES.CUSTOMERS', 'COM.CUSTOMERS', 'IND.CUSTOMERS', 'TOTAL.CUSTOMERS',
    'RES.CUST.PCT', 'COM.CUST.PCT', 'IND.CUST.PCT',
    'PC.REALGSP.STATE', 'PC.REALGSP.USA', 'PC.REALGSP.REL',
    'PC.REALGSP.CHANGE', 'UTIL.REALGSP', 'TOTAL.REALGSP',
    'UTIL.CONTRI', 'PI.UTIL.OFUSA',
    'POPULATION', 'POPPCT_URBAN', 'POPPCT_UC',
    'POPDEN_URBAN', 'POPDEN_UC', 'POPDEN_RURAL',
    'AREAPCT_URBAN', 'AREAPCT_UC', 'PCT_LAND',
    'PCT_WATER_TOT', 'PCT_WATER_INLAND'
]

all_features = categorical_features + numeric_features
```
The reason why I chose these features is that for predicting outage duration based only on factors known at the start of the outage, a wide variety of features should be used to capture the hidden and complex patterns in the dataset. Thus, all the features were provided to this final model, with the hopes of allowing the RandomForestRegressor to decipher out the patterns for itself.

Since some of these input features contained missing values, we will need to set up a system to impute values. We will use sklearn's SimpleImputer package to impute values into our numeric and categorical columns respectively.

We can identify that the input features have trivial missingness (5-12 missing values). This is well under 1% of the actual numbers of rows of data, and at first glance there does not appear to be any structural missingness patterns in these columns. We can include CAUSE.CATEGORY.DETAIL because that's just an optional column that specifies any additional details for the cause of the outage (can encode this column as None through one-hot-encoding). We will impute all numeric columns with the median (robust to skewness) and categorical columns with the mode (preserves the most common category, suitable for small amounts of missingness) for our final model because these are safe imputation values that won't skew the existing distribution of values for the distribution of these variables (regardless of the shape of their actual distributions). 

Furthermore, for numeric features, I'll be adding a *QuantileTransformer* feature which will make the distribution of the feature normal and eliminate heavy skew, and a *PolynomialFeatures* feature with degree 2, which will allow interaction and quadratic features to be created among numeric variables, allowing the model to learn nonlinear relationships without manual feature construction. 

To train our RandomForestRegressor, we need to decide what hyperparameters we would like to tune and the specific values to tune them to. Below is the parameter grid for our hyperparameters.

```python
# parameter grid for the hyperparameters
param_grid = {
    "model__n_estimators": [200, 250, 300, 350, 400],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
```
**n_estimators - number of trees:** Ranging from 200-400, this gives us a good choice of the number of trees to vote from without having too many trees and taking up unnecessary runtime. 

**max_depth - the max depth each tree can go:** This controls how far each tree can go, each having the risk of underfitting or overfitting. A depth of 10 is quite shallow but can reduce overfitting. A depth of 20 is medium-depth and could capture the complex patterns that we need. None allows for fully grown trees which could capture the depth of the complexity, but at the cost of overfitting. 

**min_samples_split - minimum samples required before splitting:** Choosing 2 versus 5 allows the model to either to be highly flexible (could overfit) or be more conservative when performing splits. 

**min_samples_leaf - minimum samples in a leaf node:** Choosing between 1 or 2 allows for very granular leaves or for smoother final predictions - just a tune-up for the final prediction rule. 

Running a GridSearchCV on this parameter grid will give us 5 * 3 * 2 * 2 = 60 parameter combinations. Running this with 3-fold cross validation, GridSearchCV will train a total of 180 models. 

The optimal hyperparameters selected by GridSearch CV are listed below:

```python
{'model__max_depth': 10, 'model__min_samples_leaf': 2, 'model__min_samples_split': 5, 'model__n_estimators': 200}
```
Finally, here is the performance of our final model on the same train-test splits:

```angular2html
R^2 train = 0.6443186555622549
R^2 test = 0.3301826691266052
RMSE on train = 53.2902548487401
RMSE on test = 106.88255734728345
```
Here, we see a significant improvement in both train and test performance from the baseline (RMSE test went from ~120 to ~107 hours), indicating that our Random Forest model is a better fit for the complexity of our dataset. However, the performance is still not exactly optimal, indicating that perhaps a more complex model is needed to understand the underlying patterns of the dataset.

## Fairness Analysis

For our fairness analysis, we will examine if our model fairly predicts both outages caused by severe weather and outages caused by non severe weather. The reasoning behind this fairness analysis is to ensure that our model does not get biased by categories with higher mean durations (we saw earlier that mean outage durations for severe weather events are statistically higher).

We will use RMSE as our accuracy metric, and thus our test statistic will be the difference in RMSE of the two groups:

**D = RMSE(severe weather) - RMSE(non-severe weather)**

Here are our statistical hypotheses:

**H0:** The model is fair. The difference in RMSE between severe and non-severe outages is due to chance.

E(D) **=** 0

**H1:** The model is unfair. The RMSE differs statistically between groups.

E(D) **!=** 0

We will perform this permutation test on our test dataset (what the model hasn't seen yet).

From this test, I got an observed difference of -51.815569553818676 and a p-value of 1.0. Since this p-value of 1.0 is well above a standard significance level of 0.05, we fail to reject H0. It's likely that our model is fair and does not differ systematically for predictions between the groups of severe and non-severe weather outages.




