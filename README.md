# Data Science Lighthouse Labs Final Project 
<p>Created by: Ken Wall</p>

### Data Science Final Project - Stock Price Prediction
<p>To determine if the TSX index price will go up/down the next day based on a wide range of financial and economic features.</p>

## Project Goals
<p>To apply supervised learning techniques to a real-world financial data set and use exploratory data analysis, dimensionality reduction and feature engineering to communicate the insights gained from the analysis. To gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked.</p>

## Process

### Part 1: Obtaining the Dataset
<p>Obtaining the financial and economic information was not challenging but tedious, because every feature had to be obtained in its own CSV file and then combined later on in python.</p>
<p>Information could be broken down into the following categories:</p>

- Commodities
- Currencies
- Economic
- Indexes

<p>The individual commodity, currency and index information could be found on Investing.com. The economic data was taken from Statistics Canada. The economic data was not in a usuable format so custom python functions had to be created to format it properly. Categories were first combined and then merged into a preprocessed dataset to be used in the baseline and the iterations of model selection.</p>

### Part 2: Exploratory Data Analysis (EDA)

**BASELINE** - No EDA was done for baseline. The preprocessed dataset was cross validated using TimeSeriesSplit and then sent for model selection. See model selection below for further details.

**DATE-TIME COMPONENTS** - No meaningful patterns or trends were identified when reviewing if the index prices were up/down in certain years, months, and days. The index increased slightly more on a daily basis than going down at an average of 52.4% of the time. The maximum month where the index went up was 82.6% of the time in 2019 and the minimum was 2017 September at 27%. During the Covid Pandemic the market actually performed better in 2019-2021 with the index increasing greater than 55% of the time. Monthly fluctions occured, but on average each year the results are 
similar across all 10 years.

**TIME-SERIES GRAPHING** - When reviewing the line graphs over the 10 year time horizon of all the features it can be seen that many of the features have the same pattern/trend. A sharp decrease during COVID and a steady rise after. It is likely that the correlation between these features is very high, which will be confirmed in the next test.

**ITERATION 1 - CORRELATION** - After running the correlation matrix and heatmap, there are 11 features that have a correlation of greater than 0.9 with each other. Highly correlated features can lead to model problems due to multicolinearity and redundant information. As part of the iterative process, iteration 1 will remove the features with correlation greater than 0.9 and see how the model performs.



### Part 3: Model Selection

**BASELINE** (ACCURACY)
- Logistic Regression: 0.59
- Random Forrest: 0.63
- Support Vector Machines: 0.58
- K-Nearest Neighbours: 0.58
- XGBoost: 0.6
- Neural Networks (MLP): 0.62

**ITERATION 1 - BASELINE + CORRELATION** (ACCURACY)
- Logistic Regression: 0.61
- Random Forrest: 0.59
- Support Vector Machines: 0.59
- K-Nearest Neighbours: 0.59
- XGBoost: 0.6
- Neural Networks (MLP): 0.6
<p>Removing the correlated features improves the models for logistic regression(assumes independent features, so improvement is expected), SVM and KNN, but reduces the two models that had the highest accuracy in the baseline case, MLP and Random Forrest.</p>

**ITERATION 2 - BASELINE + EXPONENTIAL MOVING AVERAGE (EMA)** (ACCURACY)
- Logistic Regression: 0.61
- Random Forrest: 0.59
- Support Vector Machines: 0.58
- K-Nearest Neighbours: 0.58
- XGBoost: 0.63
- Neural Networks (MLP): 0.6

**ITERATION 3 - BASELINE + RELATIVE STRENGTH INDEX (RSI)** (ACCURACY)
- Logistic Regression: 0.59
- Random Forrest: 0.62
- Support Vector Machines: 0.58
- K-Nearest Neighbours: 0.62
- XGBoost: 0.63
- Neural Networks (MLP): 0.6

**ITERATION 4 - BASELINE + VOLATILITY** (ACCURACY)
- Logistic Regression: 0.71
- Random Forrest: 0.71
- Support Vector Machines: 0.71
- K-Nearest Neighbours: 1.0
- XGBoost: 0.71
- Neural Networks (MLP): 0.71

**ITERATION 5 - BASELINE + CORRELATION + EMA + RSI** (ACCURACY)
- Logistic Regression: 0.67
- Random Forrest: 0.64
- Support Vector Machines: 0.58
- K-Nearest Neighbours: 0.58
- XGBoost: 0.64
- Neural Networks (MLP): 0.62




### Part 4: Tuning and Pipelining

## Results

## Challenges

## Future Goals