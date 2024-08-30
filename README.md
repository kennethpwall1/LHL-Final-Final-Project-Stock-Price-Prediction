# Data Science Lighthouse Labs Final Project 
<p>Created by: Ken Wall</p>

### Data Science Final Project - Stock Price Prediction
<p>To determine if the TSX index price will go up/down the next day based on a wide range of financial and economic features.</p>

## Project Goals
<p>To apply supervised learning techniques to a real-world financial data set and use exploratory data analysis, dimensionality reduction and feature engineering to communicate the insights gained from the analysis. To gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked.</p>

## Process

### <u>Part 1: Obtaining the Dataset</u>
<p>Obtaining the financial and economic information was not challenging but tedious, because every feature had to be obtained in its own CSV file and then combined later on in Python.</p>
<p>Information could be broken down into the following categories:</p>

- Commodities
- Currencies
- Economic
- Indexes

<p>The individual commodity, currency and index information could be found on Investing.com. The economic data was taken from Statistics Canada. The economic data was not in a usuable format so custom Python functions had to be created to format it properly. Categories were first combined and then merged into a preprocessed dataset to be used in the baseline and the iterations of model selection.</p>

### <u>Part 2: Exploratory Data Analysis (EDA)</u>

**BASELINE** - Time and careful consideration was given to the selection of the original features, so no EDA was done for baseline. The preprocessed dataset was cross validated using TimeSeriesSplit and then sent for model selection. See model selection below for further details.

**DATE-TIME COMPONENTS** - No meaningful patterns or trends were identified when reviewing if the index prices were up/down in certain years, months, and days. The index increased slightly more on a daily basis than going down at an average of 52.4% of the time. The maximum month where the index went up was 82.6% of the time in 2019 and the minimum was 2017 September at 27%. During the Covid Pandemic the market actually performed better in 2019-2021 with the index increasing greater than 55% of the time. Monthly fluctions occured, but on average each year the results are 
similar across all 10 years.

**TIME-SERIES GRAPHING** - When reviewing the line graphs over the 10 year time horizon of all the features it can be seen that many of the features have the same pattern/trend. A sharp decrease during COVID and a steady rise after. It is likely that the correlation between these features is very high, which will be confirmed in the next test.

**ITERATION 8 - CORRELATION** - After running the correlation matrix and heatmap, there are 11 features that have a correlation of greater than 0.9 with each other. Highly correlated features can lead to model problems due to multicolinearity and redundant information. As part of the iterative process, iteration 8 will remove the features with correlation greater than 0.9 and see how the model performs.

Other than logistical regression all other models performed worse once correlated features were removed. The best performing models are tree based models and correlated features do not inherently pose a problem for Decision Trees. Therefore I will leave the correlated features in for the first phase of model selection.



### <u>Part 3: Model Selection</u>

<p>I chose an iterative approach to model selection and ran the dataset through the following models during each iteration to determine which had the best results:</p>

- Logistic Regression
- Random Forest Classifier
- Support Vector Machines
- k-Nearest Neighbors (k-NN)
- XGBoost
- Neural Networks (MLP)

<p>The table below outlines the changes made during each iteration and the model that had the best accuracy.</p>

 <table>
  <tr>
    <th>Version</th>
    <th>Iteration Changes</th>
    <th>Best Model</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>Baseline</td>
    <td>Random Forest</td>
    <td>0.63</td>
  </tr>
   <tr>
    <td>Version 1</td>
    <td>Baseline + EMA</td>
    <td>XGBoost</td>
    <td>0.63</td>
  </tr>
   <tr>
    <td>Version 2</td>
    <td>Baseline + EMA + EMA Slope</td>
    <td>XGBoost</td>
    <td>0.69</td>
  </tr>
   <tr>
    <td>Version 3</td>
    <td>Baseline + EMA + EMA Slope + EMA/Close</td>
    <td>Random Forest</td>
    <td>0.70</td>
  </tr>
   <tr>
    <td>Version 4</td>
    <td>Baseline + EMA + EMA Slope + EMA/Close + EMA Divergence</td>
    <td>Random Forest</td>
    <td>0.68</td>
  </tr>
   <tr>
    <td>Version 5</td>
    <td>Baseline + EMA + EMA Slope + EMA Divergence</td>
    <td>XGBoost</td>
    <td>0.69</td>
  </tr>
   <tr>
    <td>Version 6</td>
    <td>Baseline + RSI</td>
    <td>XGBoost</td>
    <td>0.63</td>
  </tr>
  <tr>
    <td>Version 7</td>
    <td>Baseline + Volatility</td>
    <td>XGBoost</td>
    <td>0.61</td>
  </tr>
  <tr>
    <td>Version 8</td>
    <td>Baseline - Correlation</td>
    <td>Logistic Regression</td>
    <td>0.61</td>
  </tr>
  <tr>
    <td>Version 9</td>
    <td>Baseline + MACD</td>
    <td>XGBoost</td>
    <td>0.60</td>
  </tr>
  <tr>
    <td>Version 10</td>
    <td>Baseline + EMA + EMA Slope + EMA/Close + RSI</td>
    <td>XGBoost</td>
    <td>0.70</td>
  </tr>
  <tr>
    <td>Version 11</td>
    <td>Baseline + EMA + EMA Slope + EMA/Close + Volatility</td>
    <td>XGBoost</td>
    <td>0.66</td>
  </tr>
  <tr>
    <td>Version 12</td>
    <td>Baseline + EMA + EMA Slope + EMA/Close + MACD</td>
    <td>XGBoost</td>
    <td>0.72</td>
  </tr>
  <tr>
    <td>Version 13</td>
    <td>Baseline + All EMA Features + RSI + MACD + Volatility</td>
    <td>XGBoost</td>
    <td>0.74</td>
  </tr>
</table> 

<p>After several versions of trial and error for features engineered for time based features (exponential moving average EMA), price transformations (volatility) and technical indicators (relative strength index RSI and moving average convergence divergence MACD) I moved to model driven feature elimination using recursive feature elimination (RFE) and Random Forest Importance to see if I can improve the model above the current benchmark of 0.74 set in the Version 13 XGBoost Model. I have included all features initially to test this, baseline + all EMA features, RSI, MACD and volatility.</p>

#### Recursive Feature Elimination
XGBoost was the most successful model for the 13 iterations noted above, so I stuck with it for RFE. The RFE class require number of features and I selected a range from 20 to 35 features given that the total number of features are 40. The accuracy results are listed below. 35 features had the highest accuracy at 0.746.

 <table>
  <tr>
    <th>Number of Features</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <th>20</th>
    <th>0.738</th>
  </tr>
  <tr>
    <th>25</th>
    <th>0.741</th>
  </tr>
  <tr>
    <th>30</th>
    <th>0.730</th>
  </tr>
  <tr>
    <th>35</th>
    <th>0.746</th>
  </tr>
</table>

#### Feature Importance
<p>I also wanted to rank the importance of the features using the random forest classifier, which can be seen in the image below. There are 11 features (Trade balance to Year) that have minimal importance to the model. I dropped those features and then ran the data set through all of the models again and XGBoost had an accuracy of 0.75, which is the highest achieved so far, but similar to the XGBoost RFE result of 0.746 with 35 features noted above.</p>

![Feature Importance!](images/feature_importance.png "Feature Importance")

<p>An accuracy of 0.75 is now the current benchmark and hyper parameter tuning is the next step to see if I can improve the accuracy of the model further. See the next section for further discussion.</p>

### <u>Part 4: Tuning and Pipelining</u>

## Results

## Challenges

## Future Goals