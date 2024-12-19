import yfinance as yf

sp500 = yf.Ticker("^GSPC") #In YaHoo Finance, ticker symbol for SPY is ^GSPC.
sp500 = sp500.history(period="max") #Get all price data.
del sp500["Dividends"] #Remove as not necessary for an ETF.
del sp500["Stock Splits"] #Remove as not necessary for an ETF.
sp500["Tomorrow"] = sp500["Close"].shift(-1) #Create a new column called "Tomorrow" for the next day's closing price.
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int) #Check if tomorrow's price is greater than today's price. 1 for higher. 0 for lower.
sp500 = sp500.loc["1990-01-01":].copy() #Get data after 1990 for more relevant data.


from sklearn.ensemble import RandomForestClassifier # Use Random Forest to train the model as it will reduce likelihood of overfitting and pick up non-linear relationships.

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1) # Initialise model.


#Split data into train and test set. Since we are using timeseries data, we use the latest 100 days as the test set so that the model will not be trained on data that it will be used to test it.
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"] #List of columns used to predict target.
model.fit(train[predictors], train["Target"]) #Fit model using columns in predictors to predict the target.


from sklearn.metrics import precision_score #We will use prediction_score to find the probability that our prediction is correct.
import pandas as pd

preds = model.predict(test[predictors]) #Generate predictions by passing in the test set.
preds = pd.Series(preds, index=test.index) #Change predictions from numbpy array to Pandas series.
precision_score(test["Target"], preds) #Calculate precision_score using actual target and predicted target.


#Combine data and plot a graph to visualise why prediction is poor.
combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


#Create a function to predict price.
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


#Create a function to backtest results.
def backtest(data, model, predictors, start=2500, step=250): 
    all_predictions = [] #List of dataframes where each dataframe is a prediction for a single year.
    for i in range(start, data.shape[0], step): #First 10 years to predict 11th year. Then, first 11 years to predict 12th year. And so on...
        train = data.iloc[0:i].copy() #Create train set.
        test = data.iloc[i:(i+step)].copy() #Create test set, where test set will be the current year.
        predictions = predict(train, test, predictors, model) #Generate predictions.
        all_predictions.append(predictions) #Add predictions for that particular year to list of all predictions.
    return pd.concat(all_predictions) #Combine all dataframes into a single dataframe.

predictions = backtest(sp500, model, predictors) #Backtest data with model and predictors created.
predictions["Predictions"].value_counts() #Find the count of predicted up days and down days.
precision_score(predictions["Target"], predictions["Predictions"]) #Get prediction score.
predictions["Target"].value_counts() / predictions.shape[0] #Find percentage of days where the market actually went up and down.


#Create a variety of rolling averages.
horizons = [2,5,60,250,1000] #We will calculate the mean close price in the past 2, 5, 60... trading days.
new_predictors = [] #List containing new columns to be created.

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
  
    ratio_column = f"Close_Ratio_{horizon}" 
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"] #Find ratio of average close price today and average close price in the past 2/5/60/... days.
    
    trend_column = f"Trend_{horizon}" 
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] #Find number of days in the past 2/5/60/... days that the price went up.
    
    new_predictors+= [ratio_column, trend_column]


#Cleanup model.
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"]) #Get rid of rows that contain missing data, NaN


#Improve model.
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] #Get probability that price goes up.
    preds[preds >=.6] = 1 #Increase chance that price will actually go up on that day.
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])














