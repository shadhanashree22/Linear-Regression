# Linear-Regression
## NAME: S V SHADHANASHREE
## REGISTER NUMBER:212223230202

## a) Stock Market Prediction using Linear Regression

## Aim

To build a machine learning model using Linear Regression to predict future stock closing prices based on historical stock market data containing Open, High, Low, and Volume values.

## Algorithm

Import required libraries: Pandas, NumPy, Matplotlib, Scikit-learn.

Load Dataset (stock_data_big.csv) using Pandas.

Explore Dataset by displaying the first few rows.

Select Features (X) such as Open, High, Low, Volume and Target (y) as Close price.

Split the dataset into training and testing sets (80% train, 20% test).

Create the Linear Regression model and train it using the training set.

Predict the closing prices for the test dataset.

Evaluate the model using Mean Squared Error (MSE) and R² Score.

Visualize the Actual vs Predicted stock closing prices using Matplotlib.

Interpret results to understand prediction accuracy.

```Python
from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("stock_data_big.csv")

print(data.head())

X = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

plt.figure(figsize=(8,5))
plt.plot(y_test.values[:50], label='Actual Prices', color='blue')
plt.plot(predictions[:50], label='Predicted Prices', color='red')
plt.title("Stock Price Prediction (Actual vs Predicted)")
plt.xlabel("Time")
plt.ylabel("Stock Close Price")
plt.legend()
plt.show()
```

## b) Real-Time Sentiment Analysis of Tweets / Reviews

## Aim

To perform real-time sentiment analysis on user-provided tweets or text reviews using TextBlob and classify them into Positive, Negative, or Neutral categories.

## Algorithm

Import libraries: Pandas, TextBlob, and Matplotlib.

Load the dataset (tweets_big.csv) containing tweet/review text.

Define a sentiment function using TextBlob to compute the polarity of each text:

Polarity > 0 → Positive

Polarity < 0 → Negative

Polarity = 0 → Neutral

Apply the function to each tweet/review and store the sentiment category.

Count the occurrences of each sentiment type.

Display sentiment count results in the console.

Visualize the distribution of sentiments using a bar chart.

Show sample outputs to verify classification accuracy.


```Python

from google.colab import files
uploaded = files.upload()
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_csv("tweets_big.csv")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"


data["Sentiment"] = data["text"].apply(get_sentiment)

sentiment_counts = data["Sentiment"].value_counts()
print(sentiment_counts)

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green','red','gray'])
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Tweets/Reviews")
plt.show()

print("\nSample Results:")
print(data[["text", "Sentiment"]].head())
```
