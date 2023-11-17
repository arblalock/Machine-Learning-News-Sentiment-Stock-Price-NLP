# Machine-Learning-News-Sentiment-Stock-Price-NLP

Machine learning model which processes news sentiment using natural language processing and uses it to predict stock prices

### General notes

This was mainly organized for personal use but feel free to use for your own reference! If you would like to run it you'll need your own news and stock API access (I used free ones you'll see in the scripts). Be sure to set your own environment variables and update any file paths.

### News Loading

- Script: news_load.py
- The first step is loading news articles/headlines from news_load.py. The purpose of this script is to download news snippets and format them for the sentiment training.

### Sentiment Training

- Script: sentiment_training.py.
- This script trains the sentiment NLP model for use in the stock price prediction
- For sentiment training I used the Stanford Sentiment Analysis training set: [SSA](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2/). There are several formats, I used the one they label as "dictionary".

### Compile Data

- Script: compile_data.py
- This script generates/organizes the news sentiment and stock data and prepares it for model training

### Price Model Training/Prediction

- Script: stock_price_model.py
- This is used for the stock price model training and prediction based on the data prepared in the previous steps
- You can set predictors and outcome variables at top. I was mainly looking at whether prices and news headlines at beginning of day predicted prices at the end of the day
