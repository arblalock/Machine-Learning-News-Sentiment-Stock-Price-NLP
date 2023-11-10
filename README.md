# Machine-Learning-News-Sentiment-Stock-Price-NLP

Machine learning model which processes news sentiment using natural language processing and uses it to predict stock prices

### General notes

This was mainly organized for personal use so if you would like to use it you’ll need to make some modifications. Otherwise this repo can mainly just be used for general reference. If you would like to run it you'll need your own news and stock API access (I used free ones you'll see in the scripts). Be sure to set your own environment variables and update any file paths.

### Data prep

-Scripts: compile_data.py, news_load.py
-The first two steps are getting the stock price data using compile_data.py and then loading news articles/headlines from news_load.py.

- The news api used here is limited in how much text it gives because it's free. Using a paid service such as Bloomberg news would most likely significantly improve training.

### Training

-Script: training.py.
-For sentiment training I used the Stanford Sentiment Analysis training set. I don't have the original link but it may be this one: [SSA](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2/). There are several formats, I used the one they label as "dictionary"

### Prediction

-Script: prediction.py
-You can set predictors and outcome variables at top. I was mainly looking at whether prices and news headlines at beginning of day predicted prices at the end of the day
