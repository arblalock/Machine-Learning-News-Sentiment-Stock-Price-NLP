# Load Libraries
# %%
import os
os.chdir('/workspaces/MachineLearning/News_Sentiment/')
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import requests

STOCK_API_KEY = os.environ.get('STOCK_API_KEY')
STOCK_SYM = 'SPY'
STOCK_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={STOCK_SYM}&interval=30min&outputsize=full&apikey={STOCK_API_KEY}'
MODEL_SAVE_PATH = './saved_models/'
NEWS_SAVE_PATH = './data/news/'
TRAINING_SAVE_PATH = './data/'
STOCK_HIST_SAVE_PATH = './data/stock_hist/'
MODEL_NAME = 'first'
NEWS_NAME = '2020-07-18_2020-09-16_T05-16.csv'
STOCK_HIST_NAME_FULL = 'full_SPY_07_13_20-09_17_20.csv'
STOCK_HIST_NAME_FORMATTED = 'formatted_SPY_07_13_20-09_17_20.csv'
TRAIN_NAME = 'training.csv'

# %%
# Get/save sentiment
news_df = pd.read_csv(NEWS_SAVE_PATH+NEWS_NAME)
news_df = news_df.loc[:,~news_df.columns.str.startswith('Unnamed')]
model = tf.keras.models.load_model(MODEL_SAVE_PATH+MODEL_NAME)
news_df = news_df.fillna('x')
news_df['sentiment_headline'] = model.predict(np.asarray(news_df['title'].astype(str)))
news_df['sentiment_description'] = model.predict(np.asarray(news_df['description'].astype(str)))
news_df['sentiment_content'] = model.predict(np.asarray(news_df['content'].astype(str)))
news_df['date'] = list(map(lambda x: x.split("T")[0], list(news_df['publishedAt'])))
news_df.to_csv(NEWS_SAVE_PATH+NEWS_NAME)

# %%
# Get/save Stock Prices
r = requests.get(STOCK_URL)
result_json = r.json()
stock_df = pd.DataFrame.from_dict(result_json['Time Series (30min)'], orient='index')
stock_df.to_csv(STOCK_HIST_SAVE_PATH+STOCK_HIST_NAME_FULL)

#%%
# Format historical stock
stock_df = pd.read_csv(STOCK_HIST_SAVE_PATH+STOCK_HIST_NAME_FULL)
stock_df.rename(columns={ stock_df.columns[0]: "date" }, inplace = True)
formatted_stock_df = pd.DataFrame()

#****NOTE****: only need next line if excel has changed the date formatting, will need to change formatting if excel didnt auto format
dates = list(set(map(lambda x: datetime.strptime(x.split(" ")[0], "%m/%d/%Y").strftime('%Y-%m-%d') , list(stock_df['date']))))
dates.sort(reverse=True)
formatted_stock_df['date'] = dates
formatted_stock_df.set_index('date')
formatted_stock_df['open_price'] = list(stock_df[stock_df['date'].str.contains(' 9:30')]['4. close'])
formatted_stock_df['noon_price'] = list(stock_df[stock_df['date'].str.contains('12:00')]['4. close'])
formatted_stock_df['330_price'] = list(stock_df[stock_df['date'].str.contains('15:30')]['4. close'])
formatted_stock_df['close_price'] = list(stock_df[stock_df['date'].str.contains('16:00')]['4. close'])

formatted_stock_df['LG_12-330'] = np.where(formatted_stock_df['noon_price'] > formatted_stock_df['330_price'], 0, 1)
formatted_stock_df['LG_12-close'] = np.where(formatted_stock_df['noon_price'] > formatted_stock_df['close_price'], 0, 1)
formatted_stock_df.to_csv(STOCK_HIST_SAVE_PATH+STOCK_HIST_NAME_FORMATTED, index=False)


#%%
# Format training file
formatted_stock_df = pd.read_csv(STOCK_HIST_SAVE_PATH+STOCK_HIST_NAME_FORMATTED)
news_df = pd.read_csv(NEWS_SAVE_PATH+NEWS_NAME)
#****NOTE****: only need next line if excel has changed the date formatting
news_df['date'] = news_df['date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").strftime('%Y-%m-%d'))


training_df = formatted_stock_df.copy()
training_df['headline_mean'] = formatted_stock_df['date'].apply(lambda x: news_df[news_df['date'] == x]['sentiment_headline'].mean())
training_df['description_mean'] = formatted_stock_df['date'].apply(lambda x: news_df[news_df['date'] == x]['sentiment_description'].mean())
training_df['content_mean'] = formatted_stock_df['date'].apply(lambda x: news_df[news_df['date'] == x]['sentiment_content'].mean())
training_df = training_df.dropna()
training_df.to_csv(TRAINING_SAVE_PATH+TRAIN_NAME, index=False)
