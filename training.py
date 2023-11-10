# %%
#Import
import os
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pre_process
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

%load_ext autoreload
%autoreload 2
EMBED_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
model  = None
SENT_TRAIN_PATH = './data/sentiment_training/stanfordSent/'

# create csv
df_data_sentence = pd.read_table(SENT_TRAIN_PATH + 'dictionary.txt')
df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})
# read sentiment labels into df
df_data_sentiment = pd.read_table(SENT_TRAIN_PATH + 'sentiment_labels.txt')
df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})
#combine data frames containing sentence and sentiment
df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')
df_processed_all.to_csv(SENT_TRAIN_PATH + 'train.csv')

# %%
#Data loading
train_data = pd.read_csv(SENT_TRAIN_PATH+'train.csv')

#Globals
MODEL_SAVE_PATH = './saved_models/'
TEST_SIZE = 0.1
RAND = 10
features = ['Phrase']
target = ['sentiment_values']

# Exploring
# print(train_data.head())
print(train_data.describe())
# print(train_data.isnull().sum())

#Data processing
train_df = pre_process.create_df(train_data, features, target)
train_raw, test_raw = train_test_split(train_df, test_size=TEST_SIZE, random_state=RAND)
x_train = train_raw[features].values.ravel()
y_train = train_raw[target].values.ravel()
x_test= test_raw[features].values.ravel()
y_test= test_raw[target].values.ravel()


# %%
# Train Model
# Training settings
BATCH_SIZE = 1024
EPOCS = 10
LEARNING_RATE = 0.001
DROPOUT = 0
SHUFFLE = False
L2 = 1e-8
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# model
class MyModel(tf.keras.Model):
  def __init__(self, hub_url, l2):
    super().__init__()
    self.hub_url = hub_url
    self.l2 = l2
    self.embed = hub.load(self.hub_url)
    self.sequential = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu', activity_regularizer=regularizers.l2(l2)),
      tf.keras.layers.Dense(256, activation='relu', activity_regularizer=regularizers.l2(l2)),
      tf.keras.layers.Dense(256, activation='relu', activity_regularizer=regularizers.l2(l2)),
      tf.keras.layers.Dense(1)
    ])

  def call(self, inputs):
    embedding = self.embed(tf.squeeze(tf.cast(inputs, tf.string)))
    return self.sequential(embedding)

  def get_config(self):
    return {"hub_url":self.hub_url, "l2":self.l2}
if model != None:   
    del model
model = MyModel(EMBED_URL, L2)

model.compile(optimizer=opt,
              loss='mse',
              metrics=['mae', 'mse'])

# train           
history = model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=EPOCS,
          shuffle=SHUFFLE)

# plot
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model loss')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate
y_pred = model.predict(x_test)
model.predict(x_test)
print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
print('MAE: ' + str(mean_absolute_error(y_test, y_pred)))
print('R2: ' + str(r2_score(y_test, y_pred)))


# %%
# Save Model
MODEL_NAME = 'first'
model.save(MODEL_SAVE_PATH+MODEL_NAME)






