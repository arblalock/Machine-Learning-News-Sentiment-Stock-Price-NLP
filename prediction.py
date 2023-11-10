# Load Libraries
# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pre_process
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
import statsmodels.api as sm

TRAINING_SAVE_PATH = "./data/"
TRAIN_NAME = "training.csv"
train_df = pd.read_csv(TRAINING_SAVE_PATH + TRAIN_NAME)

# %%
# format data

# predictors = ['headline_mean', 'description_mean', 'open_price', 'noon_price']
predictors = ["headline_mean", "noon_price", "open_price", "description_mean"]
# outcome = 'LG_12-330'
outcome = "330_price"

# Corelation
plt.figure()
cor = train_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

X = train_df[predictors]
y = train_df[outcome]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

sc = StandardScaler()
X_train = pd.DataFrame(
    sc.fit_transform(X_train), index=X_train.index, columns=X_train.columns
)
X_test = pd.DataFrame(sc.transform(X_test), index=X_test.index, columns=X_test.columns)

# %%
# Regression
reg_model = LinearRegression()
# reg_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)
# X_sm = sm.add_constant(X)
# sm_model = sm.OLS(y, X_sm)
# sm_results = sm_model.fit()
# print(sm_results.summary())

regression_df = train_df[["noon_price", "330_price"]].copy()
regression_df["reg_preds"] = None
regression_df["reg_correct"] = None

for x in range(len(X_test)):
    regression_df["reg_preds"][X_test.index[x]] = y_pred_reg[x]

for x in range(len(X_test)):
    if (
        y_pred_reg[x] >= train_df["noon_price"][X_test.index[x]]
        and train_df["LG_12-330"][X_test.index[x]] == 1
    ):
        regression_df["reg_correct"][X_test.index[x]] = 1
    elif (
        y_pred_reg[x] < train_df["noon_price"][X_test.index[x]]
        and train_df["LG_12-330"][X_test.index[x]] == 0
    ):
        regression_df["reg_correct"][X_test.index[x]] = 1
    else:
        regression_df["reg_correct"][X_test.index[x]] = 0

print("regression acurracy: " + str(regression_df["reg_correct"].sum() / len(X_test)))
mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print("MSE: " + str(mse))
print("MAE: " + str(mae))
print("R2: " + str(r2))


# %%
# Classification
class_model = RandomForestClassifier(
    n_estimators=10, criterion="entropy", random_state=0
)
class_model.fit(X_train.values, y_train.values)
y_pred_class = class_model.predict(X_test)
target_names = ["Loss", "Gain"]
print(classification_report(y_test.values, y_pred_class, target_names=target_names))
print(accuracy_score(y_test.values, y_pred_class))
class_df = train_df[["noon_price", outcome]].copy()
class_df["class_preds"] = ""
for x in range(len(X_test)):
    class_df["class_preds"][X_test.index[x]] = y_pred_class[x]
