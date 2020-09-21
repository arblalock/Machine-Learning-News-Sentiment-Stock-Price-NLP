import numpy as np

def create_df(data, features, target):
    data = data.fillna('missing')
    return data.filter(features+target)

