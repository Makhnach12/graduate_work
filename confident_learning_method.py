import numpy as np
from pandas import DataFrame


def predict_confident_learning(X: DataFrame) -> np.array:
    y_prediction = []
    for _, row in X.iterrows():
        if row[1] < 0.8 and row[2] > 0.5:
            y_prediction.append(1)
        else:
            y_prediction.append(0)
    return np.array(y_prediction)