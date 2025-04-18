import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVC

TRAIN_DATA_PATH: str = 'data_part1.csv'
WEIGHT_NULL_CLASS: int = 1
WEIGHT_FIRST_CLASS: int = 100


def predict_svm(X: DataFrame) -> np.array:
    with open('models/model_svm.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    # из-за перевеса данных с классом 0 необходимо инвертировать данные
    return 1 - loaded_model.predict(X)


if __name__ == '__main__':
    data = pd.read_csv(TRAIN_DATA_PATH)
    X = data[['dis', 'iou', 'conf', 'dx1', 'dy1', 'dx2', 'dy2']]
    y = data['problem']
    clf = SVC(
        kernel='rbf',
        class_weight={0: WEIGHT_NULL_CLASS, 1: WEIGHT_FIRST_CLASS}
    )
    clf.fit(X, y)
