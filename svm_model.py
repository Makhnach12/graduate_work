import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVC

TRAIN_DATA_PATH: str = 'data/exp3/data_part1'
WEIGHT_NULL_CLASS: int = 1
WEIGHT_FIRST_CLASS: int = 100


def predict_svm(X: DataFrame, exp: str) -> np.array:
    with open(f'models/{exp}/model_svm.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    # из-за перевеса данных с классом 0 необходимо инвертировать данные
    return loaded_model.predict(X)


if __name__ == '__main__':
    data = pd.read_csv(TRAIN_DATA_PATH)
    columns_to_keep = [col for col in data.columns if
                       col not in ['frame', 'problem']]
    X = data[columns_to_keep]
    y = data['problem']
    clf = SVC(
        kernel='rbf',
        class_weight={0: WEIGHT_NULL_CLASS, 1: WEIGHT_FIRST_CLASS}
    )
    clf.fit(X, y)
    with open('models/exp3/model_svm.pkl', 'wb') as f:
        pickle.dump(clf, f)
