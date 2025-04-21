import numpy as np
from pandas import DataFrame


def predict_confident_learning(X: DataFrame) -> np.array:
    y_prediction = []
    columns_to_keep = [col for col in X.columns if
                       col not in ['frame', 'problem']]
    for _, row in X.iterrows():
        if row[columns_to_keep.index('iou')] < 0.8 and \
                row[columns_to_keep.index('conf')] > 0.5:
            y_prediction.append(1)
        else:
            y_prediction.append(0)
    return np.array(y_prediction)