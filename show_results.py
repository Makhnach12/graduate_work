from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from confident_learning_method import predict_confident_learning
from svm_model import predict_svm
from torch_model import predict_torch

DIRECTORY: str = 'data/'
NAMES: list[str] = ['data51.csv', 'data54.csv', 'data_part1.csv']


def get_results(method: Callable) -> None:
    for name in NAMES:
        data = pd.read_csv(DIRECTORY + name)

        X = data[['dis', 'iou', 'conf', 'dx1', 'dy1', 'dx2', 'dy2']]
        y = data['problem'].values

        y_prediction: np.array = method(X)

        cm = confusion_matrix(y, y_prediction)

        # Визуализация
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0 (TN/FP)', 'Predicted 1 (FN/TP)'],
                    yticklabels=['Actual 0 (TN/FN)', 'Actual 1 (FP/TP)'])
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.title('Confusion Matrix (TP, TN, FP, FN)')

        plt.savefig(f'confusion_matrix_{name}.png', dpi=300,
                    bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    get_results(predict_confident_learning)
