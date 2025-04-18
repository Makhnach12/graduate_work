import numpy as np
import torch
from pandas import DataFrame
from torch import nn


class Classifier(nn.Module):
    def __init__(self, output_bias=None):
        super(Classifier, self).__init__()
        self.dense1 = nn.Linear(7, 20)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(20, 2)

        if output_bias is not None:
            self.output.bias.data.fill_(output_bias)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


def predict_torch(X: DataFrame) -> np.array:
    model = Classifier()
    model.load_state_dict(torch.load('models/model_weights.pth'))
    model.eval()
    X_tensor = torch.from_numpy(X.values).float()
    with torch.no_grad():
        predictions = model(X_tensor)
    return predictions.argmax(dim=1).numpy()
