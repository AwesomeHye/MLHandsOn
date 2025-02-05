from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic_stack = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        prediction = self.logistic_stack(data)
        return prediction