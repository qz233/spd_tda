import torch



class Evaluator():
    def __init__(self):
        self.relative_error = []
        self.acc = []

    def update(self, input_batch, pred):
        distance = input_batch[2]
        relative_error = (distance - pred).abs() / distance 
        correct_pred = relative_error < 0.02
        self.relative_error.append(relative_error)
        self.acc.append(correct_pred)

    def aggregate(self, ):
        self.relative_error = torch.cat(self.relative_error)
        self.acc = torch.cat(self.acc)
        return {"relative_error": self.relative_error.mean(), "acc": self.acc.float().mean()}
