import numpy as np
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
        return {"relative_error": self.relative_error.mean().cpu().item(), "acc": self.acc.float().mean().cpu().item()}

def average_multiple_run(results):
    if len(results) == 1:
        return results
    else:
        averaged = {}
        for key in results[0].keys():
            averaged[f"{key}_mean"] = np.mean([x[key] for x in results])
            averaged[f"{key}_std"] = np.std([x[key] for x in results])
        return averaged