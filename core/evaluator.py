import torch
from auxiliary.metrics import *


class Evaluator():

    def __init__(self, metrics_list):
        self.metrics = {}
        self.iter_values = {}
        self.epoch_values = {}
        for m in metrics_list:
            self.metrics[m] = eval(m)()
            self.iter_values[m] = []
            self.epoch_values[m] = []

        self.iter_values["Loss"] = []
        self.epoch_values["Loss"] = []

    def update(self, pred, gt, loss):
        for k in self.metrics.keys():
            val = self.metrics[k](pred, gt)
            self.iter_values[k] += val
        self.iter_values["Loss"].append(loss)

    def update_loss(self, loss):
        self.iter_values["Loss"].append(loss)
        
    def aggregate(self):
        for k in self.iter_values.keys():
            self.iter_values[k] = np.array(self.iter_values[k])
            res = {
                "mean": np.mean(self.iter_values[k]),
                "median": np.median(self.iter_values[k]),
                "trimean": ((np.percentile(self.iter_values[k], 25) + 2 * np.median(self.iter_values[k]) + np.percentile(self.iter_values[k], 75)) / 4),
                "B-25": np.mean(self.iter_values[k][self.iter_values[k] <= np.percentile(self.iter_values[k], 25)]),
                "W-25": np.mean(self.iter_values[k][self.iter_values[k] >= np.percentile(self.iter_values[k], 75)]),
                "95-P": np.percentile(self.iter_values[k], 95),
                "99-P": np.percentile(self.iter_values[k], 99),
                "Max": np.max(self.iter_values[k]),
            }

            self.epoch_values[k].append(res)
            self.iter_values[k] = []
    
    def get_last(self, stat=None):
        out = {}
        for k in self.epoch_values.keys():
            if len(self.epoch_values[k]) > 0:
                out[k] = self.epoch_values[k][-1][stat] if stat is not None else self.epoch_values[k][-1]
            else:
                out[k] = None

        return out
