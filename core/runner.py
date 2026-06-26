import os
import torch
import numpy as np
import tqdm
from core.experiment import Experiment
from core.logger import Logger
from core.evaluator import Evaluator


import pandas as pd

import ipdb


class Runner():
    def __init__(self, cfg, **kwargs):

        self.experiment = Experiment(filepath=cfg, **kwargs)

    def train(self):

        for epoch in range(self.experiment.starting_epoch, self.experiment.n_epochs):
            self.experiment.model.model.train()
            train_loop = tqdm.tqdm(self.experiment.train_loader, desc=f"Epoch {epoch+1}/{self.experiment.n_epochs} - Training")
            for i, data in enumerate(train_loop):
                loss = 0
                loss, pred, backward_status = self.experiment.model.train_step(
                    data,
                    compute_metrics=self.experiment.train_metrics_enabled,
                )
                if backward_status != 0:
                    self.experiment.logger.log_gradient_error(epoch=epoch, iter=i)           
                if self.experiment.train_metrics_enabled:
                    self.experiment.train_metrics.update(pred, data["gt_image"].to(self.experiment.device, non_blocking=self.experiment.non_blocking), loss)
                else:
                    self.experiment.train_metrics.update_loss(loss)
                train_loop.set_postfix(loss=loss)

            self.experiment.model.eval()
            val_loop = tqdm.tqdm(self.experiment.val_loader, desc=f"Epoch {epoch+1}/{self.experiment.n_epochs} - Validation")
            with torch.no_grad():
                for i, data in enumerate(val_loop):
                    loss = 0
                    loss, pred = self.experiment.model.eval_step(data)
                    self.experiment.val_metrics.update(pred, data["gt_image"].to(self.experiment.device, non_blocking=self.experiment.non_blocking), loss)

                    if self.experiment.val_viz_list and any([x in self.experiment.val_viz_list for x in data["file_name"]]):
                        idx = [x in self.experiment.val_viz_list for x in data["file_name"]].index(True)
                        de00 = self.experiment.val_metrics.iter_values["deltaE00"][-pred.shape[0]:][idx]
                        self.experiment.logger.save_viz(pred[idx].detach().cpu(), data["gt_image"][idx].detach().cpu(), os.path.join(self.experiment.exp_dir, "val_viz", f"ep{(epoch+1):03d}_"+data["file_name"][idx]+f"(dE00={de00:.2f}).png"))

                    val_loop.set_postfix(loss=loss)

            self.experiment.train_metrics.aggregate()
            self.experiment.val_metrics.aggregate()


            self.experiment.logger.log_epoch_end(epoch, self.experiment.train_metrics, self.experiment.val_metrics)
            self.experiment.save_checkpoint(os.path.join(self.experiment.exp_dir, "last.pth"), epoch=epoch)
            # Early stopping based on validation loss
            val_loss = self.experiment.val_metrics.get_last(stat="mean")["Loss"]
            
            if self.experiment.early_stop is not None: 
                if val_loss < self.experiment.best_loss:
                    self.experiment.best_loss = val_loss
                    self.experiment.early_stop_counter = 0
                    self.experiment.model.save(os.path.join(self.experiment.exp_dir, "best.pth"))
                else:
                    self.experiment.early_stop_counter += 1
                    if self.experiment.early_stop_counter >= self.experiment.early_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            self.experiment.scheduler.step()
            
            

    def test(self):
        if self.experiment.train:
            self.experiment.model.load(os.path.join(self.experiment.exp_dir, "best.pth"))
            self.experiment.model.to(self.experiment.device, device_ids=self.experiment.device_ids)

        self.experiment.model.eval()
        test_loop = tqdm.tqdm(self.experiment.test_loader, desc=f"Testing")



        # Per-image metrics storage
        per_image_metrics = []

        with torch.no_grad():
            for i, data in enumerate(test_loop):
                loss = 0
                loss, pred = self.experiment.model.eval_step(data)
                                
                self.experiment.test_metrics.update(pred, data["gt_image"].to(self.experiment.device, non_blocking=self.experiment.non_blocking), loss)

                # Get the metrics for current batch (last batch_size values)
                batch_size = pred.shape[0]
                
                # Collect per-image metrics for each image in the batch
                for idx in range(batch_size):
                    image_metrics = {"file_name": data["file_name"][idx]}
                    for metric_name in self.experiment.test_metrics.iter_values.keys():
                        if metric_name != "Loss":
                            image_metrics[metric_name] = self.experiment.test_metrics.iter_values[metric_name][-batch_size + idx]
                    per_image_metrics.append(image_metrics)

                # Save visualizations for images in test_viz_list
                if self.experiment.test_viz_list and any([x in self.experiment.test_viz_list for x in data["file_name"]]):
                    idx = [x in self.experiment.test_viz_list for x in data["file_name"]].index(True)
                    de00 = self.experiment.test_metrics.iter_values["deltaE00"][-batch_size:][idx]
                    self.experiment.logger.save_viz(pred[idx].detach().cpu(), data["gt_image"][idx].detach().cpu(), os.path.join(self.experiment.exp_dir, "test_viz", data["file_name"][idx]+f"(dE00={de00:.2f}).png"))

                # Save visualizations for images with deltaE00 in specified range
                if self.experiment.test_viz_de00_range is not None:
                    de00_min, de00_max = self.experiment.test_viz_de00_range
                    batch_de00 = self.experiment.test_metrics.iter_values["deltaE00"][-batch_size:]
                    for idx in range(batch_size):
                        de00 = float(batch_de00[idx])
                        if de00_min <= de00 <= de00_max:
                            # Avoid saving duplicates if also in test_viz_list
                            if self.experiment.test_viz_list and data["file_name"][idx] in self.experiment.test_viz_list:
                                continue
                            self.experiment.logger.save_viz(
                                pred[idx].detach().cpu(), 
                                data["gt_image"][idx].detach().cpu(), 
                                os.path.join(self.experiment.exp_dir, "test_viz", data["file_name"][idx]+f"(dE00={de00:.2f})_WC.png")
                            )

                test_loop.set_postfix(loss=loss)

        self.experiment.test_metrics.aggregate()

        

        # Save per-image metrics
        self.experiment.logger.save_per_image_metrics(per_image_metrics)

        self.experiment.logger.log_test_result(self.experiment.test_metrics)

    def run(self):
        self.experiment.logger.log_experiment_start(self.experiment)
        if self.experiment.train:
            self.train()
        if self.experiment.test:
            self.test()