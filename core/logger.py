import os
import pandas as pd
import numpy as np
from auxiliary.experiment_utils import plot_metrics as plot
from auxiliary.color_utils import xyz2srgb
import copy
import cv2

import ipdb

class Logger():
    def __init__(self, exp_cfg, exp_dir):

        self.exp_cfg = exp_cfg
        self.exp_dir = exp_dir

    def log_experiment_start(self, experiment=None):
        msg = f"Experiment Name: {self.exp_cfg.get('exp_name')}\n"
        msg += f"Train: {self.exp_cfg.get('train')}, Test: {self.exp_cfg.get('test')}\n"
        msg += f"Model Type: {self.exp_cfg.get('model_type')}\n"
        msg += f"Model(s): {self.exp_cfg.get('model_name')}\n"

        if experiment is not None:
            # Return experiment model parameters
            num_params, num_trainable = experiment.model.num_parameters()
            msg += f"Parameters: {num_params/1e6:.4f} M (of which {num_trainable/1e6:.4f} M trainable)\n"
            if getattr(experiment, "profile_model", True):
                profile_stats = experiment.model.profile()
                # Return experiment model FLOPs and profiling info
                msg += f"FLOPs: {profile_stats['flops']/1e9:.4f} G, Params: {profile_stats['params']/1e6:.4f} M\n"
                msg += f"Inference Time: {profile_stats['inference_time_ms']:.2f} ms\n"
                msg += f"FPS: {1000/profile_stats['inference_time_ms']:.2f}\n"
                msg += f"Max Memory Allocated: {profile_stats['max_memory_allocated_mb']:.2f} MB, Reserved: {profile_stats['max_memory_reserved_mb']:.2f} MB\n"
            else:
                msg += "Model profiling: disabled\n"

        msg += f"Dataset Root: {self.exp_cfg.get('dataset_root')}\n"
        msg += f"RGB Camera: {self.exp_cfg.get('rgb_camera')}" if self.exp_cfg.get('data_type') in ["RGB", "RGB+MS"] else ""
        msg += f", Spectral Camera: {self.exp_cfg.get('spectral_camera')}" if self.exp_cfg.get('data_type') in ["MS", "RGB+MS"] else ""
        msg += f", GT Type: {self.exp_cfg.get('gt_type')}\n"
        msg += f"Train List: {self.exp_cfg.get('train_list')}, Val List: {self.exp_cfg.get('val_list')}, Test List: {self.exp_cfg.get('test_list')}\n"
        msg += f"Seed: {self.exp_cfg.get('seed')}, Device: {self.exp_cfg.get('device')}\n"
        msg += f"Metrics: {', '.join(self.exp_cfg.get('metrics'))}\n"

        if self.exp_cfg.get('train'):
            msg += f"Epochs: {self.exp_cfg.get('n_epochs')}, Workers: {self.exp_cfg.get('n_workers')}, Learning Rate: {self.exp_cfg.get('lr')}\n"
            msg += f"Train Batch Size: {self.exp_cfg.get('train_batch_size')}, Val Batch Size: {self.exp_cfg.get('val_batch_size')}, Test Batch Size: {self.exp_cfg.get('test_batch_size')}\n"
            msg += f"Early Stop: {self.exp_cfg.get('early_stop')}, Criterion: {self.exp_cfg.get('criterion')}\n"

        msg += f"Experiment Directory: {self.exp_dir}, Checkpoint: {self.exp_cfg.get('checkpoint')}\n"
        print(msg)

        with open(os.path.join(self.exp_dir, "experiment_log.txt"), "a") as f:
            f.write(msg + "\n")

    def log_epoch_end(self, epoch, train_metrics, val_metrics):
        # Print metrics in a formatted way at the end of each epoch
        train_stats = train_metrics.get_last()
        val_stats = val_metrics.get_last()
        msg = f"Epoch {epoch+1}/{self.exp_cfg.get('n_epochs')}\n"
        for k, v in train_stats.items():
            msg += f"Train {k}: {v['mean']:.4f} | Val {k}: {val_stats[k]['mean']:.4f} \n"
        print(msg)
        
        self.log_to_file(msg, os.path.join(self.exp_dir, "experiment_log.txt")) # Log to file
        self.save_metrics_to_csv(os.path.join(self.exp_dir, "experiment_log.csv"), train_metrics, val_metrics) # Log to CSV

        self.plot_metrics(train_metrics, val_metrics, os.path.join(self.exp_dir, "plots.pdf")) # Plot metrics
        

    def log_test_result(self, test_metrics):
        # Print metrics in a formatted way at the end of each epoch
        test_stats = test_metrics.get_last()
        msg = "Test results:\n"
        for k, v in test_stats.items():
            msg += f"{k}: {test_stats[k]['mean']:.4f} \n"
        print(msg)
        
        self.log_to_file(msg, os.path.join(self.exp_dir, "test_log.txt")) # Log to file
        self.save_metrics_to_csv(os.path.join(self.exp_dir, "test_log.csv"), test_metrics) # Log to CSV
        
    def log_gradient_error(self, epoch, iter):
        msg = f"Gradient error at epoch {epoch+1}, iter {iter+1}. Step skipped.\n"
        print(msg)
        self.log_to_file(msg, os.path.join(self.exp_dir, "experiment_log.txt"))

    def save_per_image_metrics(self, per_image_metrics):
        """Save per-image metrics to a CSV file.
        
        Args:
            per_image_metrics: List of dicts, each containing 'file_name' and metric values
        """
        if not per_image_metrics:
            return
        
        df = pd.DataFrame(per_image_metrics)
        # Reorder columns to have file_name first
        cols = ['file_name'] + [c for c in df.columns if c != 'file_name']
        df = df[cols]
        # Round numeric columns
        for col in df.columns:
            if col != 'file_name' and df[col].dtype in ['float64', 'float32']:
                df[col] = np.round(df[col], 4)
        
        path = os.path.join(self.exp_dir, "per_image_metrics.csv")
        df.to_csv(path, index=False)
        print(f"Saved per-image metrics to {path}")

    @staticmethod
    def log_to_file(msg, path):
        with open(path, "a") as f:
            f.write(msg + "\n")


    def save_metrics_to_csv(self, path, metrics_set1, metrics_set2=None):
        if metrics_set2:
            self.save_training_metrics_to_csv(metrics_set1, metrics_set2, path)
        else:
            self.save_test_metrics_to_csv(metrics_set1, path)


    @staticmethod
    def save_training_metrics_to_csv(train_metrics, val_metrics, path):
        # Save the metrics from all epochs to a CSV file
        epochs = len(train_metrics.epoch_values["Loss"])
        log_data = []

        for epoch in range(epochs):
            epoch_data = {"Epoch": epoch + 1}
            train_stats = {k: v[epoch] for k, v in train_metrics.epoch_values.items()}
            val_stats = {k: v[epoch] for k, v in val_metrics.epoch_values.items()}

            for metric, stats in train_stats.items():
                for stat_name, stat_value in stats.items():
                    epoch_data[f"Train_{metric}_{stat_name}"] = np.round(stat_value, 4)
            for metric, stats in val_stats.items():
                for stat_name, stat_value in stats.items():
                    epoch_data[f"Val_{metric}_{stat_name}"] = np.round(stat_value, 4)

            log_data.append(epoch_data)

        df = pd.DataFrame(log_data)
        df.to_csv(path, index=False)

    @staticmethod
    def save_test_metrics_to_csv(test_metrics, path):
        # Save the metrics from all epochs to a CSV file
        test_stats = {k: v[0] for k, v in test_metrics.epoch_values.items()}
        
        data = {}
        for metric, stats in test_stats.items():
            for stat_name, stat_value in stats.items():
                data[f"Test_{metric}_{stat_name}"] = np.round(stat_value, 4)
        log_data = [data]

        df = pd.DataFrame(log_data)
        df.to_csv(path, index=False)

    @staticmethod
    def plot_metrics(train_metrics, val_metrics, path):
        
        t_m = copy.deepcopy(train_metrics)
        v_m = copy.deepcopy(val_metrics)
        for k,_ in t_m.epoch_values.items():
            t_m.epoch_values[k] = [el["mean"] for el in t_m.epoch_values[k]]
            v_m.epoch_values[k] = [el["mean"] for el in v_m.epoch_values[k]]

        plot(t_m.epoch_values, v_m.epoch_values, path)


    def save_viz(self, pred, gt, path):

        pred = (xyz2srgb(pred[None,...])[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        gt = (xyz2srgb(gt[None,...])[0].permute(1,2,0).numpy() * 255).astype(np.uint8)

        out = np.concatenate((pred,gt), axis=1)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path, out)








