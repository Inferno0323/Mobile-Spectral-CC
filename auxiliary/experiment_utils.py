import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import random
import re
import ipdb

def seed_everything(seed, deterministic=True):
    """
    Makes experiments deterministic
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.utils.deterministic.fill_uninitialized_memory_(True)

def plot_metrics(train_metrics, val_metrics, pdf_path):
    """
    Create a plot of the loss and metrics and saves it to a multi-page pdf file
    """
   
    with PdfPages(pdf_path) as pdf:
        for k in train_metrics.keys():
            plt.figure()
            plt.plot(np.arange(1, len(train_metrics[k])+1,1), train_metrics[k], label="Train")
            plt.plot(np.arange(1, len(train_metrics[k])+1,1), val_metrics[k], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel(k)
            plt.legend()
            plt.title(k)

        for i in plt.get_fignums():
            pdf.savefig(plt.figure(i))
        plt.close("all")
