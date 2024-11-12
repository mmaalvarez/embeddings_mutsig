# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: embeddings_mutsig
#     language: python
#     name: embeddings_mutsig
# ---

# main.py
from model import CNN_DNAClassifier, train_model
import sys, os, argparse
from data import load_data
from embeddings import save_all_embeddings_probs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import seaborn as sns
import wandb
from types import SimpleNamespace


# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# +
## pass arguments
parser = argparse.ArgumentParser()
#parser.add_argument('-r', '--filename_real_data', type=str, required=True, help='Real dataset (i.e. before resamplings), all samples')

if 'ipykernel' in sys.modules:
    
    # if interactive, pass values manually

    work_dir='/home/jovyan/fsupek_data/users/malvarez/projects/embeddings_mutsig/'
    
    # Path with files
    path = f'{work_dir}/data/nn_input/test_patri/'

    #### CONFIG
    # Define your config with dot notation access
    config = SimpleNamespace(learning_rate=0.0008,
                             patience=20,
                             fc1_neurons=128,
                             fc2_neurons=128,
                             dropout1_rate=0.2,
                             dropout2_rate=0.3,
                             kernel_size1=3,
                             kernel_size2=3,
                             kernel_size3=3)
    batch_size = 256
    
else:
    
    # otherwise, load arguments
    
    args = parser.parse_args()
    #filename_real_data = args.filename_real_data


# +
train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping = load_data(path, device, batch_size)

print("loaded data")
print(label_mapping)

# N cancer types
n_ct = len(label_mapping)


# +
if not os.path.exists(f'{work_dir}/embeddings/CNN_models/'):
    os.makedirs(f'{work_dir}/embeddings/CNN_models/')

if not os.path.exists(f'{work_dir}/embeddings/loss_auc_curves/'):
    os.makedirs(f'{work_dir}/embeddings/loss_auc_curves/')
    
model_path = f'{work_dir}/embeddings/CNN_models/best_model_test11.pth'

model = CNN_DNAClassifier(config, n_ct).to(device)

# Train the model normally (NO WANDB SWEEPS)
model = train_model(model_path, work_dir, config, n_ct, train_loader, val_loader, class_weights_tensor, device, wandb=False)

print("Finished training")

model = CNN_DNAClassifier(config, n_ct).to(device)

# +
print("LOADING MODEL")

model.load_state_dict(torch.load(model_path))


# +
# Save embeddings

if not os.path.exists(f'{work_dir}/embeddings/saved_embeddings_myCNN/'):
    os.makedirs(f'{work_dir}/embeddings/saved_embeddings_myCNN/')
    
save_all_embeddings_probs(model, test_labels, test_sequences_og, label_mapping, f'{work_dir}/embeddings/saved_embeddings_myCNN/test_embeddings_probs_t11_conv1.csv')
# -


