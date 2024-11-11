# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# main.py
from model import CNN_DNAClassifier, train_model
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
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import seaborn as sns
import wandb
from types import SimpleNamespace


# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


work_dir='XXXXX'


#### CONFIG
# Define your config with dot notation access
config = SimpleNamespace(
    learning_rate=0.0008,
    patience=20,
    fc1_neurons=128,
    fc2_neurons=128,
    dropout1_rate=0.2,
    dropout2_rate=0.3,
    kernel_size1=3,
    kernel_size2=3,
    kernel_size3=3
)

# Path with files
path = f'{work_dir}/embeddings/saved_seqs/saved_seqs_filt_insALT_test11/'


train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping = load_data(path, device, batch_size=256)

print("loaded data")

print(label_mapping)


model_path=f'{work_dir}/embeddings/CNN_models/best_model_test11.pth'

model = CNN_DNAClassifier(config).to(device)


# Train the model normally (NO WANDB SWEEPS)
model = train_model(model_path, work_dir, config, train_loader, val_loader, class_weights_tensor, device, wandb=False)

print("Finished training")

model = CNN_DNAClassifier(config).to(device)


print("LOADING MODEL")

model.load_state_dict(torch.load(model_path))


#Save embeddings
save_all_embeddings_probs(model, test_labels, test_sequences_og, label_mapping, f'{work_dir}/embeddings/saved_embeddings_myCNN/test_embeddings_probs_t11_conv1.csv')
