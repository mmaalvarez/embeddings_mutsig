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
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import seaborn as sns
from types import SimpleNamespace


# set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# +
# pass arguments
parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, required=True)
parser.add_argument('--files_dir', type=str, required=True)
parser.add_argument('--training_set', type=str, required=True)
parser.add_argument('--validation_set', type=str, required=True)
parser.add_argument('--testing_set', type=str, required=True)
parser.add_argument('--all_sets', type=str, required=True)
parser.add_argument('--batch_size', type=str, default=256)
parser.add_argument('--learning_rate', type=str, default=0.0008)
parser.add_argument('--patience', type=str, default=20)
parser.add_argument('--fc1_neurons', type=str, default=128)
parser.add_argument('--fc2_neurons', type=str, default=128)
parser.add_argument('--dropout1_rate', type=str, default=0.2)
parser.add_argument('--dropout2_rate', type=str, default=0.3)
parser.add_argument('--kernel_size1', type=str, default=3)
parser.add_argument('--kernel_size2', type=str, default=3)
parser.add_argument('--kernel_size3', type=str, default=3)

if 'ipykernel' in sys.modules:
    
    # if interactive, pass values manually

    work_dir = '/home/jovyan/fsupek_data/users/malvarez/projects/embeddings_mutsig/'
    files_dir = '/data/nn_input/test_patri/'

    training_set = "all_cancertypes_df_train"
    validation_set = "all_cancertypes_df_validate"
    testing_set = "all_cancertypes_df_test"
    all_sets = "all_cancertypes_df_all"

    batch_size = 256
    learning_rate = 0.0008
    patience = 20
    fc1_neurons = 128
    fc2_neurons = 128
    dropout1_rate = 0.2
    dropout2_rate = 0.3
    kernel_size1 = 3
    kernel_size2 = 3
    kernel_size3 = 3
    
    # also create output folders
    
    if not os.path.exists(f'{work_dir}/embeddings/CNN_models/'):
        os.makedirs(f'{work_dir}/embeddings/CNN_models/')

    if not os.path.exists(f'{work_dir}/embeddings/loss_auc_curves/'):
        os.makedirs(f'{work_dir}/embeddings/loss_auc_curves/')
    
    if not os.path.exists(f'{work_dir}/embeddings/saved_embeddings_myCNN/'):
        os.makedirs(f'{work_dir}/embeddings/saved_embeddings_myCNN/')
    
else:
    # otherwise, load arguments
    
    args = parser.parse_args()
    work_dir = args.work_dir
    files_dir = args.files_dir
    training_set = args.training_set
    validation_set = args.validation_set
    testing_set = args.testing_set
    all_sets = args.all_sets
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    patience = int(args.patience)
    fc1_neurons = int(args.fc1_neurons)
    fc2_neurons = int(args.fc2_neurons)
    dropout1_rate = float(args.dropout1_rate)
    dropout2_rate = float(args.dropout2_rate)
    kernel_size1 = int(args.kernel_size1)
    kernel_size2 = int(args.kernel_size2)
    kernel_size3 = int(args.kernel_size3)

    
# input files path
path = f'{work_dir}{files_dir}'

# define your config with dot notation access
config = SimpleNamespace(learning_rate=learning_rate,
                         patience=patience,
                         fc1_neurons=fc1_neurons,
                         fc2_neurons=fc2_neurons,
                         dropout1_rate=dropout1_rate,
                         dropout2_rate=dropout2_rate,
                         kernel_size1=kernel_size1,
                         kernel_size2=kernel_size2,
                         kernel_size3=kernel_size3)


# +
train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping = load_data(path,
                                                                                                                       training_set, 
                                                                                                                       validation_set, 
                                                                                                                       testing_set, 
                                                                                                                       all_sets,
                                                                                                                       device, 
                                                                                                                       batch_size)

print("loaded data")
print(label_mapping)

# N cancer types
n_ct = len(label_mapping)


# +
model_path = f'{work_dir}/embeddings/CNN_models/best_model_{training_set}_{validation_set}_{testing_set}_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_fc1{fc1_neurons}_fc2{fc2_neurons}_dropout1{dropout1_rate}_dropout2{dropout2_rate}_kernel1{kernel_size1}_kernel2{kernel_size2}_kernel3{kernel_size3}.pth'

model = CNN_DNAClassifier(config, n_ct).to(device)

# Train the model normally (NO WANDB SWEEPS)
model = train_model(model_path, work_dir, config, n_ct, train_loader, val_loader, class_weights_tensor, device,
                    training_set, 
                    validation_set, 
                    testing_set, 
                    all_sets,
                    batch_size,
                    learning_rate,
                    patience,
                    fc1_neurons,
                    fc2_neurons,
                    dropout1_rate,
                    dropout2_rate,
                    kernel_size1,
                    kernel_size2,
                    kernel_size3)

print("Finished training")

model = CNN_DNAClassifier(config, n_ct).to(device)

# +
print("LOADING MODEL")

model.load_state_dict(torch.load(model_path))


# +
# Save embeddings

save_all_embeddings_probs(model, test_labels, test_sequences_og, label_mapping, f'{work_dir}/embeddings/saved_embeddings_myCNN/test_embeddings_probs_conv1_{training_set}_{validation_set}_{testing_set}_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_fc1{fc1_neurons}_fc2{fc2_neurons}_dropout1{dropout1_rate}_dropout2{dropout2_rate}_kernel1{kernel_size1}_kernel2{kernel_size2}_kernel3{kernel_size3}.csv')
# -


