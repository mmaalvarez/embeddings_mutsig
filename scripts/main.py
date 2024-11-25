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

# +
# main.py

# load modules
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
print(f'Using {device}\n')


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
parser.add_argument('--kernel_size_conv1', type=str, default=3)
parser.add_argument('--out_channels_conv1', type=str, default=32)
parser.add_argument('--kernel_size_conv2', type=str, default=3)
parser.add_argument('--out_channels_conv2', type=str, default=64)
parser.add_argument('--kernel_size_maxpool', type=str, default=2)
parser.add_argument('--fc1_neurons', type=str, default=256)
parser.add_argument('--dropout_fc1', type=str, default=0.5)
parser.add_argument('--fc2_neurons', type=str, default=16)
parser.add_argument('--dropout_fc2', type=str, default=0.4)    
parser.add_argument('--epochs', type=str, default=500)
parser.add_argument('--kmer', type=str, default=5)
parser.add_argument('--training_perc', type=str, default=70)
parser.add_argument('--validation_perc', type=str, default=15)
parser.add_argument('--test_perc', type=str, default=15)
parser.add_argument('--subsetting_seed', type=str, default=1)

if 'ipykernel' in sys.modules:
    
    # if interactive, pass values manually

    work_dir = '/home/jovyan/fsupek_data/users/malvarez/projects/embeddings_mutsig/'
    files_dir = '/data/nn_input/SNVs__kucab_zou_petljak_hwang/k-'

    training_set = "SNVs__kucab_zou_petljak_hwang_train"
    validation_set = "SNVs__kucab_zou_petljak_hwang_validate"
    testing_set = "SNVs__kucab_zou_petljak_hwang_test"
    all_sets = "SNVs__kucab_zou_petljak_hwang_all"

    batch_size = 256
    learning_rate = 0.0008
    patience = 20
    kernel_size_conv1 = 3
    out_channels_conv1 = 32
    kernel_size_conv2 = 3
    out_channels_conv2 = 64
    kernel_size_maxpool = 2    
    fc1_neurons = 256
    dropout_fc1 = 0.5
    fc2_neurons = 16
    dropout_fc2 = 0.4
    epochs = 500
    kmer = 5
    training_perc = 70
    validation_perc = 15
    test_perc = 15
    subsetting_seed = 1
    
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
    kernel_size_conv1 = int(args.kernel_size_conv1)
    out_channels_conv1 = int(args.out_channels_conv1)
    kernel_size_conv2 = int(args.kernel_size_conv2)
    out_channels_conv2 = int(args.out_channels_conv2)
    kernel_size_maxpool = int(args.kernel_size_maxpool)    
    fc1_neurons = int(args.fc1_neurons)
    dropout_fc1 = float(args.dropout_fc1)
    fc2_neurons = int(args.fc2_neurons)
    dropout_fc2 = float(args.dropout_fc2)
    epochs = int(args.epochs)
    kmer = int(args.kmer)
    training_perc = int(args.training_perc)
    validation_perc = int(args.validation_perc)
    test_perc = int(args.test_perc)
    subsetting_seed = int(args.subsetting_seed)
    
# input files path
path = f'{work_dir}{files_dir}{kmer}'

# define config with dot notation access
config = SimpleNamespace(learning_rate=learning_rate,
                         patience=patience,
                         kernel_size_conv1=kernel_size_conv1,
                         out_channels_conv1=out_channels_conv1,
                         kernel_size_conv2=kernel_size_conv2,
                         out_channels_conv2=out_channels_conv2,
                         kernel_size_maxpool=kernel_size_maxpool,                         
                         fc1_neurons=fc1_neurons,
                         dropout_fc1=dropout_fc1,
                         fc2_neurons=fc2_neurons,
                         dropout_fc2=dropout_fc2,
                         kmer=kmer)


# +
# load data
train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping = load_data(path,
                                                                                                                       training_set, 
                                                                                                                       validation_set, 
                                                                                                                       testing_set, 
                                                                                                                       all_sets,
                                                                                                                       kmer,
                                                                                                                       training_perc, 
                                                                                                                       validation_perc, 
                                                                                                                       test_perc, 
                                                                                                                       subsetting_seed,
                                                                                                                       device, 
                                                                                                                       batch_size)

print("Loaded data")
print(label_mapping)

# N cancer/treatment types
n_ct = len(label_mapping)


# +
# create path to store the best model from training
best_model_path = f'best_model_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_c1kernel{kernel_size_conv1}_c1out{out_channels_conv1}_maxpool{kernel_size_maxpool}_c2kernel{kernel_size_conv2}_c2out{out_channels_conv2}_fc1neu{fc1_neurons}_fc1dropout{dropout_fc1}_fc2neu{fc2_neurons}_fc2dropout{dropout_fc2}_kmer{kmer}_training{training_perc}_validation{validation_perc}_test{test_perc}_subsetting_seed{subsetting_seed}.pth'

# Train the model normally (NO WANDB SWEEPS)
model = train_model(best_model_path, work_dir, config, n_ct, train_loader, val_loader, class_weights_tensor, device,
                    all_sets,
                    batch_size,
                    learning_rate,
                    patience,
                    epochs,
                    training_perc, 
                    validation_perc, 
                    test_perc, 
                    subsetting_seed)
print("Finished training")

# +
# initiate a fresh model...
model = CNN_DNAClassifier(config, n_ct).to(device)

# ...and load in there the previously saved best weights (best_model_path)...
print("Loading model...")
model.load_state_dict(torch.load(best_model_path))

# ...to use it for generating embeddings (and save them)
save_all_embeddings_probs(model, test_labels, test_sequences_og, batch_size, label_mapping, f'test_embeddings_probs_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_c1kernel{kernel_size_conv1}_c1out{out_channels_conv1}_maxpool{kernel_size_maxpool}_c2kernel{kernel_size_conv2}_c2out{out_channels_conv2}_fc1neu{fc1_neurons}_fc1dropout{dropout_fc1}_fc2neu{fc2_neurons}_fc2dropout{dropout_fc2}_kmer{kmer}_training{training_perc}_validation{validation_perc}_test{test_perc}_subsetting_seed{subsetting_seed}.csv')
