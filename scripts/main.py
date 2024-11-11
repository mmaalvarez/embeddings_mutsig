# main.py
from model import one_hot_encode_dna, CNN_DNAClassifier, encode_labels, calculate_accuracy, train_model
from data import load_data
from conf_mat import plot_norm_conf_mat
from embeddings import save_all_embeddings_probs
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import wandb
from types import SimpleNamespace

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Path with files
path = "/g/strcombio/fsupek_home/pferrer/indelDNN/embeddings/saved_seqs/saved_seqs_filt_insALT_test11/"


train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping = load_data(path, device, batch_size=256)
print("loaded data")

print(label_mapping)

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

model_path='/g/strcombio/fsupek_home/pferrer/indelDNN/embeddings/CNN_models/best_model_test11.pth'


model = CNN_DNAClassifier(config).to(device)


# Train the model normally (NO WANDB SWEEPS)
model = train_model(model_path, config, train_loader, val_loader, class_weights_tensor, device, wandb=False)

print("Finished training")

model = CNN_DNAClassifier(config).to(device)


print("LOADING MODEL")
model.load_state_dict(torch.load(model_path))

#plot_norm_conf_mat(test_loader,test_labels_og, device, model)

#Save embeddings
save_all_embeddings_probs(model, test_labels, test_sequences_og, label_mapping, "/g/strcombio/fsupek_home/pferrer/indelDNN/embeddings/saved_embeddings_myCNN/test_embeddings_probs_t11_conv1.csv")

