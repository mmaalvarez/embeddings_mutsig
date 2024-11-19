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

# data.py
from model import one_hot_encode_dna, encode_labels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import seaborn as sns


def load_data(path, training_set, validation_set, testing_set, all_sets, device, batch_size):

  # Read data
  train_df = pd.read_csv(path + training_set)

  #Train
  train_sequences = train_df['seq'].tolist()
  train_labels = train_df['label'].tolist()
  
  #Validate
  val_df = pd.read_csv(path + validation_set)
  val_sequences = val_df['seq'].tolist()
  val_labels = val_df['label'].tolist()
  val_labels_og = val_df['label'].tolist()
  
  #Test
  test_df = pd.read_csv(path + testing_set)
  test_sequences = test_df['seq'].tolist()
  test_sequences_og = test_df['seq'].tolist()
  test_labels = test_df['label'].tolist()
  test_labels_og = test_df['label'].tolist()
  
  #All
  all_df = pd.read_csv(path + all_sets)
  all_sequences = all_df['seq'].tolist()
  all_labels = all_df['label'].tolist()
  
  train_labels_series = pd.Series(train_labels)
  label_mapping = {label: idx for idx, label in enumerate(train_labels_series.unique())}
  
  # Encode labels for all data splits
  train_labels = torch.tensor([label_mapping[label] for label in train_labels], dtype=torch.long).to(device)
  val_labels = torch.tensor([label_mapping[label] for label in val_labels], dtype=torch.long).to(device)
  test_labels = torch.tensor([label_mapping[label] for label in test_labels], dtype=torch.long).to(device)

  # One-hot encode the sequences
  train_encoded_sequences = one_hot_encode_dna(train_sequences).to(device)  # Move sequences to GPU
  val_encoded_sequences = one_hot_encode_dna(val_sequences).to(device)
  test_encoded_sequences = one_hot_encode_dna(test_sequences).to(device)
  
  '''
  # Encode the labels and move to GPU
  train_labels, _ = encode_labels(train_labels)
  val_labels, _ = encode_labels(val_labels)
  test_labels, _ = encode_labels(test_labels)
  '''
  
  # Calculate class weights
  class_counts = np.bincount(train_labels.cpu().numpy())
  total_count = len(train_labels)
  class_weights = total_count / (len(class_counts) * class_counts)  # Inverse frequency
  class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
  
  #Labels to device
  train_labels = train_labels.to(device)
  val_labels = val_labels.to(device)
  test_labels = test_labels.to(device)
  
  # Custom Dataset class
  class SequenceDataset(Dataset):
      def __init__(self, sequences, labels):
          self.sequences = sequences
          self.labels = labels
      
      def __len__(self):
          return len(self.sequences)
      
      def __getitem__(self, idx):
          sequence = self.sequences[idx]
          label = self.labels[idx]
          return sequence, label
  
  
  # Create train DataLoaders
  train_loader = DataLoader(
      SequenceDataset(train_encoded_sequences, train_labels),
      batch_size=batch_size,
      shuffle=True
  )
  
  
  # Create validation and test DataLoaders (no sampler, but shuffling is off)
  val_loader = DataLoader(
      SequenceDataset(val_encoded_sequences, val_labels),
      batch_size=batch_size,
      shuffle=True
  )
  
  # Create validation and test DataLoaders (no sampler, but shuffling is off)
  test_loader = DataLoader(
      SequenceDataset(test_encoded_sequences, test_labels),
      batch_size=batch_size,
      shuffle=False
  )
  
  return train_loader, val_loader, test_loader, test_labels, test_sequences_og, class_weights_tensor, label_mapping
