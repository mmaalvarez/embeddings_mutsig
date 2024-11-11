# conf_mat.py
from model import one_hot_encode_dna, CNN_DNAClassifier, encode_labels, encode_labels_return, get_predicted_labels, maplabel
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




def plot_norm_conf_mat(test_loader, test_labels_og, device, model):
  test_encoded_sequences, test_labels = test_loader.dataset[:]
  test_encoded_sequences, test_labels = test_encoded_sequences, test_labels
  
  ##################### FOR TEST SET
  # Get predicted labels and true labels
  
  predicted_labels_test = get_predicted_labels(model, test_encoded_sequences)
  
  true_labels_test = torch.tensor(test_labels)  # Ensure true labels are in tensor format

  # Create the confusion matrix
  conf_matrix_test = confusion_matrix(true_labels_test.cpu(), predicted_labels_test.cpu())
  ###NORMALIZED VALUES
  # Normalize the confusion matrix (row-wise normalization)

  conf_matrix_normalized_test = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]

  #Save the encoder
  _, encoder = encode_labels_return(test_labels_og)
  
  #Original label names
  original_class_test= maplabel(test_labels, encoder)
  original_class_names_test=original_class_test.values()
  
  # Plot the normalized confusion matrix using seaborn with original labels
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix_normalized_test, annot=True, fmt='.2f', cmap='Blues', 
              xticklabels=original_class_names_test, yticklabels=original_class_names_test)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Normalized Confusion Matrix TEST set')
  plt.subplots_adjust(left=0.25, bottom=0.3) 
  
  plt.savefig("/g/strcombio/fsupek_home/pferrer/indelDNN/embeddings/confusion_mat/test_confusion_mat_val_t12r.png")
  print("saved conf matrix plot")
