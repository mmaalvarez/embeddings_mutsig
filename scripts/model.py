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

# model.py
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


# Function to one-hot encode DNA sequences
def one_hot_encode_dna(sequences):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = []
        for nucleotide in seq:
            one_hot = [0, 0, 0, 0]
            one_hot[mapping[nucleotide]] = 1
            encoded_seq.append(one_hot)
        encoded_sequences.append(encoded_seq)
    return torch.tensor(encoded_sequences, dtype=torch.float32)


class CNN_DNAClassifier(nn.Module):

    def __init__(self, config, n_ct): # n_ct: Number of cancer types, e.g. int(10)

        super(CNN_DNAClassifier, self).__init__()

        # Conv Layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=config.kernel_size1, padding=int((config.kernel_size1-1)/2))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=config.kernel_size2, padding=int((config.kernel_size2-1)/2))
        
        # Use config values for fully connected layers
        self.fc1 = nn.Linear(64 * 10, config.fc1_neurons)
        self.dropout1 = nn.Dropout(config.dropout1_rate)
        self.fc2 = nn.Linear(config.fc1_neurons, config.fc2_neurons)
        self.dropout2 = nn.Dropout(config.dropout2_rate)
        self.fc3 = nn.Linear(config.fc2_neurons, n_ct)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        conv1_out = torch.relu(self.conv1(x))
        # Apply global average pooling along the sequence length dimension
        conv1_out_pooled = torch.mean(conv1_out, dim=2) 

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        embedding = torch.relu(self.fc2(x))
        embedding = self.dropout2(embedding)
        x = self.fc3(embedding)
        
        return x, embedding


# Initialize the label encoder for cancer types
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)  # Encode "colon", "ovarian", etc. as integers
    return torch.tensor(encoded_labels), encoder.classes_


# Get the predicted class (index of the max log-probability)
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def train_model(model_path, work_dir, config, n_ct, train_loader, val_loader, class_weights_tensor, device,
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
                kernel_size3,
                train_perc, 
                validation_perc, 
                test_perc, 
                subsetting_seed):
    print(config)
    # Initialize the model, loss function, and optimizer
    model = CNN_DNAClassifier(config, n_ct).to(device)  # Move model to GPU
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(500):
        model.train()  # Set the model to training mode
        batch_train_losses = []
        batch_train_accuracies = []

        # Iterate over batches in the training set
        for batch in train_loader:
            train_encoded_sequences, train_labels = batch
            train_encoded_sequences, train_labels = train_encoded_sequences.to(device), train_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(train_encoded_sequences)
            train_loss = criterion(outputs, train_labels)
            batch_train_losses.append(train_loss.item())

            # Calculate training accuracy for the batch
            train_accuracy = calculate_accuracy(outputs, train_labels)
            batch_train_accuracies.append(train_accuracy)

            # Backpropagation and optimizer step
            train_loss.backward()
            optimizer.step()

        # Average training loss and accuracy across batches
        avg_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        avg_train_accuracy = sum(batch_train_accuracies) / len(batch_train_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation step
        model.eval()  # Set the model to evaluation mode
        batch_val_losses = []
        batch_val_accuracies = []

        with torch.no_grad():
            for batch in val_loader:
                val_encoded_sequences, val_labels = batch
                val_encoded_sequences, val_labels = val_encoded_sequences.to(device), val_labels.to(device)          
                
                val_outputs, _ = model(val_encoded_sequences)
                val_loss = criterion(val_outputs, val_labels)
                batch_val_losses.append(val_loss.item())

                # Calculate validation accuracy for the batch
                val_accuracy = calculate_accuracy(val_outputs, val_labels)
                batch_val_accuracies.append(val_accuracy)

        # Average validation loss and accuracy across batches
        avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        avg_val_accuracy = sum(batch_val_accuracies) / len(batch_val_accuracies)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # Print train and validation loss/accuracy for the current epoch
        print(f'Epoch [{epoch + 1}/500], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {avg_train_accuracy:.4f}, Val Acc: {avg_val_accuracy:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(model_path))
    
    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'loss_and_accuracy_curve_2conv_{training_set}_{validation_set}_{testing_set}_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_fc1{fc1_neurons}_fc2{fc2_neurons}_dropout1{dropout1_rate}_dropout2{dropout2_rate}_kernel1{kernel_size1}_kernel2{kernel_size2}_kernel3{kernel_size3}_training{training_perc}_validation{validation_perc}_test{test_perc}_subsetting_seed{subsetting_seed}.png')  # Save the plot as an image file
    #plt.show()

    return model
