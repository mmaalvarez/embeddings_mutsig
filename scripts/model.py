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

        # ensure kmer size and convolutional layers are odd values
        assert config.kmer % 2 == 1, "ERROR: k-mer size is an even number, it must be odd"
        assert config.kernel_size_conv1 % 2 == 1, "ERROR: kernel size for convolutional layer 1 is an even integer, must be odd"
        assert config.kernel_size_conv2 % 2 == 1, "ERROR: kernel size for convolutional layer 2 is an even integer, must be odd"
        
        ## conv Layers
        # in_channels=4 because there are 4 possible bases (A: [1 0 0 0] ; C: [0 1 0 0] ; G: [0 0 1 0] ; T: [0 0 0 1])
        self.conv1 = nn.Conv1d(in_channels=4, 
                               out_channels=config.out_channels_conv1, 
                               kernel_size=config.kernel_size_conv1, 
                               # default stride is 1
                               stride = 1,
                               # use 'same' padding to maintain input size (==k-mer)
                               padding = int((config.kernel_size_conv1-1)/2))
        
        self.conv2 = nn.Conv1d(in_channels=config.out_channels_conv1,
                               out_channels=config.out_channels_conv2, 
                               kernel_size=config.kernel_size_conv2, 
                               stride = 1,
                               padding = int((config.kernel_size_conv2-1)/2))
        
        # Determine whether to use ceil_mode based on kmer value (i.e. only if kmer==3, to avoid rounding down to 0 after maxpooling)
        ceil_mode = config.kmer == 3
        
        # Maxpool applied after each convolutional layer
        self.pool = nn.MaxPool1d(kernel_size=config.kernel_size_maxpool,
                                 # same stride as the size
                                 stride=config.kernel_size_maxpool,
                                 ceil_mode=ceil_mode)
        
        # Calculate final flattened size after convolutions and pooling
        L_in = config.kmer
        L_conv1 = L_in + 2*int((config.kernel_size_conv1-1)/2) - config.kernel_size_conv1 + 1
        L_pool1 = ((L_conv1 - config.kernel_size_maxpool) // config.kernel_size_maxpool) + 1
        L_conv2 = L_pool1 + 2*int((config.kernel_size_conv2-1)/2) - config.kernel_size_conv2 + 1
        L_pool2 = ((L_conv2 - config.kernel_size_maxpool) // config.kernel_size_maxpool) + 1
        flatten_size = config.out_channels_conv2 * L_pool2

        ## fully connected layers
        self.fc1 = nn.Linear(flatten_size, config.fc1_neurons)
        self.dropout_fc1 = nn.Dropout(config.dropout_fc1)
        self.fc2 = nn.Linear(config.fc1_neurons, config.fc2_neurons)
        self.dropout_fc2 = nn.Dropout(config.dropout_fc2)
        self.fc3 = nn.Linear(config.fc2_neurons, n_ct)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, channels) --> change to (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # first convolution
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # second convolution
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Flatten the output for the fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout_fc1(x)
        
        # embedding
        embedding = self.fc2(x)
        embedding = torch.relu(embedding)
        embedding = self.dropout_fc2(embedding)
        
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


def train_model(best_model_path, work_dir, config, n_ct, train_loader, val_loader, class_weights_tensor, device,
                all_sets,
                batch_size,
                learning_rate,
                patience,
                epochs,
                training_perc, 
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
    for epoch in range(epochs):
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
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {avg_train_accuracy:.4f}, Val Acc: {avg_val_accuracy:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(best_model_path))
    
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
    # Save the plot as an image file
    plt.savefig(f'loss_and_accuracy_curve_{all_sets}_batch_size{batch_size}_learning_rate{learning_rate}_patience{patience}_c1kernel{config.kernel_size_conv1}_c1out{config.out_channels_conv1}_maxpool{config.kernel_size_maxpool}_c2kernel{config.kernel_size_conv2}_c2out{config.out_channels_conv2}_fc1neu{config.fc1_neurons}_fc1dropout{config.dropout_fc1}_fc2neu{config.fc2_neurons}_fc2dropout{config.dropout_fc2}_kmer{config.kmer}_training{training_perc}_validation{validation_perc}_test{test_perc}_subsetting_seed{subsetting_seed}.png')
    #plt.show()

    return model
