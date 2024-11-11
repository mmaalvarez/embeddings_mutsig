# embeddings.py
from model import one_hot_encode_dna, CNN_DNAClassifier, encode_labels, calculate_accuracy, train_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import numpy
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import torch
import einops
import torch.nn as nn
from typing import Callable
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Rearrange
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GET EMBEDDINGS WITH PROBABILITIES
def get_embeddings_with_probs(model, sequences, batch_size=32):
    model.eval()
    all_embeddings = []
    all_probs = []
    all_predicted_labels = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            print("BATCH:")
            print(str(i))
            batch_sequences = sequences[i:i + batch_size]
            outputs, embeddings = model(one_hot_encode_dna(batch_sequences).to(device))
            probs = torch.softmax(outputs, dim=1)  # Probabilities
            _, predicted_labels = torch.max(outputs, 1)

            all_embeddings.append(embeddings)
            all_probs.append(probs)
            all_predicted_labels.append(predicted_labels)

    # Concatenate all results
    embeddings = torch.cat(all_embeddings)
    probs = torch.cat(all_probs)
    predicted_labels = torch.cat(all_predicted_labels)

    return embeddings, probs, predicted_labels  



def save_all_embeddings_probs(model, labels, sequences, label_mapping, output_file):
  
    embeddings, probabilities, predicted_labels = get_embeddings_with_probs(model, sequences)

    # Print the type of probabilities to check if it's a numpy array
    print(f'Type of probabilities: {type(probabilities)}')

    # Convert probabilities to a NumPy array if it's a tensor
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    # Get max probabilities for each sequence
    
    max_probabilities = numpy.max(probabilities, axis=1)  # Get max prob for each sequence

    # Now save the embeddings, true labels, predicted labels, and max probabilities
    save_embeddings_csv_probs(sequences, labels, predicted_labels, embeddings, max_probabilities, label_mapping, output_file)

    print("SAVE ALL EMBEDDINGS FUNCTION CALLED CSV FUNCTION")


#Save with probabilities
# Ensure save_embeddings_csv accepts max_probabilities as an additional argument
def save_embeddings_csv_probs(sequences, true_encoded_labels, predicted_encoded_labels, embeddings, max_probabilities, label_mapping, output_file):
    # Convert both true and predicted encoded labels back to original string labels
    true_labels = map_encoded_to_original_labels(true_encoded_labels, label_mapping)

    # Prepare the data for the DataFrame
    print("SHAPE")
    print(embeddings.shape)
    print("error")
    embeddings_df = pd.DataFrame(embeddings.cpu().numpy())  # Assuming embeddings is a tensor
   
    data = {
        'seq': sequences,
        'true_label': true_labels,
        'predicted_label': map_encoded_to_original_labels(predicted_encoded_labels, label_mapping),
        'max_probability': max_probabilities  # Add max probabilities here
    }
    
    
    # Check lengths before creating the DataFrame
    lengths = {key: len(value) for key, value in data.items()}
    print("Lengths of data components:", lengths)
    
    
    # Ensure all components are the same length
    if len(set(lengths.values())) != 1:
        raise ValueError("All components must have the same length.")
    # Combine into a single DataFrame
    df = pd.DataFrame(data)
    print("created pd dataframe")
    
    df = pd.concat([df, embeddings_df], axis=1)
    print("concatenated embeddings")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(df.head())
    print(f'Embeddings and probabilities saved to {output_file}')


# Function to map the encoded labels back to original labels using the label_mapping
def map_encoded_to_original_labels(encoded_labels, label_mapping):
    # Convert to numpy array if it's a list
    if isinstance(encoded_labels, list):
        encoded_labels = np.array(encoded_labels)

    if isinstance(encoded_labels, torch.Tensor):
        encoded_labels = encoded_labels.cpu().numpy()  # Convert tensor to numpy

    # Create a reverse mapping from encoded values to original labels
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}

    # Map encoded labels back to original labels
    original_labels = [reverse_mapping[idx] for idx in encoded_labels]

    print("MAPPED TO ORIGINAL LABELS")
    return original_labels
