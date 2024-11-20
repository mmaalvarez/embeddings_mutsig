# Mutational Signatures Extraction with Neural Network Embeddings

## Network Architecture
### Layer Structure
- **Convolutional Layers**
  - Conv1: 32 output channels
  - Conv2: 64 output channels

- **Fully Connected Layers**
  - FC1: 128 neurons
  - FC2: 128 neurons 
  - Output: matches number of classes

### Components
- **Activation Functions**
  - ReLU after convolutions and FC1
  - Softmax for final output

- **Regularization**
  - Dropout1: 0.2 rate
  - Dropout2: 0.3 rate

### Training Parameters
- Batch size: 256
- Learning rate: 0.0008
- Early stopping patience: 20
- Validation and test sets implemented

## Data Processing

### Input Format
- SNVs and flanking sequences per sample
- Single k-mer size per experiment
- .csv (comma-delimited) files; "label"- and "seq"-named columns are mandatory, and other columns are ignored ; example for 2 SNVs in a 15-mer context (i.e. central nucleotide is the mutated locus):
```
label,seq,SNV
prost_radiotherapy,TAGACGATACTGCAT,C>T
adeno_metastasis,CGAGTGAGGCATAAG,A>G
```

### Encoding
- One-hot encoding for sequences
- Example for nucleotides: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]

### Training Strategy
- Randomized batch composition across samples
- Labels format: ['MSH6ko', 'MSH2ko', ...] → [0, 1, ...]
- Loss: Cross-entropy with softmax activation

## Signature Analysis
- FC1 layer (128 neurons) represents signature exposures
- Test different FC1 sizes: [4,8,16,32,64,128]
- Map signatures to motifs by analyzing embedding changes

## Installation
```bash
# Using Singularity
singularity pull docker://mmalvareza/embeddings_mutsig:latest ./containers

# Using Docker
docker pull mmalvareza/embeddings_mutsig:latest ./containers
```

- The Nextflow version tested is 24.10.0.5928
