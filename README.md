# Mutational Signatures Extraction with Neural Network Embeddings

# Inspired by use of embeddings in Yaacov et al. 2024 doi.org/10.1016/j.xcrm.2024.101608

## Network Architecture
### Layer Structure
- **Convolutional Layers**
  - Conv1
  - Conv2

- **Fully Connected Layers**
  - FC1
  - FC2 --> Embeddings (signature exposures) are the output of FC2 (after ReLU and dropout)
  - Output: matches number of classes

### Components
- **Activation Functions**
  - ReLU applied to all convolutional and fully connected layers
  - Softmax applied to Output

- **Optimization**
  - Cross-entropy to get loss
  - Adam optimizer

- **Regularization**
  - Dropout in FC1
  - Dropout in FC2
  - Early stopping patience
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
- Test different k-mers (input sequence length) and FC2 sizes (i.e. K signatures)
- Map signatures to motifs by analyzing embedding changes

## Installation
```bash
# Using Singularity
singularity pull docker://mmalvareza/embeddings_mutsig:latest ./containers

# Using Docker
docker pull mmalvareza/embeddings_mutsig:latest ./containers
```

- The Nextflow version tested is 24.10.0.5928
