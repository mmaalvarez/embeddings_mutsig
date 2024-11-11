# Mutational signatures extraction with neural network embeddings

Total layers:
	First conv layer
	Second conv layer
	Two fully connected layers (fc1 and fc2)
	Total: 4 main layers (2 conv + 2 fc)
Neurons per layer:
	Conv1: output channels = 32
	Conv2: output channels = 64
	fc1: config.fc1_neurons = 128
	fc2: config.fc2_neurons = 128
	Final output: matches number of classes
Activation functions:
	ReLU after each convolutional and first fully connected layer
	Softmax for final output (implicit in cross-entropy loss)
Batch size:
	256 (specified in main.py: batch_size=256)
Optimizer:
	Not explicitly shown in the provided code snippets, though it would be defined where the code is truncated
Learning rate:
	0.0008 (from config in main.py: learning_rate=0.0008)
Bias:
	Not explicitly specified in the shown code, but PyTorch layers include bias by default
Dropout:
	two dropout layers:
	dropout1_rate = 0.2
	dropout2_rate = 0.3
Validation:
	Validation is performed (there's a val_loader)
	Early stopping is implemented with patience=20 (from config)
Testing:
	Test set is used (there's a test_loader)

no data augmentation (1000s of SNVs/sample across 100s samples)

per sample, take all SNVs and their +- 1-20 nut. contexts (i.e. 3-mer to 41-mer), and store as sequences, e.g. for sample 1

	snv	pos	...-mers	15-mer	...-mers
	SNV1	chr1:2412434	...	TAGACGA(C>T)ACTGCAT	...
	SNV2	chr2:42342344	...	CGAGTGA(A>G)GCATAAG	...
	...

NOTE: in a single NN experiment only 1 type of kmer will be used, i.e. not mixing kmers

- this is to identify the optimal SNV context size


and then for the NN these sequences are to be converted to 1-hot encoding, e.g. if 'T', [A,C,G,T] -> [0,0,0,1]

example input layer for 15-mer (7 nt on each side of the SNV)

SNV1 
----
	[0,0,0,1]
	[1,0,0,0]
	[0,0,1,0]
	[1,0,0,0]
	[0,1,0,0]
	[0,0,1,0]
	[1,0,0,0]
	[0,0,0,1], i.e. the mutated allele (T)
	[1,0,0,0]
	[0,1,0,0]
	[0,0,0,1]
	[0,0,1,0]
	[0,1,0,0]
	[1,0,0,0]
	[0,0,0,1]

so all SNVs of all samples are fed 1 by 1 (in batches) accompanied by the given sample's label, so e.g. a MSH2ko sample:

	['MSH6ko', 'MSH2ko', ...] -> [0, 1, ...]

and this is what the NN has to predict, so use softmax as final act. funct. to get e.g. [0.1, 0.8, ...] + cross entropy for loss assessment compared to the actual label [0, 1, ...] --> -log(0.8)=cost

NOTE: don't submit all snvs from a sample, then all snvs by another sample, i.e. submit e.g. 1 snv from 1 sample, then 1 snv from a different sample, etc, randomized


so the first fully connected layer (fc1), after the convolution layers, is the "signature exposures", and the n neurons == K

- Originally used 128 neurons in the fc1, that would be K=128 signatures, maybe too much, can try diff. values

-- by convention the n of neurons should vary by powers of 2, so can try fc1=[4,8,16,32,64,128], and the convolutional and fc2 layers should be powers (larger or smaller) thereof

- there is another fc layer afterwards, and then softmax

- then, for a given K and kmer, I have to map each signature (neuron) to a main k-nucleotide motif

-- do this by feeding 1 kmer sequence, e.g. k As, and then change each nucleotide 1 by 1 to each of the other 3 bases, and see how the embeddings (i.e. exposures) change in each neuron ("signature")
