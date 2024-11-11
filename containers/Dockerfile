FROM pytorch/pytorch:latest

LABEL author="https://github.com/mmaalvarez" \
      version="v1.0" \
      description="Environment for 1) running DNA sequence classification using PyTorch and related libraries, and 2) extracting embeddings"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    gfortran \
    git \
    wget \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install \
    torch \
    torchvision \
    torchaudio \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    numpy \
    scipy \
    tqdm \
    wandb \
    einops

# Set environment variables
ENV LC_ALL=C
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages

# Set default command
ENTRYPOINT ["python3"]
CMD ["main.py"]
