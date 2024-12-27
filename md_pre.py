# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:36:30 2024

This script will load text and image embeddings seperatly and merge them together

Your file structure should look like this

project/
│
├── script.py                 # this script
├── baseline_pie_embeddings_literal_train.pt
├── plainBERT_pie_embeddings_literal_sample.pt
├── disc_pie_embeddings_idiom_train.pt
├── Train/                    # save df under the train folder
└── Sample/                   # save df under the train folder

@author: yuyue
"""

import torch
import pandas as pd
import os
from pathlib import Path

# Get the current working path
current_dir = Path.cwd()  

# Build the embedding path
plainB_pie_literal_train_path = current_dir / "plainBERT_pie_embeddings_literal_train.pt"
plainB_pie_literal_sample_path = current_dir / "plainBERT_pie_embeddings_literal_sample.pt"
plainB_pie_idiom_train_path = current_dir / "plainBERT_pie_embeddings_idiom_train.pt"
plainB_pie_idiom_sample_path = current_dir / "plainBERT_pie_embeddings_idiom_sample.pt"

baseline_pie_literal_train_path = current_dir / "baseline_pie_embeddings_literal_train.pt"
baseline_pie_literal_sample_path = current_dir / "baseline_pie_embeddings_literal_sample.pt"
baseline_pie_idiom_train_path = current_dir / "baseline_pie_embeddings_idiom_train.pt"
baseline_pie_idiom_sample_path = current_dir / "baseline_pie_embeddings_idiom_sample.pt"

disc_pie_literal_train_path = current_dir / "disc_pie_embeddings_literal_train.pt"
disc_pie_literal_sample_path = current_dir / "disc_pie_embeddings_literal_sample.pt"
disc_pie_idiom_train_path = current_dir / "disc_pie_embeddings_idiom_train.pt"
disc_pie_idiom_sample_path = current_dir / "disc_pie_embeddings_idiom_sample.pt"

vit_pie_literal_train_path = current_dir / "VIT_pie_embeddings_literal_train.pt"
vit_pie_literal_sample_path = current_dir / "VIT_pie_embeddings_literal_sample.pt"
vit_pie_idiom_train_path = current_dir / "VIT_pie_embeddings_idiom_train.pt"
vit_pie_idiom_sample_path = current_dir / "VIT_pie_embeddings_idiom_sample.pt"

image_literal_train_path = current_dir / "vit_embeddings_train_literal.pt"
image_literal_sample_path = current_dir / "vit_embeddings_sample_literal.pt"
image_idiom_train_path = current_dir / "vit_embeddings_train_idiomatic.pt"
image_idiom_sample_path = current_dir / "vit_embeddings_sample_literal.pt"


# Load files
plainB_pie_literal_train = torch.load(plainB_pie_literal_train_path)
plainB_pie_literal_simple = torch.load(plainB_pie_literal_sample_path)
plainB_pie_idiom_train = torch.load(plainB_pie_idiom_train_path)
plainB_pie_idiom_simple = torch.load(plainB_pie_idiom_sample_path)

baseline_pie_literal_train = torch.load(baseline_pie_literal_train_path)
baseline_pie_literal_simple = torch.load(baseline_pie_literal_sample_path)
baseline_pie_idiom_train = torch.load(baseline_pie_idiom_train_path)
baseline_pie_idiom_simple = torch.load(baseline_pie_idiom_sample_path)

disc_pie_literal_train = torch.load(disc_pie_literal_train_path)
disc_pie_literal_simple = torch.load(disc_pie_literal_sample_path)
disc_pie_idiom_train = torch.load(disc_pie_idiom_train_path)
disc_pie_idiom_simple = torch.load(disc_pie_idiom_sample_path)

vit_pie_literal_train = torch.load(vit_pie_literal_train_path)
vit_pie_literal_simple = torch.load(vit_pie_literal_sample_path)
vit_pie_idiom_train = torch.load(vit_pie_idiom_train_path)
vit_pie_idiom_simple = torch.load(vit_pie_idiom_sample_path)

image_idiom_train = torch.load(image_idiom_train_path)
image_literal_train = torch.load(image_literal_train_path)

print("All files loaded successfully.")



print("All embeddings loaded successfully!")

plainB_pie_literal_trainDF = pd.DataFrame(list(plainB_pie_literal_train.items()), columns=["compound", "embedding"])
plainB_pie_literal_simpleDF = pd.DataFrame(list(plainB_pie_literal_simple.items()), columns=["compound", "embedding"])
plainB_pie_idiom_trainDF = pd.DataFrame(list(plainB_pie_idiom_train.items()), columns=["compound", "embedding"])
plainB_pie_idiom_simpleDF = pd.DataFrame(list(plainB_pie_idiom_simple.items()), columns=["compound", "embedding"])

baseline_pie_literal_trainDF = pd.DataFrame(list(baseline_pie_literal_train.items()), columns=["compound", "embedding"])
baseline_pie_literal_simpleDF = pd.DataFrame(list(baseline_pie_literal_simple.items()), columns=["compound", "embedding"])
baseline_pie_idiom_trainDF = pd.DataFrame(list(baseline_pie_idiom_train.items()), columns=["compound", "embedding"])
baseline_pie_idiom_simpleDF = pd.DataFrame(list(baseline_pie_idiom_simple.items()), columns=["compound", "embedding"])

disc_pie_literal_trainDF = pd.DataFrame(list(disc_pie_literal_train.items()), columns=["compound", "embedding"])
disc_pie_literal_simpleDF = pd.DataFrame(list(disc_pie_literal_simple.items()), columns=["compound", "embedding"])
disc_pie_idiom_trainDF = pd.DataFrame(list(disc_pie_idiom_train.items()), columns=["compound", "embedding"])
disc_pie_idiom_simpleDF = pd.DataFrame(list(disc_pie_idiom_simple.items()), columns=["compound", "embedding"])

vit_pie_literal_trainDF = pd.DataFrame(list(vit_pie_literal_train.items()), columns=["compound", "embedding"])
vit_pie_literal_simpleDF = pd.DataFrame(list(vit_pie_literal_simple.items()), columns=["compound", "embedding"])
vit_pie_idiom_trainDF = pd.DataFrame(list(vit_pie_idiom_train.items()), columns=["compound", "embedding"])
vit_pie_idiom_simpleDF = pd.DataFrame(list(vit_pie_idiom_simple.items()), columns=["compound", "embedding"])

try:
    image_idiom_train = torch.load(image_idiom_train_path)
    print(f"Successfully loaded: {image_idiom_train_path.name}")
except FileNotFoundError:
    print(f"File not found: {image_idiom_train_path}")

try:
    image_literal_train = torch.load(image_literal_train_path)
    print(f"Successfully loaded: {image_literal_train_path.name}")
except FileNotFoundError:
    print(f"File not found: {image_literal_train_path}")

try:
    image_idiom_sample = torch.load(image_idiom_sample_path)
    print(f"Successfully loaded: {image_idiom_sample_path.name}")
except FileNotFoundError:
    print(f"File not found: {image_idiom_sample_path}")

try:
    image_literal_sample = torch.load(image_literal_sample_path)
    print(f"Successfully loaded: {image_literal_sample_path.name}")
except FileNotFoundError:
    print(f"File not found: {image_literal_sample_path}")

image_embedding_idiom_train = []
image_embedding_literal_train = []
image_embedding_idiom_sample = []
image_embedding_literal_sample = []

if 'image_idiom_train' in locals():
    for compound, values in image_idiom_train.items():
        for value in values:
            image_name = value['image_name']
            embedding = value['embedding']
            image_embedding_idiom_train.append({'compound': compound, 'image_name': image_name, 'embedding': embedding})
    image_embedding_idiom_train = pd.DataFrame(image_embedding_idiom_train)
    print("Successfully created image_embedding_idiom_train DataFrame")
else:
    print("image_idiom_train is not loaded or does not exist")


if 'image_literal_train' in locals():
    for compound, values in image_literal_train.items():
        for value in values:
            image_name = value['image_name']
            embedding = value['embedding']
            image_embedding_literal_train.append({'compound': compound, 'image_name': image_name, 'embedding': embedding})
    image_embedding_literal_train = pd.DataFrame(image_embedding_literal_train)
    print("Successfully created image_embedding_literal_train DataFrame")
else:
    print("image_literal_train is not loaded or does not exist")


if 'image_idiom_sample' in locals():
    for compound, values in image_idiom_sample.items():
        for value in values:
            image_name = value['image_name']
            embedding = value['embedding']
            image_embedding_idiom_sample.append({'compound': compound, 'image_name': image_name, 'embedding': embedding})
    image_embedding_idiom_sample = pd.DataFrame(image_embedding_idiom_sample)
    print("Successfully created image_embedding_idiom_sample DataFrame")
else:
    print("image_idiom_sample is not loaded or does not exist")

# 处理 image_literal_sample
if 'image_literal_sample' in locals():
    for compound, values in image_literal_sample.items():
        for value in values:
            image_name = value['image_name']
            embedding = value['embedding']
            image_embedding_literal_sample.append({'compound': compound, 'image_name': image_name, 'embedding': embedding})
    image_embedding_literal_sample = pd.DataFrame(image_embedding_literal_sample)
    print("Successfully created image_embedding_literal_sample DataFrame")
else:
    print("image_literal_sample is not loaded or does not exist")



# merge together

Baseline_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    baseline_pie_idiom_trainDF.rename(columns={'embedding': 'baseline_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_idiom_trainDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

disc_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    disc_pie_idiom_trainDF.rename(columns={'embedding': 'disc_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)


Baseline_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    baseline_pie_literal_trainDF.rename(columns={'embedding': 'baseline_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_literal_trainDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

disc_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    disc_pie_literal_trainDF.rename(columns={'embedding': 'disc_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)


train_dir = current_dir / "Train"
sample_dir = current_dir / "Sample"

train_dir.mkdir(exist_ok=True)
sample_dir.mkdir(exist_ok=True)

for file in current_dir.glob("*.pt"):

    data = torch.load(file)
    
    if "train" in file.name.lower():
        target_path = train_dir / file.name
    elif "sample" in file.name.lower():
        target_path = sample_dir / file.name
    else:
        print(f"Skipping file: {file.name}")
        continue 

    torch.save(data, target_path)
    print(f"Saved {file.name} to {target_path}")

print("All files have been processed.")