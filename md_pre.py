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
import torch.nn as nn
import numpy as np
from vector_comparison_net import VectorComparisonNet

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
plainB_pie_literal_sample = torch.load(plainB_pie_literal_sample_path)
plainB_pie_idiom_train = torch.load(plainB_pie_idiom_train_path)
plainB_pie_idiom_sample = torch.load(plainB_pie_idiom_sample_path)

baseline_pie_literal_train = torch.load(baseline_pie_literal_train_path)
baseline_pie_literal_sample = torch.load(baseline_pie_literal_sample_path)
baseline_pie_idiom_train = torch.load(baseline_pie_idiom_train_path)
baseline_pie_idiom_sample = torch.load(baseline_pie_idiom_sample_path)

disc_pie_literal_train = torch.load(disc_pie_literal_train_path)
disc_pie_literal_sample = torch.load(disc_pie_literal_sample_path)
disc_pie_idiom_train = torch.load(disc_pie_idiom_train_path)
disc_pie_idiom_sample = torch.load(disc_pie_idiom_sample_path)

vit_pie_literal_train = torch.load(vit_pie_literal_train_path)
vit_pie_literal_sample = torch.load(vit_pie_literal_sample_path)
vit_pie_idiom_train = torch.load(vit_pie_idiom_train_path)
vit_pie_idiom_sample = torch.load(vit_pie_idiom_sample_path)

image_idiom_train = torch.load(image_idiom_train_path)
image_literal_train = torch.load(image_literal_train_path)

print("All files loaded successfully.")



print("All embeddings loaded successfully!")

plainB_pie_literal_trainDF = pd.DataFrame(list(plainB_pie_literal_train.items()), columns=["compound", "embedding"])
plainB_pie_literal_sampleDF = pd.DataFrame(list(plainB_pie_literal_sample.items()), columns=["compound", "embedding"])
plainB_pie_idiom_trainDF = pd.DataFrame(list(plainB_pie_idiom_train.items()), columns=["compound", "embedding"])
plainB_pie_idiom_sampleDF = pd.DataFrame(list(plainB_pie_idiom_sample.items()), columns=["compound", "embedding"])

baseline_pie_literal_trainDF = pd.DataFrame(list(baseline_pie_literal_train.items()), columns=["compound", "embedding"])
baseline_pie_literal_sampleDF = pd.DataFrame(list(baseline_pie_literal_sample.items()), columns=["compound", "embedding"])
baseline_pie_idiom_trainDF = pd.DataFrame(list(baseline_pie_idiom_train.items()), columns=["compound", "embedding"])
baseline_pie_idiom_sampleDF = pd.DataFrame(list(baseline_pie_idiom_sample.items()), columns=["compound", "embedding"])

disc_pie_literal_trainDF = pd.DataFrame(list(disc_pie_literal_train.items()), columns=["compound", "embedding"])
disc_pie_literal_sampleDF = pd.DataFrame(list(disc_pie_literal_sample.items()), columns=["compound", "embedding"])
disc_pie_idiom_trainDF = pd.DataFrame(list(disc_pie_idiom_train.items()), columns=["compound", "embedding"])
disc_pie_idiom_sampleDF = pd.DataFrame(list(disc_pie_idiom_sample.items()), columns=["compound", "embedding"])

vit_pie_literal_trainDF = pd.DataFrame(list(vit_pie_literal_train.items()), columns=["compound", "embedding"])
vit_pie_literal_sampleDF = pd.DataFrame(list(vit_pie_literal_sample.items()), columns=["compound", "embedding"])
vit_pie_idiom_trainDF = pd.DataFrame(list(vit_pie_idiom_train.items()), columns=["compound", "embedding"])
vit_pie_idiom_sampleDF = pd.DataFrame(list(vit_pie_idiom_sample.items()), columns=["compound", "embedding"])

#---------------image embeddings--------------------------------
image_idiom_train = torch.load(image_idiom_train_path)
image_literal_train = torch.load(image_literal_train_path)
image_idiom_sample = torch.load(image_idiom_sample_path)
image_literal_sample = torch.load(image_literal_sample_path)

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
#---------------image embeddings--------------------------------

#---------------model init--------------------------------
embedding_dim = 768
hidden_dim = 128
model = VectorComparisonNet(embedding_dim, hidden_dim)
#---------------model init--------------------------------

# merge together

#---------------Baseline--------------------------------
Baseline_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    baseline_pie_idiom_trainDF.rename(columns={'embedding': 'baseline_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

Baseline_idioms_train['image_embedding'] = Baseline_idioms_train['image_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)
Baseline_idioms_train['baseline_embedding'] = Baseline_idioms_train['baseline_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

Baseline_idioms_train_result = []

for compound, entries in Baseline_idioms_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['baseline_embedding']
        print(f"image_embedding dimensions: {image_embedding.dim()}")
        print(f"Text embedding dimensions: {text_embedding.dim()}")
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    Baseline_idioms_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })


Baseline_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    baseline_pie_literal_trainDF.rename(columns={'embedding': 'baseline_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)


Baseline_literal_train['image_embedding'] = Baseline_literal_train['baseline_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)
Baseline_literal_train['baseline_embedding'] = Baseline_literal_train['baseline_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

Baseline_literal_train_result = []

for compound, entries in Baseline_literal_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['baseline_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    Baseline_literal_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })
#---------------Baseline--------------------------------
    
#----------plainB--------------------------------
plainB_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_idiom_trainDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_idioms_train['image_embedding'] = plainB_idioms_train['image_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)
plainB_idioms_train['plainB_embedding'] = plainB_idioms_train['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_idioms_train_result = []

for compound, entries in plainB_idioms_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['plainB_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    plainB_idioms_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })
    
    
plainB_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_literal_trainDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_literal_train['image_embedding'] = plainB_literal_train['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_literal_train['plainB_embedding'] = plainB_literal_train['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_literal_train_result = []

for compound, entries in plainB_literal_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['plainB_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    plainB_literal_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })
    
plainB_literal_sample = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_literal_sampleDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_literal_sample['image_embedding'] = plainB_literal_sample['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_literal_sample['plainB_embedding'] = plainB_literal_sample['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_literal_sample_result = []

for compound, entries in plainB_literal_sample.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['plainB_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    plainB_literal_sample_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })

plainB_idiom_sample = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    plainB_pie_idiom_sampleDF.rename(columns={'embedding': 'plainB_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

plainB_idiom_sample['image_embedding'] = plainB_idiom_sample['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_idiom_sample['plainB_embedding'] = plainB_idiom_sample['plainB_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

plainB_idiom_sample_result = []

for compound, entries in plainB_idiom_sample.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['plainB_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    plainB_idiom_sample_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })
    
#--------------------disc---------------------------------
disc_idioms_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    disc_pie_idiom_trainDF.rename(columns={'embedding': 'disc_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

disc_idioms_train['image_embedding'] = disc_idioms_train['image_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

disc_idioms_train['disc_embedding'] = disc_idioms_train['disc_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

disc_idioms_train['disc_embedding'] = disc_idioms_train['disc_embedding'].apply(
    lambda x: torch.from_numpy(x).to(torch.float32) if isinstance(x, np.ndarray) else x
)

disc_idioms_train_result = []

for compound, entries in disc_idioms_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['disc_embedding']
        print(f"text_embedding type: {type(text_embedding)}, dtype: {text_embedding.dtype}")
        print(f"image_embedding type: {type(image_embedding)}, dtype: {image_embedding.dtype}")
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    disc_idioms_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })







disc_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    disc_pie_literal_trainDF.rename(columns={'embedding': 'disc_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

disc_literal_train['image_embedding'] = disc_literal_train['disc_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

disc_literal_train['disc_embedding'] = disc_literal_train['disc_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

disc_literal_train['disc_embedding'] = disc_literal_train['disc_embedding'].apply(
    lambda x: torch.from_numpy(x).to(torch.float32) if isinstance(x, np.ndarray) else x
)

disc_literal_train_result = []

for compound, entries in disc_literal_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['disc_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    disc_literal_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })
    
#-------------------------------vit----------------------
vit_literal_train = pd.merge(
    image_embedding_literal_train.rename(columns={'embedding': 'image_embedding'}),  
    vit_pie_literal_trainDF.rename(columns={'embedding': 'vit_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

vit_literal_train['image_embedding'] = vit_literal_train['vit_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

vit_literal_train['vit_embedding'] = vit_literal_train['vit_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

vit_literal_train['vit_embedding'] = vit_literal_train['vit_embedding'].apply(
    lambda x: torch.from_numpy(x).to(torch.float32) if isinstance(x, np.ndarray) else x
)

vit_literal_train_result = []

for compound, entries in vit_literal_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['vit_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    vit_literal_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })


vit_idiom_train = pd.merge(
    image_embedding_idiom_train.rename(columns={'embedding': 'image_embedding'}),  
    vit_pie_idiom_trainDF.rename(columns={'embedding': 'vit_embedding'}),
    on='compound',                                               
    how='inner'                                                  
)

vit_idiom_train['image_embedding'] = vit_idiom_train['vit_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

vit_idiom_train['vit_embedding'] = vit_idiom_train['vit_embedding'].apply(
    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, list) else x
)

vit_idiom_train['vit_embedding'] = vit_idiom_train['vit_embedding'].apply(
    lambda x: torch.from_numpy(x).to(torch.float32) if isinstance(x, np.ndarray) else x
)

vit_idiom_train_result = []

for compound, entries in vit_idiom_train.groupby('compound'):
    scores = []
    for _, row in entries.iterrows():
        image_id = row['image_name']
        image_embedding = row['image_embedding']
        text_embedding = row['vit_embedding']
        
        assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
        assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, "Invalid baseline_embedding type"

        similarity_score = model(image_embedding, text_embedding).item()
        scores.append((image_id, similarity_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_image_ids = [f"'{item[0]}'" for item in scores]  

    vit_idiom_train_result.append({
        'compound': compound,
        'expected_order': f"[{', '.join(ranked_image_ids)}]"
    })



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





    



# 创建 submission 文件夹
submission_dir = "submission"
os.makedirs(submission_dir, exist_ok=True)  # 如果文件夹不存在，则创建

# 创建 submission 文件夹
submission_dir = "submission"
os.makedirs(submission_dir, exist_ok=True)  # 如果文件夹不存在，则创建

# 定义保存结果的函数
def save_results_to_tsv(results, name):
    """
    将结果保存为 TSV 文件
    :param results: 要保存的结果列表
    :param name: 文件名（不带扩展名）
    """
    results_df = pd.DataFrame(results)
    output_file = os.path.join(submission_dir, f"{name}.tsv")
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved results to {output_file}")


# 保存结果到不同的 TSV 文件
save_results_to_tsv(Baseline_literal_train_result, "Baseline_literal_train_result")
save_results_to_tsv(Baseline_idioms_train_result, "Baseline_idiom_train_result")
save_results_to_tsv(plainB_literal_train_result, "plainB_literal_train_result")
save_results_to_tsv(plainB_idioms_train_result, "plainB_idiom_train_result")
save_results_to_tsv(disc_literal_train_result, "disc_literal_train_result")
save_results_to_tsv(disc_idioms_train_result, "disc_idioms_train_result")
save_results_to_tsv(vit_literal_train_result, "vit_literal_train_result")
save_results_to_tsv(vit_idiom_train_result, "vit_idiom_train_result")

"""
save_results_to_tsv(Baseline_literal_train_result, "Baseline_literal_train_result")
save_results_to_tsv(plainB_literal_train_result, "Baseline_idioms_train_result")
save_results_to_tsv(Baseline_literal_train_result, "plainB_literal_train_result")
save_results_to_tsv(plainB_literal_train_result, "plainB_idioms_train_result")
save_results_to_tsv(Baseline_literal_train_result, "disc_literal_train_result")
save_results_to_tsv(plainB_literal_train_result, "disc_idioms_train_result")
save_results_to_tsv(plainB_literal_train_result, "vit_literal_train_result")
save_results_to_tsv(plainB_literal_train_result, "vit_idioms_train_result")
"""