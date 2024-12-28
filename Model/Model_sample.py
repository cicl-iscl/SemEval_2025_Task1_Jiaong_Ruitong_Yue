# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:30:16 2024

@author: yuyue
"""
import sys
import torch
import pandas as pd
import os
from pathlib import Path
import torch.nn as nn
import numpy as np
from vector_comparison_net import VectorComparisonNet
from helper_functions import *

# Get the current working path
current_dir = Path.cwd()  

plainB_pie_literal_sample_path = current_dir / "plainBERT_pie_embeddings_literal_sample.pt"
plainB_pie_literal_sample = torch.load(plainB_pie_literal_sample_path)

plainB_pie_idiom_sample_path = current_dir / "plainBERT_pie_embeddings_idiom_sample.pt"
plainB_pie_idiom_sample = torch.load(plainB_pie_idiom_sample_path)

baseline_pie_literal_sample_path = current_dir / "baseline_pie_embeddings_literal_sample.pt"
baseline_pie_literal_sample = torch.load(baseline_pie_literal_sample_path)
baseline_pie_idiom_sample_path = current_dir / "baseline_pie_embeddings_idiom_sample.pt"
baseline_pie_idiom_sample = torch.load(baseline_pie_idiom_sample_path)

disc_pie_literal_sample_path = current_dir / "disc_pie_embeddings_literal_sample.pt"
disc_pie_literal_sample = torch.load(disc_pie_literal_sample_path)
disc_pie_idiom_sample_path = current_dir / "disc_pie_embeddings_idiom_sample.pt"
disc_pie_idiom_sample = torch.load(disc_pie_idiom_sample_path)

vit_pie_literal_sample_path = current_dir / "VIT_pie_embeddings_literal_sample.pt"
vit_pie_literal_sample = torch.load(vit_pie_literal_sample_path)
vit_pie_idiom_sample_path = current_dir / "VIT_pie_embeddings_idiom_sample.pt"
vit_pie_idiom_sample = torch.load(vit_pie_idiom_sample_path)

image_literal_sample_path = current_dir / "vit_embeddings_sample_literal.pt"
image_idiom_sample_path = current_dir / "vit_embeddings_sample_literal.pt"

print("All files loaded successfully.")

plainB_pie_literal_sampleDF = pd.DataFrame(list(plainB_pie_literal_sample.items()), columns=["compound", "embedding"])
plainB_pie_idiom_sampleDF = pd.DataFrame(list(plainB_pie_idiom_sample.items()), columns=["compound", "embedding"])

baseline_pie_literal_sampleDF = pd.DataFrame(list(baseline_pie_literal_sample.items()), columns=["compound", "embedding"])
baseline_pie_idiom_sampleDF = pd.DataFrame(list(baseline_pie_idiom_sample.items()), columns=["compound", "embedding"])

disc_pie_literal_sampleDF = pd.DataFrame(list(disc_pie_literal_sample.items()), columns=["compound", "embedding"])
disc_pie_idiom_sampleDF = pd.DataFrame(list(disc_pie_idiom_sample.items()), columns=["compound", "embedding"])

vit_pie_literal_sampleDF = pd.DataFrame(list(vit_pie_literal_sample.items()), columns=["compound", "embedding"])
vit_pie_idiom_sampleDF = pd.DataFrame(list(vit_pie_idiom_sample.items()), columns=["compound", "embedding"])

image_idiom_sample = torch.load(image_idiom_sample_path)
image_literal_sample = torch.load(image_literal_sample_path)

image_embedding_idiom_sample = []
image_embedding_literal_sample = []

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
    
#---------------model init--------------------------------
embedding_dim = 768
hidden_dim = 128
model = VectorComparisonNet(embedding_dim, hidden_dim)
#---------------model init--------------------------------

#----------plainB sample--------------------------------
plainB_idiom_sample = preprocess_and_merge(
    image_embedding_idiom_sample,
    plainB_pie_idiom_sampleDF,
    'embedding'
)

plainB_idiom_sample_result = calculate_scores(
    plainB_idiom_sample,
    'embedding',
    model
)

plainB_literal_sample = preprocess_and_merge(
    image_embedding_literal_sample,
    plainB_pie_literal_sampleDF,
    'embedding'
)

plainB_literal_sample_result = calculate_scores(
    plainB_literal_sample,
    'embedding',
    model
)

#----------Baseline sample--------------------------------
Baseline_idioms_sample = preprocess_and_merge(
    image_embedding_idiom_sample,
    baseline_pie_idiom_sampleDF,
    'embedding'
)

Baseline_idioms_sample_result = calculate_scores(
    Baseline_idioms_sample,
    'embedding',
    model
)

Baseline_literal_sample = preprocess_and_merge(
    image_embedding_literal_sample,
    baseline_pie_literal_sampleDF,
    'embedding'
)

Baseline_literal_sample_result = calculate_scores(
    Baseline_literal_sample,
    'embedding',
    model
)

#----------disc sample--------------------------------
disc_idioms_sample = preprocess_and_merge(
    image_embedding_idiom_sample,
    disc_pie_idiom_sampleDF,
    'embedding'
)

disc_idioms_sample_result = calculate_scores(
    disc_idioms_sample,
    'embedding',
    model
)

disc_literal_sample = preprocess_and_merge(
    image_embedding_literal_sample,
    disc_pie_literal_sampleDF,
    'embedding'
)

disc_literal_sample_result = calculate_scores(
    disc_literal_sample,
    'embedding',
    model
)

#----------vit sample--------------------------------
vit_idiom_sample = preprocess_and_merge(
    image_embedding_idiom_sample,
    vit_pie_idiom_sampleDF,
    'embedding'
)

vit_idiom_sample_result = calculate_scores(
    vit_idiom_sample,
    'embedding',
    model
)

vit_literal_sample = preprocess_and_merge(
    image_embedding_literal_sample,
    vit_pie_literal_sampleDF,
    'embedding'
)

vit_literal_sample_result = calculate_scores(
    vit_literal_sample,
    'embedding',
    model
)


# 定义保存路径
submission_dir = current_dir / "submission"
train_dir = submission_dir / "Train"
sample_dir = submission_dir / "Sample"

# 创建目录
train_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)

# 调用函数处理文件
process_pt_files(
    current_dir=current_dir,
    target_dirs={
        "train": train_dir,
        "sample": sample_dir
    }
)



save_results_to_tsv(Baseline_literal_sample_result, "Baseline_literal_sample_result")
save_results_to_tsv(plainB_literal_sample_result, "Baseline_idioms_sample_result")

save_results_to_tsv(Baseline_literal_sample_result, "plainB_literal_sample_result")
save_results_to_tsv(plainB_literal_sample_result, "plainB_idioms_sample_result")

save_results_to_tsv(Baseline_literal_sample_result, "disc_literal_sample_result")
save_results_to_tsv(plainB_literal_sample_result, "disc_idioms_sample_result")

save_results_to_tsv(plainB_literal_sample_result, "vit_literal_sample_result")
save_results_to_tsv(plainB_literal_sample_result, "vit_idioms_sample_result")


