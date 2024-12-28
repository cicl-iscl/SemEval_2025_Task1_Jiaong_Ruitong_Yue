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
from helper_functions import *

# Get the current working path
current_dir = Path.cwd()  

# Build the embedding path
plainB_pie_literal_train_path = current_dir / "plainBERT_pie_embeddings_literal_train.pt"
plainB_pie_idiom_train_path = current_dir / "plainBERT_pie_embeddings_idiom_train.pt"


baseline_pie_literal_train_path = current_dir / "baseline_pie_embeddings_literal_train.pt"
baseline_pie_idiom_train_path = current_dir / "baseline_pie_embeddings_idiom_train.pt"


disc_pie_literal_train_path = current_dir / "disc_pie_embeddings_literal_train.pt"
disc_pie_idiom_train_path = current_dir / "disc_pie_embeddings_idiom_train.pt"

vit_pie_literal_train_path = current_dir / "VIT_pie_embeddings_literal_train.pt"
vit_pie_idiom_train_path = current_dir / "VIT_pie_embeddings_idiom_train.pt"


image_literal_train_path = current_dir / "vit_embeddings_train_literal.pt"
image_idiom_train_path = current_dir / "vit_embeddings_train_idiomatic.pt"


# Load files
plainB_pie_literal_train = torch.load(plainB_pie_literal_train_path)
plainB_pie_idiom_train = torch.load(plainB_pie_idiom_train_path)


baseline_pie_literal_train = torch.load(baseline_pie_literal_train_path)
baseline_pie_idiom_train = torch.load(baseline_pie_idiom_train_path)

disc_pie_literal_train = torch.load(disc_pie_literal_train_path)
disc_pie_idiom_train = torch.load(disc_pie_idiom_train_path)


vit_pie_literal_train = torch.load(vit_pie_literal_train_path)
vit_pie_idiom_train = torch.load(vit_pie_idiom_train_path)

image_idiom_train = torch.load(image_idiom_train_path)
image_literal_train = torch.load(image_literal_train_path)

print("All files loaded successfully.")



print("All embeddings loaded successfully!")

plainB_pie_literal_trainDF = pd.DataFrame(list(plainB_pie_literal_train.items()), columns=["compound", "embedding"])
plainB_pie_idiom_trainDF = pd.DataFrame(list(plainB_pie_idiom_train.items()), columns=["compound", "embedding"])


baseline_pie_literal_trainDF = pd.DataFrame(list(baseline_pie_literal_train.items()), columns=["compound", "embedding"])
baseline_pie_idiom_trainDF = pd.DataFrame(list(baseline_pie_idiom_train.items()), columns=["compound", "embedding"])


disc_pie_literal_trainDF = pd.DataFrame(list(disc_pie_literal_train.items()), columns=["compound", "embedding"])
disc_pie_idiom_trainDF = pd.DataFrame(list(disc_pie_idiom_train.items()), columns=["compound", "embedding"])


vit_pie_literal_trainDF = pd.DataFrame(list(vit_pie_literal_train.items()), columns=["compound", "embedding"])
vit_pie_idiom_trainDF = pd.DataFrame(list(vit_pie_idiom_train.items()), columns=["compound", "embedding"])


#---------------image embeddings--------------------------------
image_idiom_train = torch.load(image_idiom_train_path)
image_literal_train = torch.load(image_literal_train_path)

image_embedding_idiom_train = []
image_embedding_literal_train = []

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



#---------------image embeddings--------------------------------

#---------------model init--------------------------------
embedding_dim = 768
hidden_dim = 128
model = VectorComparisonNet(embedding_dim, hidden_dim)
#---------------model init--------------------------------

# merge together

#---------------Baseline--------------------------------

#---------------Baseline--------------------------------
    
#----------plainB--------------------------------
Baseline_idioms_train = preprocess_and_merge(
    image_embedding_idiom_train,
    baseline_pie_idiom_trainDF,
    'embedding'
)

Baseline_idioms_train_result = calculate_scores(
    Baseline_idioms_train,
    'embedding',
    model
)

Baseline_literal_train = preprocess_and_merge(
    image_embedding_literal_train,
    baseline_pie_literal_trainDF,
    'embedding'
)

Baseline_literal_train_result = calculate_scores(
    Baseline_literal_train,
    'embedding',
    model
)

plainB_literal_train = preprocess_and_merge(
    image_embedding_literal_train,
    plainB_pie_literal_trainDF,
    'embedding'
)

plainB_literal_train_result = calculate_scores(
    plainB_literal_train,
    'embedding',
    model
)

# 处理 plainB_idiom_sample 数据
plainB_idiom_train = preprocess_and_merge(
    image_embedding_idiom_train,
    plainB_pie_idiom_trainDF,
    'embedding'
)

plainB_idiom_train_result = calculate_scores(
    plainB_idiom_train,
    'embedding',
    model
)

    
#--------------------disc---------------------------------
# 合并和预处理数据
disc_idioms_train = preprocess_and_merge(
    image_embedding_idiom_train,  # 图像嵌入数据
    disc_pie_idiom_trainDF,       # 文本嵌入数据
    'embedding'              # 文本嵌入列名称
)

# 计算相似性分数
disc_idioms_train_result = calculate_scores(
    disc_idioms_train,           # 预处理后的数据
    'embedding',            # 文本嵌入列名称
    model                        # 神经网络模型
)


# 预处理和合并数据
disc_literal_train = preprocess_and_merge(
    image_embedding_literal_train,
    disc_pie_literal_trainDF,
    'embedding'  # 指定文本嵌入列的名称
)

# 计算相似性分数
disc_literal_train_result = calculate_scores(
    disc_literal_train,
    'embedding',  # 指定文本嵌入列的名称
    model  # 使用的神经网络模型
)

    
#-------------------------------vit----------------------
vit_idiom_train = preprocess_and_merge(
    image_embedding_idiom_train,
    vit_pie_idiom_trainDF,
    'embedding'
)

vit_idiom_train_result = calculate_scores(
    vit_idiom_train,
    'embedding',
    model
)

#-------------------------------vit literal----------------------
vit_literal_train = preprocess_and_merge(
    image_embedding_literal_train,  # 图像嵌入数据
    vit_pie_literal_trainDF,        # 文本嵌入数据
    'embedding'                     # 文本嵌入列名称
)

vit_literal_train_result = calculate_scores(
    vit_literal_train,  # 预处理后的数据
    'embedding',        # 文本嵌入列名称
    model               # 神经网络模型
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


# 保存结果到不同的 TSV 文件
save_results_to_tsv(Baseline_literal_train_result, "Baseline_literal_train_result")
save_results_to_tsv(Baseline_idioms_train_result, "Baseline_idiom_train_result")

save_results_to_tsv(plainB_literal_train_result, "plainB_literal_train_result")
save_results_to_tsv(plainB_idiom_train_result, "plainB_idiom_train_result")

save_results_to_tsv(disc_literal_train_result, "disc_literal_train_result")
save_results_to_tsv(disc_idioms_train_result, "disc_idioms_train_result")

save_results_to_tsv(vit_literal_train_result, "vit_literal_train_result")
save_results_to_tsv(vit_idiom_train_result, "vit_idiom_train_result")