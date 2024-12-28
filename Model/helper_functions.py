# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:17:28 2024

@author: yuyue

Main script for processing embeddings and saving results.
"""

import torch
import pandas as pd
import numpy as np
import os


def preprocess_and_merge(image_embedding_df, text_embedding_df, text_column_name):
    """
    合并两个 DataFrame，并对嵌入列进行预处理。
    
    :param image_embedding_df: 图像嵌入 DataFrame
    :param text_embedding_df: 文本嵌入 DataFrame
    :param text_column_name: 文本嵌入列名称
    :return: 预处理后的 DataFrame
    """
    merged_df = pd.merge(
        image_embedding_df.rename(columns={'embedding': 'image_embedding'}),
        text_embedding_df.rename(columns={'embedding': text_column_name}),
        on='compound',
        how='inner'
    )

    # 对嵌入列进行类型转换
    merged_df['image_embedding'] = merged_df['image_embedding'].apply(
        lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, (list, np.ndarray)) else x
    )
    merged_df[text_column_name] = merged_df[text_column_name].apply(
        lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0) if isinstance(x, (list, np.ndarray)) else x
    )

    return merged_df


def calculate_scores(df, text_column_name, model):
    """
    计算图像和文本嵌入之间的相似性分数。
    
    :param df: 预处理后的 DataFrame
    :param text_column_name: 文本嵌入列名称
    :param model: 用于计算相似性的神经网络模型
    :return: 包含 compound 和 expected_order 的结果列表
    """
    results = []
    for compound, entries in df.groupby('compound'):
        scores = []
        for _, row in entries.iterrows():
            image_id = row['image_name']
            image_embedding = row['image_embedding']
            text_embedding = row[text_column_name]

            # 验证嵌入类型
            assert isinstance(image_embedding, torch.Tensor) and image_embedding.dtype == torch.float32, "Invalid image_embedding type"
            assert isinstance(text_embedding, torch.Tensor) and text_embedding.dtype == torch.float32, f"Invalid {text_column_name} type"

            # 计算相似性分数
            similarity_score = model(image_embedding, text_embedding).item()
            scores.append((image_id, similarity_score))

        # 排序并格式化输出
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_image_ids = [f"'{item[0]}'" for item in scores]

        results.append({
            'compound': compound,
            'expected_order': f"[{', '.join(ranked_image_ids)}]"
        })

    return results



def create_directory(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory '{dir_name}' is ready.")


def save_results_to_tsv(results, name, dir_name="submission"):
    create_directory(dir_name)

    results_df = pd.DataFrame(results)
    output_file = os.path.join(dir_name, f"{name}.tsv")
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved results to {output_file}")
    
def process_pt_files(current_dir, target_dirs):
    """
    处理 .pt 文件，根据文件名的关键词将文件分类保存到目标目录。
    
    :param current_dir: 当前目录路径
    :param target_dirs: 目标目录字典，键为关键词，值为目录路径
    """
    for keyword, target_dir in target_dirs.items():
        target_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在

    for file in current_dir.glob("*.pt"):
        data = torch.load(file)
        target_path = None

        for keyword, target_dir in target_dirs.items():
            if keyword in file.name.lower():
                target_path = target_dir / file.name
                break

        if target_path is None:
            print(f"Skipping file: {file.name}")
            continue

        torch.save(data, target_path)
        print(f"Saved {file.name} to {target_path}")

    print("All files have been processed.")


if __name__ == "__main__":
   pass
