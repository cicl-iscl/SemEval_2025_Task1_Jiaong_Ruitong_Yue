# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:38:47 2024

@author: yuyue
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class VectorComparisonNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(VectorComparisonNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_emb, text_emb):
        if image_emb.dim() == 1:
            image_emb = image_emb.unsqueeze(dim=0)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(dim=0)

        combined = torch.cat([image_emb, text_emb], dim=-1)
        hidden = self.relu(self.fc1(combined))
        output = self.sigmoid(self.fc2(hidden))
        return output


    def prepare_training_data(image_embeddings, text_embeddings, golden_rank):
       """
       准备训练数据。
      :param image_embeddings: 图片嵌入字典 {image_id: embedding}
      :param text_embeddings: 文本嵌入字典 {compound: embedding}
      :param golden_rank: 黄金排名 {compound: [image_ids]}
      :return: 训练数据 [(image_emb, text_emb, label)]
       """
       training_data = []

       for compound, ranked_images in golden_rank.items():
           if compound in text_embeddings:
              text_embedding = text_embeddings[compound]
              for i, image_id in enumerate(ranked_images):
                 if image_id in image_embeddings:
                    image_emb = image_embeddings[image_id]
                    label = len(ranked_images) - i  # 越靠前分数越高
                    training_data.append((image_emb, text_embedding, label))
       return training_data


    # Pairwise Ranking Loss
    
    
def pairwise_ranking_loss(predictions, labels):
    """
    计算排序损失。
    :param predictions: 模型预测的相似性分数列表
    :param labels: golden_rank 转化的标签分数
    :return: 总损失值
    """
    loss = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] > labels[j]:  # i 应高于 j
                diff = predictions[i] - predictions[j]
                loss += F.relu(1 - diff)  # 如果差距小于 margin，增加损失
    return loss

    
    def compute_ranking_loss(model, image_embeddings, text_embedding, golden_rank, device):
      """
      基于 MarginRankingLoss 的训练损失。
      :param model: 神经网络模型
      :param image_embeddings: 当前 compound 的图片嵌入 {image_id: embedding}
      :param text_embedding: 当前 compound 的文本嵌入
      :param golden_rank: 当前 compound 的图片黄金排名
      :param device: 使用的计算设备
      :return: 总损失
      """
      total_loss = 0
      predictions = []
      
      margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)

       # 计算每个图片的预测分数
      for image_id, image_emb in image_embeddings.items():
        image_emb = image_emb.to(device)
        text_embedding = text_embedding.to(device)
        with torch.no_grad():
            score = model(image_emb, text_embedding).item()
        predictions.append((image_id, score))

    # 根据 golden_rank 的顺序生成 pairwise 数据
      for i in range(len(golden_rank)):
        for j in range(i + 1, len(golden_rank)):
            if golden_rank[i] > golden_rank[j]:  # i 的排名优于 j
                # 获取预测分数
                pred_i = predictions[i][1]
                pred_j = predictions[j][1]
                # 定义目标值
                label = torch.tensor(1.0, dtype=torch.float32, device=device)
                # 计算 MarginRankingLoss
                total_loss += margin_ranking_loss(
                    torch.tensor(pred_i, device=device).unsqueeze(0),
                    torch.tensor(pred_j, device=device).unsqueeze(0),
                    label
                )

      return total_loss



    def evaluate_model(model, image_embeddings, text_embedding, golden_value):
       """
       评估模型排序性能。
       :param model: 训练后的模型
       :param image_embeddings: 图片嵌入
       :param text_embedding: 文本嵌入
       :param golden_value: golden value 排序
       :return: Spearman 相关系数
       """
       model.eval()
       predictions = []
       text_embedding = text_embedding.to(device)

       for image_name, image_emb in image_embeddings.items():
           image_emb = image_emb.to(device)
           with torch.no_grad():
               score = model(image_emb, text_embedding).item()
           predictions.append((image_name, score))

       predictions.sort(key=lambda x: x[1], reverse=True)
       predicted_order = [p[0] for p in predictions]

       rank_golden = {name: i for i, name in enumerate(golden_value)}
       rank_predicted = {name: i for i, name in enumerate(predicted_order)}
       spearman_corr = torch.tensor([
           (rank_golden[name] - rank_predicted[name]) ** 2
           for name in golden_value
       ]).mean().item()

       return spearman_corr


if __name__ == "__main__":
    # 初始化模型、优化器和数据
    file_path = r"C:\Users\yuyue\Downloads\AdMIRe Subtask A Sample\Subtask A\Model\golden_truth.tsv"
    golden_rank = pd.read_csv(file_path, sep='\t')
    embeddings_idiom = r"C:\Users\yuyue\Downloads\AdMIRe Subtask A Sample\Subtask A\submission\plainB_idiom_train.tsv"
    embeddings_literal = r"C:\Users\yuyue\Downloads\AdMIRe Subtask A Sample\Subtask A\submission\plainB_literal_train.tsv"
    embeddings_idiom = pd.read_csv(embeddings_idiom, sep='\t')
    embeddings_literal = pd.read_csv(embeddings_literal, sep='\t')
    
    embeddings = pd.concat([embeddings_idiom, embeddings_literal], axis=0, ignore_index=True)
    # 生成 image_id 到图片嵌入的字典
    image_embeddings_dict = {
            row['image_name']: row['image_embedding'] for _, row in embeddings.iterrows()
     }

    # 生成 compound 到文本嵌入的字典
    text_embeddings_dict = {
            row['compound']: row['embedding'] for _, row in embeddings.iterrows()
     }

    embedding_dim = 512
    hidden_dim = 128
    model = VectorComparisonNet(embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    golden_rank_dict = golden_rank.set_index('compound')['image_name'].apply(eval).to_dict()

    embedding_dim = 512
    hidden_dim = 128
    model = VectorComparisonNet(embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 准备训练数据
    training_data = VectorComparisonNet.prepare_training_data(image_embeddings_dict, text_embeddings_dict, golden_rank)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for compound, entries in golden_rank_dict.items():
            # 当前 compound 的数据
            current_image_embeddings = [image_embeddings_dict[img] for img in entries if img in image_embeddings_dict]
            current_text_embedding = text_embeddings_dict[compound]

            # 计算预测分数
            predictions = []
            for img_emb in current_image_embeddings:
                img_emb = img_emb.unsqueeze(0).to(device)
                current_text_embedding = current_text_embedding.unsqueeze(0).to(device)
                predictions.append(model(img_emb, current_text_embedding).squeeze().item())

            # 计算 pairwise ranking loss
            loss = pairwise_ranking_loss(predictions, list(range(len(entries))))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
