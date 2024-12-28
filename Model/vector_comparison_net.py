# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:38:47 2024

@author: yuyue
"""


import torch
import torch.nn as nn

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

if __name__ == "__main__":
    # test
    model = VectorComparisonNet(embedding_dim=768, hidden_dim=128)
    image_emb = torch.randn(768)
    text_emb = torch.randn(768)
    similarity_score = model(image_emb, text_emb)
    print(f"Similarity score: {similarity_score.item()}")
