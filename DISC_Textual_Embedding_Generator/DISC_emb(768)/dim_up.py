import torch
import pandas as pd
import torch.nn as nn

idiom_sample = torch.load('disc_pie_embeddings_idiom_sample_original.pt')
idiom_train = torch.load('disc_pie_embeddings_idiom_train_original.pt')

literal_sample = torch.load('disc_pie_embeddings_literal_sample_original.pt')
literal_train = torch.load('disc_pie_embeddings_literal_train_original.pt')


def linear_projection(e):
    w = nn.Linear(512, 768)
    return w(e)


for c, e in idiom_sample.items():
    t = torch.from_numpy(e)
    idiom_sample[c] = linear_projection(t)

for c, e in idiom_train.items():
    t = torch.from_numpy(e)
    idiom_train[c] = linear_projection(t)

for c, e in literal_sample.items():
    t = torch.from_numpy(e)
    literal_sample[c] = linear_projection(t)

for c, e in literal_train.items():
    t = torch.from_numpy(e)
    literal_train[c] = linear_projection(t)

torch.save(idiom_sample, 'disc_pie_embeddings_idiom_sample.pt')
torch.save(idiom_train, 'disc_pie_embeddings_idiom_train.pt')
torch.save(literal_sample, 'disc_pie_embeddings_literal_sample.pt')
torch.save(literal_train, 'disc_pie_embeddings_literal_train.pt')


