from transformers import BertTokenizerFast, BertModel
import torch
import pandas as pd

# read from tsv
df = pd.read_csv('subtask_a_train.tsv', sep="\t")
# separate the data by sentence_type
train_idiom_df = df[df['sentence_type'] == 'idiomatic']
train_literal_df = df[df['sentence_type'] == 'literal']
# get the list of compounds
pie_idiom = train_idiom_df['compound'].tolist()
pie_literal = train_literal_df['compound'].tolist()

# initialize the tokenizer and the model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def obtain_pie_embeddings(pies):
    """
    Obtain embeddings for compounds
    :param pies: a list of compounds
    :return: the embeddings for compounds
    """
    pie_embeddings = dict()
    for p in pies:
        inputs = tokenizer(p, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # mean pooling the results
        pie_embeddings[p] = last_hidden_state.mean(dim=1)
    return pie_embeddings


pie_embeddings_idiom = obtain_pie_embeddings(pie_idiom)
pie_embeddings_literal = obtain_pie_embeddings(pie_literal)
# iterate the dictionaries to check the output
for key, value in pie_embeddings_idiom.items():
    print(f"{key}, {value.shape}")
for key, value in pie_embeddings_literal.items():
    print(f"{key}, {value.shape}")

# save the embeddings in .pt
torch.save(pie_embeddings_idiom, "Baseline_emb/baseline_pie_embeddings_idioms.pt")
torch.save(pie_embeddings_literal, "Baseline_emb/baseline_pie_embeddings_literal.pt")
