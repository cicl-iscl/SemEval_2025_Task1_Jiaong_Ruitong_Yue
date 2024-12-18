from transformers import BertTokenizerFast, BertModel
import torch
import pandas as pd

# read from tsv
df = pd.read_csv('subtask_a_train.tsv', sep="\t")
# separate the data by sentence_type
train_idiom_df = df[df['sentence_type'] == 'idiomatic']
train_literal_df = df[df['sentence_type'] == 'literal']
# get the list of sentences and its corresponding compounds
sent_idiom = train_idiom_df['sentence'].tolist()
sent_literal = train_literal_df['sentence'].tolist()
pie_idiom = train_idiom_df['compound'].tolist()
pie_literal = train_literal_df['compound'].tolist()

# initialize the tokenizer and the model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def slice_pie_embeddings(sentences, pies):
    """
  Given a list of sentences and a list of pie inside each sentence, return the start and end positions
  of each pie in the tokenized sentences.
  sentences: list of sentences
  pies: list of pies
  returns: a dictionary mapping each compound to its embedding
  """
    pie_embeddings = dict()
    for i, (sen, p) in enumerate(zip(sentences, pies)):
        sen = sen.lower()
        # model can't take 'offset_mapping' as argument,so i set two encoder
        inputs = tokenizer(sen, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        encoded = tokenizer(sen, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True,
                            return_offsets_mapping=True)
        offsets = encoded['offset_mapping'].squeeze().tolist()
        # delete invalid offsets, for example(0, 0)
        valid_offsets = []
        for idx, (start, end) in enumerate(offsets):
            if start < end:
                valid_offsets.append((idx, start, end))
        # obtain the embeddings of all the tokens in the sentences
        outputs = model(**inputs)
        sent_tok_embeddings = outputs.last_hidden_state
        # locate the compound
        pie_star = sen.find(p)
        pie_end = pie_star + len(p)
        # find the position of the compound in offsets
        start_idx, end_idx = None, None
        for idx, start, end in valid_offsets:
            if start <= pie_star < end:
                start_idx = idx
            if start < pie_end <= end:
                end_idx = idx
                break
        if start_idx is None or end_idx is None:
            raise ValueError(f" '{p}' can not be found in '{sen}'!")
        # retrieve the embeddings of tokens in the compound
        pie_tok_embeddings = sent_tok_embeddings[:, start_idx:end_idx + 1, :]
        pie_embedding = pie_tok_embeddings.mean(dim=1)  # mean pooling over tokens
        pie_embeddings[p] = pie_embedding
    return pie_embeddings


# using method slice_pie_embeddings to acquire the embeddings of all the compounds
pie_embeddings_idiom = slice_pie_embeddings(sent_idiom, pie_idiom)
pie_embeddings_literal = slice_pie_embeddings(sent_literal, pie_literal)
# iterate the dictionaries to check the output
for key, value in pie_embeddings_idiom.items():
    print(f"{key}, {value.shape}")
for key, value in pie_embeddings_literal.items():
    print(f"{key}, {value.shape}")
# save the embeddings in .pt
torch.save(pie_embeddings_idiom, "plainBERT_emb/plainB_pie_embeddings_idioms.pt")
torch.save(pie_embeddings_literal, "plainBERT_emb/plainB_pie_embeddings_literal.pt")
