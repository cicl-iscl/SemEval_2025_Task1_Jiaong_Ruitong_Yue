from transformers import CLIPModel, CLIPTokenizer
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

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")


def compute_offsets(sentences, tokenizer):
    """
    A method that calculate the offsets of the sentences based on the CLIPTokenizer
    :param sentences: a list of sentences
    :param tokenizer: CLIPTokenizer
    :return: the offsets of the sentences
    """
    offsets = []
    # iterate to obtain the offset of each sentence
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        # move along with the token
        cur = 0
        offset = []
        for tok in tokens:
            # remove the special marks
            clean_tok = tok.replace("</w>", "").lstrip("Ġ")
            # find the position of the token in the original sentence
            start = sent.find(clean_tok, cur)
            if start == -1:
                raise ValueError(f" Token '{clean_tok}' can not be found in '{sent}'!")
            end = start + len(clean_tok)
            offset.append((start, end))
            # update the current position
            cur = end
        offsets.append(offset)
    return offsets


def slice_pie_embeddings(sentences, pies):
    """
    Given a list of sentences and a list of pie inside each sentence, return the start and end positions
  of each pie in the tokenized sentences.
    :param sentences: list of sentences
    :param pies: list of pies
    :return: a dictionary mapping each compound to its embedding
    """
    pie_embeddings = dict()
    encoded = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    # the model can't take offsets_mapping as arguments, so i write an extra method to get the offsets
    offsets = compute_offsets(sentences, tokenizer)
    with torch.no_grad():
        # use the text model of the model
        outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        sent_tok_embeddings = outputs.last_hidden_state
    for i, (sen, p) in enumerate(zip(sentences, pies)):
        sen = sen.lower()
        # locate the compound
        pie_start = sen.find(p)
        pie_end = pie_start + len(p)
        pie_idx = []
        # find the position of the compound in the offsets
        for idx, (start, end) in enumerate(offsets[i]):
            if start >= pie_start and end <= pie_end:
                pie_idx.append(idx)
        # retrieve the embeddings of tokens in the compound
        pie_tok_embeddings = sent_tok_embeddings[i, pie_idx]
        pie_embedding = torch.mean(pie_tok_embeddings, dim=0)
        pie_embeddings[p] = pie_embedding
    return pie_embeddings


pie_embeddings_idiom = slice_pie_embeddings(sent_idiom, pie_idiom)
pie_embeddings_literal = slice_pie_embeddings(sent_literal, pie_literal)
# iterate the dictionaries to check the output
for key, value in pie_embeddings_idiom.items():
    print(f"{key}, {value.shape}")
for key, value in pie_embeddings_literal.items():
    print(f"{key}, {value.shape}")
# save the embeddings in .pt
torch.save(pie_embeddings_idiom, "VIT_emb/VIT_pie_embeddings_idioms.pt")
torch.save(pie_embeddings_literal, "VIT_emb/VIT_pie_embeddings_literal.pt")




