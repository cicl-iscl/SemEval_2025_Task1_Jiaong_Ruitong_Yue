# SemEval_2025_Task1_Jiaong_Ruitong_Yue
SubTask A in SemEval 2025 is to rank the images according to how well they represent the sense in which the nominal compound is used in the given context sentences.
As shown in the flowchart below, we plan to use the CLIP ViT-L/14 model to generate embeddings for the given images. For the noun compounds (NCs) in the sentences, we will employ the IDentifier of Idiomatic Expressions via Semantic Compatibility (DISC)—a model specifically designed to identify idioms in sentences and generate their embeddings. This will allow us to obtain embeddings representing either the idiomatic or literal meaning of the NCs within the given contextual sentence.
Next, we will calculate the cosine similarity between the image embeddings and the text embedding. Based on this, we can rank the images according to how well they represent the intended sense of the text. In the final step, we aim to evaluate the correlation between the text embedding and its corresponding image embeddings using cross-entropy. This will help us assess the model's degree of alignment between the text and the images.