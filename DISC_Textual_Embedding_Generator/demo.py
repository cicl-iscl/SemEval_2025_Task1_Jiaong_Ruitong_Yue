#!/usr/bin/env python
# coding: utf-8

# # Demo
# This demo shows how to use DISC model with your own sentence input. 
# 
# The script will first load the model and data processing module. 
# Then, it will run inference on the input sentence and output the detected idiom from the sentence. 
# 

# In[1]:


from IPython.display import display, HTML
import torch
import numpy as np
from tqdm import tqdm
from src.utils.model_util import load_model_from_checkpoint
from src.model.read_comp_triflow import ReadingComprehensionDetector as DetectorMdl
from config import Config as config
from demo_helper.data_processor import DataHandler
from demo_helper.visualize_helper import simple_scoring_viz


# ## 1. Load model

# In[2]:


data_handler = DataHandler(config)
detector_model= load_model_from_checkpoint(DetectorMdl, data_handler.config)


# ## 2. Set and prepare input sentences

# In[3]:


sentences = [
    # The following examples are idioms
    'If you’re head over heels, you’re completely in love.',
    'If you keep someone at arm’s length, you avoid becoming friendly with them.',
    'If you’re a chip off the old block, you’re similar in some distinct way to your father or mother.',
    'He must face the music for his transgression.',
    'Getting fired turned out to be a blessing in disguise.',
    'I’m sorry but I just can’t seem to wrap my head around it.',
    'At the end of the day, it is you who will take the heat.',
    'At the end of the day, it is you who will take the responsibility.',
    'I don’t want to be Hayley’s friend anymore, she stabbed me in the back!',
    'Why not go to the post office on your way to the mall and kill two birds with one stone?',
    'Hey, I’m feeling pretty angry right now. I’m going to go blow off some steam.',
    'As a rule of thumb, you should usually pay for your date’s dinner, too.',
    'If you burn the candle at both ends, you work excessively hard, say, by keeping two jobs or by leading a busy social life in the evening.',
    # The following examples are similes
    'You were as brave as a lion.',
    'This house is as clean as a whistle.',
    "Sometimes you feel like a nut, sometimes you don't.",
    # Negative examples (no idioms)
    "We will also see which library is recommended to use on each occasion and the unique capabilities of each library."
]


# In[4]:


data = data_handler.prepare_input(sentences)


# ## 3. Model inference

# In[5]:


with torch.no_grad():
    ys_ = detector_model(data)
    probs = torch.nn.functional.softmax(ys_, dim=-1)
ys_ = ys_.cpu().detach().numpy()
probs = probs.cpu().detach().numpy()
idiom_class_probs = probs[:, :, -1].tolist()
predicts = np.argmax(ys_, axis=2)


# ## 4. Extract output

# In[6]:


ys_.shape


# In[7]:


sentences_tkns = data['xs_bert'].cpu().detach().numpy().tolist()
sentences_tkns = [data_handler.tokenizer.convert_ids_to_tokens(s) for s in sentences_tkns]


# In[8]:


print('Visualize Results by Scoring: ')
for i in range(len(sentences_tkns)):
    s = simple_scoring_viz(sentences_tkns[i], idiom_class_probs[i], 'YlGn')
    display(HTML(s))


# In[9]:


predicts = predicts == 4
predicts = predicts.astype(float)
print('Visualize Results by Classification: ')
for i in range(len(sentences_tkns)):
    s = simple_scoring_viz(sentences_tkns[i], predicts[i], 'YlGn')
    display(HTML(s))


# In[ ]:




