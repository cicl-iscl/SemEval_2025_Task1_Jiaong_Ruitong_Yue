# **SemEval 2025 Task 1 - Jiaong, Ruitong, Yue**  

## **Subtask A: Image Ranking Based on Nominal Compound Sense**  

Subtask A in **SemEval 2025** focuses on ranking images according to how well they represent the **intended sense** (idiomatic or literal) of a **nominal compound (NC)** in a given context sentence.  

---
## Our Paper

Our paper is available at:  
[here](SemEval_2024_Task_1__AdMIRe__Advancing_Multimodal_Idiomaticity_Representation.pdf)


## **Methodology**  

### **1. Generating Image Embeddings**  
- We use the **CLIP ViT-L/14** model to extract embeddings from the provided images.  

### **2. Generating Text Embeddings**  
- For the **NCs in the sentences**, we utilize the **IDentifier of Idiomatic Expressions via Semantic Compatibility (DISC)** model.  
- This model is specifically designed to **detect idioms** and generate embeddings for either the **idiomatic or literal meaning** of NCs within the context.  

### **3. Computing Similarity Scores**  
- We calculate the **cosine similarity** between the **image embeddings** and the **text embedding**.  
- The images are ranked based on their similarity scores, reflecting how well they align with the intended meaning of the NCs.  

### **4. Evaluating Alignment with Cross-Entropy**  
- In the final step, we assess the correlation between the **text embedding** and its corresponding **image embeddings** using **cross-entropy loss**.  
- This evaluation helps determine the **degree of alignment** between the text and image representations.  

---

## **External Links & References**  

For more details, refer to:  
- [SemEval 2025 Task 1 Overview](https://semeval2025-task1.github.io/)  
- ManyICL method is described in the [official research paper](https://arxiv.org/html/2405.09798v1).  

---
