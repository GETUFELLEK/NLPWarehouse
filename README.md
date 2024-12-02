# **Fine-Tuning Llama-2 for Swahili News Classification**

This repository outlines the fine-tuning process of **Llama-2**, a pre-trained large language model (LLM), for the task of **Swahili News Classification**. The goal was to adapt Llama-2, originally designed for general-purpose natural language understanding and generation, to classify Swahili news articles into predefined categories.

---

## **Motivation**
- **Low-Resource Language Support**: Swahili, like many African languages, suffers from limited representation in pre-trained language models. This project addresses this gap by fine-tuning Llama-2 on a Swahili dataset.
- **Domain Adaptation**: While Llama-2 excels in general NLP tasks, fine-tuning ensures optimal performance for domain-specific tasks like news categorization.
- **Memory Efficiency**: By leveraging **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)**, we optimized the fine-tuning process to require significantly less computational resources, making it feasible on mid-range hardware.

---

## **Fine-Tuning Process**

### **1. Pre-Trained Model**
- We started with the **Llama-2-7B model** using the `meta-llama/Llama-2-7b-hf` checkpoint from Hugging Face.
- Added a **classification head** with `num_labels` corresponding to the number of news categories.

### **2. Dataset**
- Used a Swahili news dataset with the following structure:
  - `text`: The news article content.
  - `label`: The category of the news article (e.g., politics, sports, business).
- Split the dataset into training and validation sets.

### **3. Preprocessing**
- Tokenized the dataset using Llama-2’s tokenizer to convert text into model-compatible input features (`input_ids`, `attention_mask`).
- Batched the inputs for efficient training.

### **4. Efficient Fine-Tuning**
To handle the computational challenges of fine-tuning a large model like Llama-2-7B, we employed **PEFT** and **LoRA**, which adjust only a small subset of model parameters, significantly reducing memory usage and training time:
- **LoRA Configuration**:
  - Low-rank adaptation with `r=8` and `lora_alpha=32`.
  - Targeted modules: `q_proj` and `v_proj`.
  - Dropout rate: `0.1`.

### **5. Training**
- Fine-tuned the model using Hugging Face's `Trainer` API with the following configurations:
  - **Batch Size**: Adjusted to fit GPU memory constraints (4 per device with gradient accumulation).
  - **Learning Rate**: `2e-5` with Adam optimizer.
  - **Number of Epochs**: 3.
  - **Mixed Precision Training**: Enabled (`fp16`) for faster training and reduced memory usage.

### **6. Evaluation**
- Monitored validation loss and accuracy during training.
- Saved the best-performing model checkpoint.

---

## **Key Modifications**
- **Classification Head**: Added a linear layer to Llama-2 to handle multi-class classification.
- **Parameter-Efficient Fine-Tuning**: Leveraged **PEFT** and **LoRA** to reduce computational overhead, enabling training on mid-range hardware.
- **Tokenizer Compatibility**: Ensured the Swahili dataset was tokenized correctly using Llama-2’s tokenizer.

---

## **Results**
- Achieved improved classification accuracy on Swahili news articles compared to zero-shot performance of the pre-trained model.
- Demonstrated the ability of Llama-2 to adapt to low-resource languages with minimal modifications and efficient resource usage.

---

## **Future Work**
- Expand the dataset to include additional African languages and dialects.
- Experiment with multilingual models to improve generalization across languages.
- Explore reinforcement learning for further domain adaptation.

---


