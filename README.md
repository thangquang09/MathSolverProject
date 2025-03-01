# Fine-Tuned LLM for Mathematical Problem Solving

## 1. Introduction

This project focuses on fine-tuning a large language model (LLM) to solve elementary-level math problems. The primary goal is to enhance the model’s ability to understand and generate accurate solutions for basic arithmetic and word problems. The fine-tuning process leverages a dataset sourced from Hugging Face Datasets, ensuring diverse and well-structured training data. By optimizing the model’s performance on these tasks, the project explores the potential of LLMs in educational and tutoring applications.

---

## 2. Datasets

| Dataset Name                          | Description                                                 | Source              |
|---------------------------------------|-------------------------------------------------------------|---------------------|
| `hllj/vi_gsm8k`                       | A Vietnamese version of the GSM8K dataset, containing high-quality grade school math problems with detailed solutions. | Hugging Face Datasets |
| `hllj/vi_grade_school_math_mcq`       | A dataset of multiple-choice grade school math problems in Vietnamese, designed to evaluate mathematical reasoning. | Hugging Face Datasets |

---

## 3. Training

### **1. Model Selection and Quantization**  
The base model used for fine-tuning is `vilm/vinallama-7b-chat`. To optimize training efficiency, **4-bit quantization** is applied using `BitsAndBytesConfig`. This reduces memory usage while maintaining performance, making it feasible to train on limited hardware.  


### **2. LoRA for Efficient Fine-Tuning**  
LoRA (Low-Rank Adaptation) is applied to fine-tune only specific layers of the model while keeping the majority of parameters frozen. The configuration includes:  
- **Rank (`R=16`)** – Defines the size of the LoRA projection matrices.  
- **Scaling factor (`LORA_ALPHA=32`)** – Controls the impact of LoRA updates.  
- **Dropout (`LORA_DROPOUT=0.05`)** – Prevents overfitting.  
- **Target modules** – LoRA is applied to specific layers such as `q_proj`, `k_proj`, `v_proj`, etc.  

This approach significantly reduces computational cost while maintaining adaptability to the new dataset.  

### **3. Training Configuration**  
The model is trained using `Trainer` with the following parameters:  
- **Batch size:** 16 (per device)  
- **Gradient accumulation steps:** 2 (effective batch size increases)  
- **Number of epochs:** 2  
- **Learning rate:** 2e-4  
- **FP16:** Enabled  
- **Save total limit:** 3  
- **Logging steps:** 10  
- **Output directory:** `model/experiments`  
- **Optimization algorithm:** `paged_adamw_8bit`  
- **Scheduler:** Cosine learning rate decay with a warmup ratio of 5%  
- **Dataloader workers:** 4  
- **Reporting:** None  
- **DDP find unused parameters:** False (supports multi-GPU)  

Additionally, **gradient checkpointing** is enabled to reduce memory usage.  

---

## 4. Inference

**How to Use the Inference Script**  

1. **Run the script**:  
   ```bash
   python inference.py
   ```  

2. **Enter a question** when prompted.  

3. **(Optional) Enter answer choices** (A, B, C, D). Press **Enter** if not needed.  

4. **Wait for the model to generate an answer**.  

5. **View the response** printed in the console.

## 5. Future Improvements

### **Future Improvements**  

1. **Expanding the Training Dataset**  
   - Incorporate **more diverse and complex math problems**.  
   - Fine-tune with **Vietnamese math textbooks** for better accuracy.  

2. **Improving Reasoning Capabilities**  
   - Use **Chain-of-Thought (CoT) prompting** to improve step-by-step explanations.  
   - Implement **self-consistency decoding** to enhance answer reliability.  

3. **Optimizing Model Efficiency**  
   - Experiment with **8-bit LoRA fine-tuning** for better trade-off between efficiency and performance.  
   - Reduce inference latency using **FlashAttention or speculative decoding**.  

4. **Enhancing User Interaction**  
   - Develop a **web-based or chatbot interface** for easier access.  
   - Implement **speech-to-text and text-to-speech** for interactive learning.  

5. **Multimodal Capabilities**  
   - Extend the model to support **image-based math problems** (e.g., equations, diagrams).  

---