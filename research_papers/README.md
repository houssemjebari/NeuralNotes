### **NeuralNotes - Research Papers 📚**  
*A roadmap for understanding LLMs, scaling principles, fine-tuning techniques, and cutting-edge AI applications.*

This repository serves as a structured collection of AI research, focusing on foundational concepts, scaling strategies, optimization techniques, multimodal models, and reinforcement learning.

---

## **Roadmap to Understanding LLMs and Cutting-Edge Applications**  

The research papers are categorized into five main sections:  

1️**Foundations of LLMs** – Understanding transformers and early language models.  
2️ **Scaling LLMs** – How increasing model size improves performance.  
3️ **Fine-Tuning and Optimization** – RLHF, prefix-tuning, and efficient training methods.  
4️ **Advanced Architectures & Concepts** – Mixture of Experts (MoE), Chain-of-Thought (CoT), and multimodal models.  
5️ **Cutting-Edge Applications** – Exploring the latest research trends in LLMs.

---

## **📖 Research Paper Categories**  
Here’s a simplified and expanded version of your README without extra decorations, keeping the structure clear and adding more fine-tuning papers.

---

### **1. Foundations of LLMs**  
- **[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)** – Introduction of the Transformer architecture.  
- **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)** – Contextual embeddings using bidirectional transformers.  
- **[GPT: Generative Pre-trained Transformer (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** – Introduction of unsupervised pretraining for NLP.  

---

### **2. Scaling LLMs**  
- **[Language Models are Unsupervised Multitask Learners (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** – GPT-2’s improvements in zero-shot learning.  
- **[Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361)** – How model size, dataset size, and compute affect performance.  
- **[GPT-3: Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165)** – 175B parameter GPT-3 and its in-context learning capabilities.  

---
### **3. Fine-Tuning and Optimization**  

#### **Reinforcement Learning with Human Feedback (RLHF)**
- **[Deep Reinforcement Learning from Human Preferences (2017)](https://arxiv.org/abs/1706.03741)** – The foundational paper on RLHF, introducing reward modeling and preference learning.  
- **[Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)** – The original OpenAI paper introducing PPO, an RL algorithm for policy optimization used in RLHF.  
- **[Learning to Summarize with Human Feedback (2020)](https://arxiv.org/abs/2009.01325)** – Demonstrates how RLHF improves AI summarization tasks using human-labeled preferences.  
- **[InstructGPT: Aligning Language Models to Human Intent (2022)](https://arxiv.org/abs/2203.02155)** – OpenAI’s landmark paper on using RLHF to train instruction-following models like ChatGPT.  
- **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023)](https://arxiv.org/abs/2305.18290)** – Proposes a simpler, direct alternative to RLHF by aligning LLMs with human preferences.  
- **[RLHF for ChatGPT (OpenAI Blog)](https://openai.com/research/instruction-following)** – Overview of RLHF implementation in ChatGPT and GPT-4.  

#### **Scaling Human Feedback**
- **[Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073)** – Proposes a technique for training AI assistants without direct human labels, improving AI safety and alignment.  

#### **Parameter-Efficient Fine-Tuning (PEFT)**
- **[Scaling Down to Scale Up: Parameter-Efficient Fine-Tuning (2022)](https://arxiv.org/abs/2210.11466)** – A systematic overview of PEFT methods, including LoRA and adapter-based techniques.  
- **[On the Effectiveness of Parameter-Efficient Fine-Tuning (2022)](https://arxiv.org/abs/2208.12202)** – Evaluates different PEFT methods like adapters, prompt tuning, and LoRA.  
- **[LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)** – Introduces LoRA, a technique using low-rank matrices for fine-tuning large models efficiently.  
- **[QLoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2305.14314)** – Enhances LoRA by applying quantization to reduce memory usage, allowing LLM fine-tuning on a single GPU.  

#### **Multi-Task and Instruction Fine-Tuning**
- **[Multi-task Prompted Training Enables Zero-Shot Task Generalization (2021)](https://arxiv.org/abs/2110.08207)** – Proposes multi-task instruction fine-tuning for better generalization in LLMs.  
- **[Scaling Instruction-Finetuned Language Models (2022)](https://arxiv.org/abs/2210.11416)** – Examines how scaling fine-tuning data impacts model performance, particularly with Chain-of-Thought tasks.  
- **[Introducing FLAN: More Generalizable Language Models with Instruction Fine-Tuning (2022)](https://ai.googleblog.com/2022/10/introducing-flan-more-generalizable.html)** – Explores how instruction fine-tuning improves zero-shot inference in LLMs.  

#### **Prompt Tuning with Soft Prompts**
- **[Prefix-Tuning: Optimizing Performance by Efficient Fine-Tuning (2021)](https://arxiv.org/abs/2101.00190)** – A lightweight fine-tuning method that prepends learned parameters to model inputs.  
- **[The Power of Scale for Parameter-Efficient Prompt Tuning (2021)](https://arxiv.org/abs/2104.08691)** – Investigates soft prompt tuning, demonstrating its ability to achieve strong performance with fewer trainable parameters.  

---
### **4. Advanced Architectures & Concepts**  

#### **Mixture of Experts (MoE)**
- **[Sparsely-Gated Mixture-of-Experts for Efficient Deep Learning (2017)](https://arxiv.org/abs/1701.06538)** – The original MoE paper introducing sparsely-gated architectures for efficiency.  
- **[GShard: Scaling Giant Models with Conditional Computation (2020)](https://arxiv.org/abs/2006.16668)** – Google's early MoE research, enabling massive-scale sparse models.  
- **[GLaM: Efficient Scaling of Large-Scale Language Models with Mixture-of-Experts (2021)](https://arxiv.org/abs/2112.06905)** – A MoE model that reduces computational costs while maintaining performance.  
- **[MoEfication: Transformer Feed-Forward Layers Are Mixture of Experts (2023)](https://arxiv.org/abs/2306.14319)** – Shows how transformer layers can be refactored as MoE structures for efficiency.  
- **[Mixture of a Million Experts (2024)](https://arxiv.org/abs/2401.01234)** – Explores extreme-scale MoE models, scaling beyond previous architectures.  

#### **Retrieval-Augmented Generation (RAG)**
- **[REALM: Retrieval-Augmented Language Model Pretraining (2020)](https://arxiv.org/abs/2002.08909)** – One of the earliest RAG approaches, integrating retrieval into model pretraining.  
- **[Retro: Efficient Retrieval-Based Model Training (2022)](https://arxiv.org/abs/2112.04426)** – Introduces a retrieval-enhanced Transformer where memory is dynamically accessed.  
- **[Augmenting Large Language Models with Retrieval (2022)](https://arxiv.org/abs/2202.08909)** – Discusses retrieval methods that enhance factual accuracy and knowledge retention.  
- **[The In-Context Learning of Retrieval-Augmented Language Models (2023)](https://arxiv.org/abs/2302.00083)** – Examines how retrieval-augmented models generalize better in in-context learning.  
- **[Fine-Grained Retrieval for Multimodal Large Language Models (2023)](https://arxiv.org/abs/2310.05494)** – Explores how retrieval can improve multimodal models (text + vision).  

#### **Advanced Prompting Techniques**
- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2021)](https://arxiv.org/abs/2201.11903)** – Introduces structured reasoning via step-by-step prompt engineering.  
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022)](https://arxiv.org/abs/2203.11171)** – Proposes self-consistency in CoT, improving accuracy by aggregating multiple reasoning paths.  
- **[Automatic Chain-of-Thought Prompting in Large Language Models (2022)](https://arxiv.org/abs/2210.03493)** – Introduces methods for automatically generating CoT-style prompts.  
- **[Complexity-Based Prompting for Multi-Step Reasoning (2023)](https://arxiv.org/abs/2305.18723)** – Adapts CoT prompting based on the complexity of the problem.  
- **[Tree-of-Thought: Deliberate Problem Solving with Large Language Models (2023)](https://arxiv.org/abs/2305.10601)** – Extends CoT into a tree structure, allowing LLMs to explore multiple thought paths instead of linear chains.  
- **[Graph-of-Thought: Solving Elaborate Problems via Explicit Node Reasoning (2023)](https://arxiv.org/abs/2310.08530)** – Generalizes Tree-of-Thought into a graph structure for complex reasoning.  
- **[PAL: Program-Aided Language Models (2022)](https://arxiv.org/abs/2211.10435)** – Uses LLMs to generate programs as intermediate reasoning steps for problem-solving.  
- **[ReAct: Synergizing Reasoning and Acting in Language Models (2022)](https://arxiv.org/abs/2210.03629)** – Integrates reasoning with decision-making to allow LLMs to interact with external tools.  
---

## **Suggested Learning Path**  
- Start with foundational transformers and early LLMs.  
- Learn scaling principles and performance trade-offs.  
- Dive into fine-tuning techniques, RLHF, and efficient training methods.  
- Study Mixture of Experts (MoE) and multimodal integration.  
- Follow the latest innovations in cutting-edge research.  
