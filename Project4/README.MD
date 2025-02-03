# Project 4: Retrieval-Augmented Generation (RAG) System

## Objective
Create a Retrieval-Augmented Generation (RAG) system using an embedding model, a vector store, and an LLM (Large Language Model). (1 Point)

## Acceptance Criteria
1. **Pipeline Setup**: 
   - Establish a pipeline that integrates a dense embedding model with a vector store.
   - Connect the vector store to an LLM.
   - Refer to the schemas provided in the linked presentation for detailed guidance.

2. **Demonstration**:
   - Present your system in a **live demo** during the class.
   - Note: A command-line interface (CLI) is sufficient; no frontend development is required.

3. **Model Usage**:
   - Using external API-served LLMs (e.g., GPT-4) is allowed.
   - However, using a **local LLM model** will be rewarded with 0.5 additonal points.

4. **Embedding Models**:
   - External embedding models served through APIs are **not allowed**.

## Additional Points
- **Local LLM Usage**: 
  - Earn **0.5 additional points** if you use a local LLM model (e.g., Mistral 1.5B).
  
- **Model Serving and Vector Databases**:
  - Earn **2 additional points** for using any model-serving package (e.g., BentoML, TorchServe, Nvidia Triton) in combination with a **standalone vector database** (e.g., Weaviate, Milvus).
  - Note: FAISS or LLAMA Index are **not standalone vector databases**.
  - This setup will likely require REST API communication between the models.

- **Maximum Points**:
  - The total maximum points achievable for this task is **2.0**.

## Useful Links
- **Frontend (Optional)**:
  - [Streamlit for Conversational Apps](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
  
- **Class Presentation**:
  - [Presentation Link](https://docs.google.com/presentation/d/1UyXhqp7qmAp1J0ktqwnaCNxqA-0_dZ3ZANPkyAgPOPM/edit?usp=sharing)

- **Additional Resources**:
  - [RAG Tutorial (LangChain)](https://python.langchain.com/docs/tutorials/rag/)
  - [Sentence Transformers](https://sbert.net/)
  - [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
  - [Relevant Paper](https://arxiv.org/pdf/2407.01219)
