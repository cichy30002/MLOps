# MLOps Course Repository  
#### Where I learned pretty useful skills for my ML skillset

## Project 1 - Model Deployment  
The task was to deploy a model on the cloud—AWS Lambda in this case. I chose EfficientNet for an image recognition task and successfully set it up on Lambda. Using AWS S3 triggers, I prepared a pipeline that classifies every image placed in a bucket and returns a text file with the predicted class to the bucket.

## Project 2 - PyTorch Lightning  
In this project, I learned how to utilize PyTorch Lightning for model training. The problem chosen for optimization was Body Fat Regression. The model was a simple neural network with two hidden layers and dropout.

## Project 3 - Inference Optimization  
This project focused on learning to optimize trained model inference through quantization, weight pruning, and layer pruning. I used the model from the previous task. Although the results weren't very impressive since the model was already quite small, I gained a deeper understanding of these optimization concepts.

## Project 4 - Retrieval Augmented Generation (RAG)  
For the last project, we worked on a simple but highly useful architecture—RAG. I wanted to avoid using APIs, so I chose a small embeddings model (stella-en 1.5B), the open-source vector store Milvus, and a local LLM (Llama 3.2 3B). All the components were integrated using LangChain and LangGraph. For testing, I fed it with the Bee Movie script and asked some tricky questions. 