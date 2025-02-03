# Project 3: Inference Optimization

The focus of this project is on optimizing inference, specifically aimed at reducing the time required for making predictions by decreasing the number of computations involved.

## Optimization Approaches

We will explore the following methods to achieve inference optimization:

- **Model Quantization** (0.25 points)
- **Model Weights Pruning** (0.25 points)
- **Layer Pruning** (0.5 points)
- **Optimized Inference Libraries** (e.g., ONNX) (0.25 points)
- **Knowledge Distillation** (0.5 points)

## Objective

Your task is to implement these optimization methods on a selected model architecture (For example:  the one from Project 2).

### Requirements:
1. **Calculate and Compare Metrics**:
   - Use an reasonable metric (e.g., F1 score for classification tasks) to assess both the unoptimized and optimized models.
   - Measure and report the difference in inference time for each method applied.

2. **Multiple Methods**:
   - If applying multiple methods, assess the combined impact on both inference time and model performance.

3. **Present Results**:
   - Organize your findings in a clear format, such as a table or chart, to highlight changes in performance and inference time.

## Class Presentation

During class, you will be asked to present:
- Which optimization methods you implemented.
- How each method (as well as in combination)  affected inference time and model accuracy.

## Submission Guidelines

- Please don’t copy outright whole solutions, like the one linked below.
- Please upload your code to a repository. 
- This homework can grant you up to **1 point maximum.**
- **Deadline**: November 27, 2024


## Useful links:
- https://pytorch.org/tutorials/intermediate/pruning_tutorial.html 
- https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html 
- https://pytorch.org/docs/stable/quantization.html
- https://onnx.ai/
