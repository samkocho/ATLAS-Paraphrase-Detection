# Paraphrase Detection Project

## Overview

This project aims to develop a model capable of detecting whether two sentences are paraphrases of each other. The project uses various natural language processing (NLP) techniques and machine learning models, including BERT, RoBERTa, and classical machine learning models such as Random Forest and XGBoost. The dataset used for training and testing is the Microsoft Research Paraphrase Corpus (MRPC).

## Project Structure

The project is organized into the following notebooks:

1. **EDA.ipynb**: This notebook contains the Exploratory Data Analysis (EDA) performed on the MRPC dataset. It provides insights into the data distribution, class imbalance, and some basic text preprocessing steps.
   
2. **BERT.ipynb**: This notebook focuses on building and training a BERT-based model for paraphrase detection. It includes steps for tokenization, model training, evaluation, and fine-tuning.

3. **SBERT.ipynb**: This notebook demonstrates the use of Sentence-BERT (SBERT) for paraphrase detection. It covers the process of obtaining sentence embeddings using SBERT and training a classifier on these embeddings to determine if pairs of sentences are paraphrases.

4. **ROBERTA.ipynb**: Similar to the BERT notebook, this one covers the implementation of a RoBERTa-based model. It details the process of leveraging the RoBERTa transformer for paraphrase detection tasks.

5. **RF_XGBoost.ipynb**: This notebook applies classical machine learning models, specifically Random Forest and XGBoost, to the paraphrase detection problem. It includes feature extraction techniques and model evaluation metrics.

6. **Gradio.ipynb**: This notebook demonstrates how to deploy the paraphrase detection model using Gradio, a Python library for building user-friendly web interfaces. It includes the creation of an interactive demo for users to test the model.

## Dataset

The dataset used is the Microsoft Research Paraphrase Corpus (MRPC), which consists of pairs of sentences along with labels indicating whether each pair is a paraphrase. The dataset is split into training and testing sets provided as `msr_paraphrase_train.txt` and `msr_paraphrase_test.txt`.

The dataset can be downloaded from [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398).


## Methods Used

### BERT and RoBERTa
- **Tokenization**: Tokenization of sentences using BERT and RoBERTa tokenizers.
- **Model Training**: Training the BERT and RoBERTa models on the MRPC dataset.
- **Evaluation**: Evaluating the model performance using metrics such as accuracy, F1-score, and confusion matrix.

### SBERT
- **Sentence Embeddings**: Obtaining sentence embeddings using Sentence-BERT (SBERT).
- **Classifier Training**: Training a classifier on the SBERT embeddings to determine if pairs of sentences are paraphrases.
- **Evaluation**: Assessing the classifier's performance using accuracy, F1-score, and other relevant metrics.


### Random Forest and XGBoost
- **Feature Extraction**: Extracting features from sentences using TF-IDF and word embeddings.
- **Model Training**: Training Random Forest and XGBoost models on the extracted features.
- **Evaluation**: Evaluating model performance using cross-validation and test set metrics.

### Deployment with Gradio
- **Interface Creation**: Creating an interactive web interface using Gradio for real-time paraphrase detection.
- **Demo**: Providing a user-friendly demo for testing the model on new sentence pairs.

