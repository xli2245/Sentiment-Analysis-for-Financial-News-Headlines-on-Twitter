# Sentiment Analysis for Financial News Headlines on Twitter
### Teammate: Xue Li, Minyi Dai, Qiyuan Chen
Sentiment analysis of news headlines plays a vital role in various practical applications, including assessing public opinion, conducting market analysis, and monitoring media coverage. In this research, we employ natural language processing models to perform sentiment analysis on financial news headlines, with a focus on the Twitter financial news dataset. We evaluate model performance using both micro-F1 and macro-F1 scores. To enhance the sentiment analysis pipeline, we implement various model selection strategies, utilize different models, and apply distinct data engineering methods. As a result, the refined pipeline achieves a 7% improvement in performance compared to the baseline pipeline, paving the way for the development of a more precise sentiment analysis tool.
## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Sentiment classification framework](#Sentiment-classification-framework)
  - [Environment](#environment)
  - [Model running](#model-running)
  - [Pretrained model weight](#model-weight)
- [Results](#Results)
## Setup
Clone this repo:
```
git clone https://github.com/xli2245/CS839-sentiment-classification
```
## Dataset
The original dataset is downloaded from [Twitter Final News dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment). All the used data have been provided under the data folder.

## Sentiment classification framework
### Environment
The model training, validation and testing are performed using the [Monai Docker](https://hub.docker.com/r/projectmonai/monai).
### Model running
1.  Model training
```
python train.py
```
2. Model validation / testing
```
python predict.py
```
### Pretrained model weight
For the fin-bert and roberta-base pretrained model, the download links are provided under each folder. The downloaded pretrained models for the others like (BERT, DeBERTa ...) can be also be found in the google drive folder. The name is "model.tar.gz".

To unzip the model weight
```
tar -xvf ./model.tar.gz
mv ./model/* ./
rm -rf ./model
```

## Results
All the results and training logs could be found under the results folder.
