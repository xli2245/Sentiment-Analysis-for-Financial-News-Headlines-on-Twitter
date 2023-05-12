#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import random
import torch
import numpy as np
import pandas as pd 

import transformers
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaForSequenceClassification

from trainers import crossEntropyLossTrainer, weightedCrossEntropyLossTrainer, diceCrossEntropyLossTrainer

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# ori 123, 44, 1234
def seed_everything(selected_seed):
    torch.manual_seed(selected_seed)
    torch.cuda.manual_seed(selected_seed)
    np.random.seed(selected_seed)
    random.seed(selected_seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # get total accuracy and accuracy for each class
    acc_total = accuracy_score(labels, predictions)
    matrix = confusion_matrix(labels, predictions)
    accuracy_0, accuracy_1, accuracy_2 = matrix.diagonal()/matrix.sum(axis=1)
    # get recall and precision for each class
    precision_class, recall_class, F1_class, _ = precision_recall_fscore_support(labels, predictions)
    precision_0, precision_1, precision_2 = precision_class
    recall_0, recall_1, recall_2 = recall_class
    F1_0, F1_1, F1_2 = F1_class
    # get recall and precision for the whole class (micro)
    precision_micro, recall_micro, F1_micro, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    precision_macro, recall_macro, F1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    F1_sum = F1_micro + F1_macro

    res = {'accuracy_0' : accuracy_0, 'accuracy_1': accuracy_1, 'accuracy_2': accuracy_2,\
            'precision_0' : precision_0, 'precision_1': precision_1, 'precision_2': precision_2,\
            'recall_0' : recall_0, 'recall_1': recall_1, 'recall_2': recall_2,\
            'F1_0': F1_0, 'F1_1': F1_1, 'F1_2': F1_2,\
            'precision_macro': precision_macro, 'recall_macro': recall_macro, \
            'F1_micro': F1_micro, 'F1_macro': F1_macro, 'F1_sum': F1_sum, 'accuracy': acc_total}
    return res
    

def train_model(model, dataset_train, dataset_val, output_dir, best_metric):
    args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=30,
            load_best_model_at_end=True,
            metric_for_best_model=best_metric,
        )

    trainer = crossEntropyLossTrainer(
            model=model,
            args=args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            compute_metrics=compute_metrics)

    # trainer = weightedCrossEntropyLossTrainer(
    #         model=model,
    #         args=args,
    #         train_dataset=dataset_train,
    #         eval_dataset=dataset_val,
    #         compute_metrics=compute_metrics
    # )

    # trainer = diceCrossEntropyLossTrainer(
    #         model=model,
    #         args=args,
    #         train_dataset=dataset_train,
    #         eval_dataset=dataset_val,
    #         compute_metrics=compute_metrics
    # )

    trainer.train() 
    return model, trainer


def get_model(model_name = 'bert', num_labels=3):
    model_name_zoo = ['bert', 'fin-bert', 'albert', 'roberta']
    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'fin-bert':
        model = BertForSequenceClassification.from_pretrained('fin-bert',num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('fin-bert')
    elif model_name == 'albert':
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v2',num_labels=num_labels)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif model_name == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name == 'deberta':
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base',num_labels=num_labels)
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    else:
        raise ValueError("the selected model is not included ...")
    return model, tokenizer


if __name__ == '__main__':
    selected_seed = 42
    seed_everything(selected_seed)

    tag = 'up_new'

    train_data_path = "./data/sent_train_" + tag + ".csv"
    valid_data_path = "./data/sent_valid_" + tag + ".csv"
    test_data_path = "./data/sent_test_" + tag + ".csv"

    train_df = pd.read_csv(train_data_path)
    valid_df = pd.read_csv(valid_data_path)
    test_df = pd.read_csv(test_data_path)

    train_df = train_df[train_df.iloc[:, 0].apply(lambda x: isinstance(x, str))]
    train_df.reset_index(drop=True, inplace=True)
    train_df.head()

    valid_df = valid_df[valid_df.iloc[:, 0].apply(lambda x: isinstance(x, str))]
    valid_df.reset_index(drop=True, inplace=True)
    valid_df.head()
    
    test_df = test_df[test_df.iloc[:, 0].apply(lambda x: isinstance(x, str))]
    test_df.reset_index(drop=True, inplace=True)
    test_df.head()

    
    dataset_train = Dataset.from_pandas(train_df)
    dataset_val = Dataset.from_pandas(valid_df)
    dataset_test = Dataset.from_pandas(test_df)

    model_name_zoo = ['bert', 'fin-bert', 'albert', 'roberta', 'deberta']
    model_name = model_name_zoo[3]
    best_metric_zoo = ['eval_loss', 'eval_accuracy', 'eval_F1_micro', 'eval_F1_macro', 'eval_F1_sum']
    best_metric = best_metric_zoo[4]
    model, tokenizer = get_model(model_name, 3)

    output_dir = './results/' + tag + '/' + model_name + '/' + best_metric
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    dataset_train = dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_val = dataset_val.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
    dataset_test = dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length' , max_length=128), batched=True)

    # dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # dataset_val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    # dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model, trainer = train_model(model, dataset_train, dataset_val, output_dir, best_metric)
    model.eval()
    print(trainer.predict(dataset_test).metrics)


