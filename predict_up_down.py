#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import random
import torch
import json
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

from train import compute_metrics
from process_log import read_dicts_from_txt, find_min_max_values, sort_checkpoint_subfolders, remove_train_info


def load_saved_model(model_name = 'bert', num_labels=3, saved_path=None):
    if not saved_path:
        raise ValueError("the model path is None, please specify it!")
    model_name_zoo = ['bert', 'fin-bert', 'albert', 'roberta']
    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained(saved_path,num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'fin-bert':
        model = BertForSequenceClassification.from_pretrained(saved_path,num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('fin-bert')
    elif model_name == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(saved_path,num_labels=num_labels)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif model_name == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(saved_path,num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name == 'deberta':
        model = DebertaForSequenceClassification.from_pretrained(saved_path,num_labels=num_labels)
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    else:
        raise ValueError("the selected model is not included ...")
    return model, tokenizer


if __name__ == '__main__':
    tag = 'up_new'
    test_data_path = "./data/sent_test_" + tag + ".csv"
    test_df = pd.read_csv(test_data_path)
    
    test_df = test_df[test_df.iloc[:, 0].apply(lambda x: isinstance(x, str))]
    test_df.reset_index(drop=True, inplace=True)
    test_df.head()

    dataset_test = Dataset.from_pandas(test_df)

    model_name_zoo = ['bert', 'fin-bert', 'albert', 'roberta', 'deberta']
    model_name = model_name_zoo[3]
    best_metric_zoo = ['eval_loss', 'eval_accuracy', 'eval_F1_micro', 'eval_F1_macro', 'eval_F1_sum']
    best_metric = best_metric_zoo[4]

    model_folder = './results/' + tag + '/' + model_name + '/' + best_metric
    input_log = './log_' + tag + '.txt'
    log_path = model_folder + '/log.txt'
    remove_train_info(input_log, log_path)
    saved_test_log_path = model_folder + '/log_analysis.txt'
    result_dict = read_dicts_from_txt(log_path)
    idx_info = find_min_max_values(result_dict)

    csv_results = []
    target_cols = []

    with open(saved_test_log_path, 'a') as file:
        for key in idx_info.keys():
            if key in best_metric_zoo or key == 'last_epoch':
                target_idx = idx_info[key]['index']
                file.write(f'================= {key} at epoch {target_idx} =================\n')
                print(f'================= {key} =================')
                idx_to_path = sort_checkpoint_subfolders(model_folder)
                saved_path = model_folder + '/checkpoint-' + str(idx_to_path[target_idx])
                model, tokenizer = load_saved_model(model_name, 3, saved_path)
                dataset_test = dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length' , max_length=128), batched=True)
                # dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
                dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
                model.eval()
                trainer = crossEntropyLossTrainer(model=model, compute_metrics=compute_metrics)
                pred_results = trainer.predict(dataset_test).metrics
                pred_results['saved_epoch'] = target_idx
                pred_results['saved_best_matix'] = key
                if not target_cols:
                    for col_name in pred_results.keys():
                        if 'F1' in col_name or 'accuracy' in col_name or 'precision' in col_name or 'recall' in col_name or 'saved_' in col_name:
                            target_cols.append(col_name)
                csv_results.append(pred_results)
                file.write(json.dumps(pred_results))
    df = pd.DataFrame(csv_results, columns=target_cols)
    df.to_csv(model_folder + "/output.csv", index=False)
    cmd = 'find ' + model_folder + ' -name "checkpoint-*" -exec rm -rf {} \\;'
    os.system(cmd)
    cmd = 'find ' + model_folder + ' -name "runs" -exec rm -rf {} \\;'
    os.system(cmd)

