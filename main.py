import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

from model import HLG
from train_evaluate import train, evaluate
from utils import get_device

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True


def main(config, model_id, data_processor, load_data):
    if not os.path.exists(config.output_dir + model_id):
        os.makedirs(config.output_dir + model_id)

    if not os.path.exists(config.cache_dir + model_id):
        os.makedirs(config.cache_dir + model_id)

    output_model_file = os.path.join(config.output_dir, model_id, WEIGHTS_NAME)
    output_config_file = os.path.join(config.output_dir, model_id, CONFIG_NAME)

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    processor = data_processor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Train and dev
    if config.do_train:
        train_dataloader, train_examples_len = load_data(
            config.data_dir, processor, config.max_seq_length, config.train_batch_size, "train", config.num_workers)
        dev_dataloader, _ = load_data(
            config.data_dir, processor, config.max_seq_length, config.dev_batch_size, "dev", config.num_workers)

        num_train_optimization_steps = int(
            train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * (
                                               config.num_train_epochs + 1)

        model = HLG.from_pretrained(config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())

        bert_parameters = [(n, p) for n, p in param_optimizer if 'bert' in n]
        model_parameters = [(n, p) for n, p in param_optimizer if 'bert' not in n]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_parameters if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': config.bert_learning_rate},
            {'params': [p for n, p in bert_parameters if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': config.bert_learning_rate},
            {'params': [p for n, p in model_parameters if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': config.model_learning_rate},
            {'params': [p for n, p in model_parameters if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': config.model_learning_rate}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             warmup=config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        """ 损失函数准备 """
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer, criterion,
              config.gradient_accumulation_steps, device, label_list, output_model_file, output_config_file,
              config.early_stop)
    else:
        bert_config = BertConfig(output_config_file)
        model = HLG(bert_config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

    """ Test """
    test_dataloader, _ = load_data(
        config.data_dir, processor, config.max_seq_length, config.test_batch_size, "test", config.num_workers)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    test_loss, test_acc, test_report, test_auc = evaluate(model, test_dataloader, criterion, device, label_list)

    print("-------------- Test -------------")
    print(f'\t  Loss: {test_loss: .3f} | Acc: {test_acc * 100: .3f} % | AUC:{test_auc}')

    for label in label_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
    print_list = ['macro avg', 'weighted avg']

    for label in print_list:
        print('\t {}: Precision: {} | recall: {} | f1 score: {}'.format(
            label, test_report[label]['precision'], test_report[label]['recall'], test_report[label]['f1-score']))
