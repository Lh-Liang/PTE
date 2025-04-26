import os
import csv
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import itertools
from collections import OrderedDict
from transformers.models.bert import BertForSequenceClassification, BertForMaskedLM
from dataset import BBCDataset, CTDataset, SMSDataset, AgNewsDataset, collate_fn



class Config():
    def __init__(self):
        self.batch_size = 8
        self.epochs = 1
        self.log_freq = 100
        self.data_path = '../../data/sms/333/test.csv'
        self.model_path = '../../bert-base-uncased'
        self.weight_path = '/home/llh/PythonProject/PTE/test/pet-liheng/ckpts/2024-09-16T13-59-41/sms.pt'
        self.model1_path = '/home/llh/PythonProject/PTE/test/pet-wp/ckpts/2024-09-17T13-53-10/epoch375_step1500_acc0.979167.pt'
        self.weight1_path = '/home/llh/PythonProject/PTE/test/pet-wp/ckpts/2024-09-17T13-53-10/epoch375_step1500_acc0.979167_weight.pt'
        self.eval_freq = 50
        self.max_seq_length = 512
        self.pattern_ids = 3
        self.labels_num = 1100
        self.Dataset = SMSDataset(self.data_path, self.model_path, self.weight_path, self.pattern_ids, self.max_seq_length, self.labels_num)



class PteModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained(config.model_path).requires_grad_(True)
        self.weight = nn.Parameter(torch.tensor([[1.] * config.labels_num] * 2)).to(config.device).requires_grad_(True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        weight = self.weight

        return out.logits, weight #torch.Size([B, max_length, 30522])

class PteCriterion(nn.Module):
    def __init__(self, config, m2c_tensor, filler_len):
        super().__init__()
        self.config = config
        self.m2c = m2c_tensor.to(config.device)
        self.filler_len = filler_len.to(config.device)

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor, weight) -> torch.Tensor:
        cls_logits = logits[torch.max(torch.zeros_like(self.m2c), self.m2c)]
        cls_logits = cls_logits * (self.m2c > 0).float()
        cls_logits = (weight * cls_logits).sum(axis=1) / self.filler_len
        return cls_logits

    def predict(self, cls_logits):
        cls_logits = cls_logits.cpu()
        cls_logits = cls_logits.detach()
        cls_logits = cls_logits.numpy()
        predictions = np.argmax(cls_logits, axis=1)

        return predictions

    def forward(self, logits, mlm_labels, weight):
        masked_logits = logits[mlm_labels >= 0]  # [B, vocab_size]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml, weight) for ml in
                                  masked_logits])  # [B, label_size]
        predictions = self.predict(cls_logits)
        return predictions

def setup_training(config):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    return config

def prepare_data_loader(config, num_workers=1):
    train_dataset = config.Dataset
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)
    return train_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len

def prepare_model_and_optimizer(config, model, m2c_tensor, filler_len, total_step):
    model = model
    model.model.load_state_dict(torch.load(config.model1_path, map_location=config.device))
    model.weight = torch.load(config.weight1_path, map_location=config.device)['weight']
    model.to(config.device)
    criterion = PteCriterion(config, m2c_tensor, filler_len)
    criterion.to(config.device)
    return model, criterion

def trainer():
    config = Config()
    config = setup_training(config)
    train_iter, m2c_tensor, filler_len = prepare_data_loader(config)
    total_step = config.epochs * len(train_iter)
    model = PteModel(config)
    model, criterion = prepare_model_and_optimizer(config, model, m2c_tensor, filler_len, total_step)

    all_predictions = []
    all_labels = []
    for i, batch in enumerate(train_iter):
        model.eval()
        input_ids, token_type_ids, attention_mask, mlm_labels, labels = [w.to(config.device) for w in batch]

        logit, weight = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        predictions = criterion(logit, mlm_labels, weight)
        all_predictions.append(predictions)

        labels1 = labels.cpu()
        labels1 = labels1.detach()
        labels1 = labels1.numpy()
        labels1 = list(itertools.chain.from_iterable(labels1))
        all_labels.append(labels1)

    all_predictions = list(itertools.chain.from_iterable(all_predictions))
    all_labels = list(itertools.chain.from_iterable(all_labels))

    acc = accuracy_score(all_labels, all_predictions)
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    micro_precision = precision_score(all_labels, all_predictions, average='micro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    micro_recall = recall_score(all_labels, all_predictions, average='micro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')

    print('acc:' + str(acc))
    print('macro precision:' + str(macro_precision))
    print('micro precision:' + str(micro_precision))
    print('macro recall:' + str(macro_recall))
    print('micro recall:' + str(micro_recall))
    print('macro f1:' + str(macro_f1))
    print('micro f1:' + str(micro_f1))

if __name__ == '__main__':
    trainer()
