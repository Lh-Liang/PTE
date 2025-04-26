import torch
import torch.nn as nn
from transformers.models.bert import BertForMaskedLM
import numpy as np


class PteModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        np.random.seed(config.seed)
        self.model = BertForMaskedLM.from_pretrained(config.model_path).requires_grad_(True)
        self.weight = nn.Parameter(torch.tensor(np.random.rand(config.classes_num, 30521))).requires_grad_(True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        weight = self.weight

        return out.logits, weight # torch.Size([B, max_length, 30522])


class PteCriterion(nn.Module):
    def __init__(self, config, m2c_tensor, filler_len):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.m2c = m2c_tensor.to(config.device) # [label_size, max_num_verbalizers]
        self.filler_len = filler_len.to(config.device) # [label_size]

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor, weight) -> torch.Tensor:


        # m2c [label_size, max_num_verbalizers]
        # torch.max(torch.zeros_like(self.m2c), self.m2c) [label_size, max_num_verbalizers] Filter out the m2c whose median value is -1
        # cls_logits [label_size, max_num_verbalizers] the logits of label
        cls_logits = logits[torch.max(torch.zeros_like(self.m2c), self.m2c)]

        # self.m2c > 0 [label_size, max_num_verbalizers] Returns a bool value. In m2c, values greater than 0 return True, and values less than or equal to 0 return False
        # cls_logits * (self.m2c > 0)  [label_size, max_num_verbalizers]
        cls_logits = cls_logits * (self.m2c > 0).float()

        cls_logits = (weight * cls_logits).sum(axis=1) / self.filler_len # Average the verbalizers
        return cls_logits # [label_size] Represents the value of each current logits for each class

    def predict(self, cls_logits):
        cls_logits = cls_logits.cpu()
        cls_logits = cls_logits.detach()
        cls_logits = cls_logits.numpy()
        predictions = np.argmax(cls_logits, axis=1)

        return predictions


    def forward(self, logits, mlm_labels, labels, weight):

        masked_logits = logits[mlm_labels >= 0] # [B, vocab_size]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml, weight) for ml in
                                  masked_logits]) # [B, label_size]
        predictions = self.predict(cls_logits)
        loss = self.criterion(cls_logits, labels.view(-1))
        return loss, predictions





