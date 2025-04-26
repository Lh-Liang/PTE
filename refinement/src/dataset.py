import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

class BBCDataset(Dataset):
    def __init__(self, data_path, model_path, weight_path, pattern_id, max_length, labels_num):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = tokenizer.get_vocab()
        vocab.pop(tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        pre_weight = torch.load(weight_path)['weight'].cpu().detach().numpy()

        top_indices = []
        for row in pre_weight:
            row_vocab = []
            indices = np.argsort(row)[-labels_num:]
            for i in list(indices):
                row_vocab.append(vocab_list[i])
            top_indices.append(row_vocab)

        VERBALIZER = {
            "business": top_indices[0],
            "tech": top_indices[1],
            "politics": top_indices[2],
            "sport": top_indices[3],
            "entertainment": top_indices[4]
        }
        VERBALIZER_INDEX_LABEL = {
            "business": 0,
            "tech": 1,
            "politics": 2,
            "sport": 3,
            "entertainment": 4
        }
        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()} # convert label to int

        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # load tokenizer
        self.mask = self.tokenizer.mask_token  # load tokenizer
        self.mask_id = self.tokenizer.mask_token_id  # load tokenizer

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                ArticleId, text, label= row
                if idx == 0:
                    continue
                if label not in VERBALIZER_INDEX_LABEL: continue
                example = [text, VERBALIZER_INDEX_LABEL[label]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 # init the value of [len(VERBALIZER_LABEL), max_num_verbalizers] to -1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor #[label_size, max_num_verbalizers]

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len # The number of tokens in different Verbalizers may vary

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids


    def encode(self, text):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = ['theme:', self.mask, '.', text, '.']
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels


    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label
class AgNewsDataset(Dataset):
    def __init__(self, data_path, model_path, weight_path, pattern_id, max_length, labels_num):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = tokenizer.get_vocab()
        vocab.pop(tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        pre_weight = torch.load(weight_path)['weight'].cpu().detach().numpy()

        top_indices = []
        for row in pre_weight:
            row_vocab = []
            indices = np.argsort(row)[-labels_num:]
            for i in list(indices):
                row_vocab.append(vocab_list[i])
            top_indices.append(row_vocab)

        VERBALIZER = {
            "1": top_indices[0],
            "2": top_indices[1],
            "3": top_indices[2],
            "4": top_indices[3]
        }
        VERBALIZER_INDEX_LABEL = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3
        }
        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                if label not in VERBALIZER_INDEX_LABEL: continue
                example = [text_a, text_b, VERBALIZER_INDEX_LABEL[label]]
                self.examples.append(example)


    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 #[len(VERBALIZER_LABEL), max_num_verbalizers]所有值都为-1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids

    def encode(self, text_a, text_b):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text_a, text_b]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'News:', text_a, text_b]
        elif self.pattern_id == 2:
            prompt_text = [text_a, '(', self.mask, ')', text_b]
        elif self.pattern_id == 3:
            prompt_text = [text_a, text_b, '(', self.mask, ')']
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text_a, text_b]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text_a, text_b]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels


    def __getitem__(self, idx):
        text_a, text_b, label = self.examples[idx]
        feature = self.encode(text_a, text_b)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label
class CTDataset(Dataset):
    def __init__(self, data_path, model_path, weight_path, pattern_id, max_length, labels_num):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = tokenizer.get_vocab()
        vocab.pop(tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        pre_weight = torch.load(weight_path)['weight'].cpu().detach().numpy()

        top_indices = []
        for row in pre_weight:
            row_vocab = []
            indices = np.argsort(row)[-labels_num:]
            for i in list(indices):
                row_vocab.append(vocab_list[i])
            top_indices.append(row_vocab)

        VERBALIZER = {
            "other_cyberbullying": top_indices[0],
            "not_cyberbullying": top_indices[1],
            "gender": top_indices[2],
            "religion": top_indices[3],
            "age": top_indices[4],
            "ethnicity": top_indices[5]
        }
        VERBALIZER_INDEX_LABEL = {
            "other_cyberbullying": 0,
            "not_cyberbullying": 1,
            "gender": 2,
            "religion": 3,
            "age": 4,
            "ethnicity": 5
        }
        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                text, label= row
                if idx == 0:
                    continue
                if label not in VERBALIZER_INDEX_LABEL: continue
                example = [text, VERBALIZER_INDEX_LABEL[label]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 #[len(VERBALIZER_LABEL), max_num_verbalizers]所有值都为-1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids


    def encode(self, text):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'theme:', text]
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels


    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label
class SMSDataset(Dataset):

    def __init__(self, data_path, model_path, weight_path, pattern_id, max_length, labels_num):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab = tokenizer.get_vocab()
        vocab.pop(tokenizer.unk_token)
        vocab_list = sorted(vocab.keys())

        pre_weight = torch.load(weight_path)['weight'].cpu().detach().numpy()

        top_indices = []
        for row in pre_weight:
            row_vocab = []
            indices = np.argsort(row)[-labels_num:]
            for i in list(indices):
                row_vocab.append(vocab_list[i])
            top_indices.append(row_vocab)

        VERBALIZER = {
            "Spam": top_indices[0],
            "Non-Spam": top_indices[1]
        }
        VERBALIZER_INDEX_LABEL = {
            "Spam": 0,
            "Non-Spam": 1
        }
        self.VERBALIZER_LABEL = {VERBALIZER_INDEX_LABEL[k]: v for k, v in VERBALIZER.items()}

        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        self.max_length = max_length
        self.pattern_id = pattern_id
        self.max_num_verbalizers = max(len(v) for k, v in self.VERBALIZER_LABEL.items())
        self.m2c_tensor = self._build_m2c_tensor()
        self.filler_len = self._build_filler_len()

        with open(data_path, encoding='Windows-1252') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                no, text, label= row
                if idx == 0:
                    continue
                if label not in VERBALIZER_INDEX_LABEL: continue
                example = [text, VERBALIZER_INDEX_LABEL[label]]
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _build_m2c_tensor(self):
        m2c_tensor = torch.ones([len(self.VERBALIZER_LABEL), self.max_num_verbalizers], dtype=torch.long) * -1 #[len(VERBALIZER_LABEL), max_num_verbalizers]所有值都为-1
        for label_idx, verbalizers in self.VERBALIZER_LABEL.items():
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self.tokenizer.encode(verbalizer, add_special_tokens=False)[0]
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _build_filler_len(self):
        filler_len = torch.tensor([len(verbalizers) for label, verbalizers in self.VERBALIZER_LABEL.items()],
                                  dtype=torch.float)
        return filler_len

    def get_verbalization_ids(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids


    def encode(self, text):

        if self.pattern_id == 0:
            prompt_text = [self.mask, ':', text]
        elif self.pattern_id == 1:
            prompt_text = [self.mask, 'sort:', text]
        elif self.pattern_id == 2:
            prompt_text = [text, '(', self.mask, ')']
        elif self.pattern_id == 3:
            prompt_text = ['(', self.mask, ')', text]
        elif self.pattern_id == 4:
            prompt_text = ['[ Category:', self.mask, ']', text]
        elif self.pattern_id == 5:
            prompt_text = [self.mask, '-', text]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

        feature = self.tokenizer(''.join(prompt_text),
                                 add_special_tokens=False,
                                 max_length=self.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
        return feature

    def get_mlm_labels(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels


    def __getitem__(self, idx):
        text, label = self.examples[idx]
        feature = self.encode(text)
        input_ids = feature.input_ids
        token_type_ids = feature.token_type_ids
        attention_mask = feature.attention_mask

        # get_mask_positions
        mlm_labels = self.get_mlm_labels(input_ids.tolist()[0])
        return input_ids, token_type_ids, attention_mask, mlm_labels, label

def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, mlm_labels, labels = zip(*batch)
    input_ids = torch.stack([w.squeeze() for w in input_ids])
    token_type_ids = torch.stack([w.squeeze() for w in token_type_ids])
    attention_mask = torch.stack([w.squeeze() for w in attention_mask])
    mlm_labels = torch.stack([torch.Tensor(mlm_label).long() for mlm_label in mlm_labels])
    labels = torch.stack([torch.Tensor([label]).long() for label in labels])

    return input_ids, token_type_ids, attention_mask, mlm_labels, labels
