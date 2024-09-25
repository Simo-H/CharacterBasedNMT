import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def padding_collate_fn(batch, pad_token_html, pad_token_json):

    src_batch, tgt_batch = list(zip(*batch))
    src_batch = pad_sequence(src_batch, padding_value=pad_token_html)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_token_json)
    return src_batch, tgt_batch


class HTML_JSON_Dataset(Dataset):
    def __init__(self, html_data, json_data):
        self.html_data = html_data
        self.json_data = json_data

    def __len__(self):
        return len(self.html_data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.html_data[idx]), torch.LongTensor(self.json_data[idx])