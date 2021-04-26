import json
from dataclasses import dataclass

import torch
from torch.utils.data import (DataLoader, RandomSampler, Dataset)


@dataclass
class InputExample:
    guid: str
    input_ids: str
    token_type_ids: str
    word_lengths_1: str
    word_lengths_2: str
    word_lengths_3: str
    label: str


@dataclass
class InputFeatures:
    input_ids: list
    token_type_ids: list
    g_c: list
    g_w: list
    g_sw_w: list
    g_sw_s: list
    input_mask: list
    char_len: int
    word_len: int
    label_id: int


def convert_examples_to_features(examples, label_list, max_seq_length):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example {} of {}".format(ex_index, len(examples)))

        input_ids = json.loads(example.input_ids)
        token_type_ids = json.loads(example.token_type_ids)
        word_lengths_1 = json.loads(example.word_lengths_1)
        word_lengths_2 = json.loads(example.word_lengths_2)
        word_lengths_3 = json.loads(example.word_lengths_3)

        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:(max_seq_length - 2)]
            token_type_ids = token_type_ids[:(max_seq_length - 2)]

        # construct graph
        # 构建数组 [(from, to), (from, to)] ...
        word_seg_ids = []
        for segmentor_idx, word_lengths in enumerate([word_lengths_1, word_lengths_2, word_lengths_3]):
            start_idx = 0
            for w_len in word_lengths:
                word_seg_ids.append((start_idx, start_idx + w_len, segmentor_idx))
                start_idx += w_len

        # 不同分词器得到的同样的词作为相同的节点
        word_ids = sorted(list(set(map(lambda w: w[:2], word_seg_ids))))

        # 构建节点对
        g_c, g_w = [], []
        word_idx = 0
        for (word_idx, char_pos_pair) in enumerate(word_ids):
            char_start, char_end = char_pos_pair
            for char_idx in range(char_start, char_end):
                if char_idx >= max_seq_length - 2:
                    break
                if word_idx >= max_seq_length:
                    continue
                g_c.append(char_idx)
                g_w.append(word_idx)

        char_len = len(input_ids)
        word_len = word_idx

        # g_sw = word sentence graph
        g_sw_w, g_sw_s = [], []
        for word_seg_id in word_seg_ids:
            if len(g_sw_w) >= max_seq_length:
                break
            word_id = word_ids.index(word_seg_id[:2])
            if word_id >= max_seq_length:
                continue
            g_sw_w.append(word_id)
            g_sw_s.append(word_seg_id[2])

        input_mask = [1] * char_len
        padding = [0] * (max_seq_length - char_len)

        input_ids += padding
        token_type_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          g_c=g_c,
                          g_w=g_w,
                          g_sw_w=g_sw_w,
                          g_sw_s=g_sw_s,
                          char_len=char_len,
                          word_len=word_len,
                          input_mask=input_mask,
                          label_id=label_id))
    return features


def convert_features_to_tensors(features, batch_size, num_workers):
    data = GraphTensorDataset(features)
    sampler = RandomSampler(data)
    # sampler = DistributedSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, collate_fn=collate, num_workers=num_workers)
    return dataloader


class GraphTensorDataset(Dataset):
    def __init__(self, features):
        self.input_ids = []
        self.token_type_ids = []
        self.input_mask = []
        self.char_mask = []
        self.word_mask = []
        self.sentence_mask = []
        self.label_ids = []
        self.graphs = []

        for f in features:
            self.input_ids.append(torch.tensor(f.input_ids))
            self.token_type_ids.append(torch.tensor(f.token_type_ids))
            self.input_mask.append(torch.tensor(f.input_mask))
            self.label_ids.append(torch.tensor(f.label_id))
            self.graphs.append(
                [torch.tensor(f.g_c),
                 torch.tensor(f.g_w),
                 torch.tensor(f.g_sw_w),
                 torch.tensor(f.g_sw_s)])

            char_mask = torch.zeros(len(f.input_mask))
            char_mask[:torch.tensor(f.g_c).max()] = torch.ones(int(torch.tensor(f.g_c).max()))
            self.char_mask.append(char_mask)

            word_mask = torch.zeros(len(f.input_mask))
            word_mask[:torch.tensor(f.g_w).max()] = torch.ones(int(torch.tensor(f.g_w).max()))
            self.word_mask.append(word_mask)

            sentence_mask = torch.zeros(len(f.input_mask))
            sentence_mask[:3] = torch.tensor([1, 1, 1])
            self.sentence_mask.append(sentence_mask)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.graphs[idx], self.input_mask[idx], self.char_mask[
            idx], self.word_mask[idx], \
               self.sentence_mask[idx], self.label_ids[idx]

    def __len__(self):
        return len(self.input_ids)


def collate(batch):
    batch_data = list(zip(*batch))
    input_ids, token_type_ids, batch_graph, input_mask, char_mask, word_mask, sentence_mask, label_ids = batch_data
    max_sequence_len = input_ids[0].size(-1)
    batch_c2w, batch_w2s = [], []
    for (g_c, g_w, g_sw_w, g_sw_s) in batch_graph:
        # Construct C-W-S Graph
        c2w = torch.stack([g_c, g_w])
        c2w_graph_size = torch.Size([max_sequence_len, max_sequence_len])
        c2w_graph_item_values = torch.ones(g_c.size(0))
        c2w_graph = torch.sparse.IntTensor(c2w, c2w_graph_item_values, c2w_graph_size)

        w2s = torch.stack([g_sw_w, g_sw_s])
        w2s_graph_size = torch.Size([max_sequence_len, max_sequence_len])
        w2s_graph_item_values = torch.ones(g_sw_w.size(0))
        w2s_graph = torch.sparse.IntTensor(w2s, w2s_graph_item_values, w2s_graph_size)

        batch_c2w.append(c2w_graph.to_dense())
        batch_w2s.append(w2s_graph.to_dense())

    return torch.stack(input_ids), torch.stack(token_type_ids), torch.stack(batch_c2w), \
           torch.stack(batch_w2s), torch.stack(input_mask), torch.stack(char_mask), \
           torch.stack(word_mask), torch.stack(sentence_mask), torch.stack(label_ids)


def load_data(data_dir, processor, max_length, batch_size, data_type, num_workers=0):
    label_list = processor.get_labels()

    if data_type == "train":
        examples = processor.get_train_examples(data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif data_type == "test":
        examples = processor.get_test_examples(data_dir)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(examples, label_list, max_length)

    dataloader = convert_features_to_tensors(features, batch_size, num_workers)

    examples_len = len(examples)

    return dataloader, examples_len
