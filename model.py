import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn


class HLG(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(HLG, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.c1_to_w1 = GCNLayer(config.hidden_size)
        self.c1_to_c2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.w1_to_s1 = GCNLayer(config.hidden_size)
        self.w1_to_w2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.s1_to_w2 = GCNLayer(config.hidden_size)
        self.w2_to_c2 = GCNLayer(config.hidden_size)
        self.relu = nn.ReLU()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, graphs, attention_mask, char_mask, word_mask, sentence_mask, token_type_ids=None):
        char_reps, _ = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        # Padding
        char_reps = F.pad(char_reps, [0, 0, 0, graphs[0].size(1) - char_reps.size(1)])

        w1 = self.c1_to_w1(torch.transpose(graphs[0], -1, -2), char_reps, char_mask)
        s1 = self.w1_to_s1(torch.transpose(graphs[1], -1, -2), w1, word_mask)
        w2_1 = self.s1_to_w2(graphs[1], s1, sentence_mask)
        w2_2 = self.relu(self.w1_to_w2(w1))
        c2_1 = self.w2_to_c2(graphs[0], w2_1 + w2_2, word_mask)
        c2_2 = self.relu(self.c1_to_c2(char_reps))

        output = torch.sum(c2_1 + c2_2, dim=1)
        return self.classifier(output)


class GCNLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GCNLayer, self).__init__()
        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, adj, nodes_hidden, nodes_mask):
        # norm adj
        scale = adj.sum(dim=-1)
        scale[scale == 0] = 1
        adj /= scale.unsqueeze(-1).repeat(1, 1, adj.shape[-1])
        nodes_hidden = nodes_hidden * nodes_mask.unsqueeze(-1)
        nodes_hidden = self.gcn(torch.matmul(adj, nodes_hidden))
        return self.relu(nodes_hidden)
