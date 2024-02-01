import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout, mlp_layers, bidirection):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=bidirection)
        h_dims = hidden_dims * 2 if bidirection else hidden_dims
        if mlp_layers == 1:
            self.proj = nn.Linear(h_dims, out_dims)
        else:
            self.proj = nn.Sequential(nn.Linear(h_dims, out_dims), nn.ReLU(), nn.Linear(out_dims, out_dims))
            # TODO: 3 layers
            # for _ in range(mlp_layers - 1):
            #     self.proj.add_module(nn.ReLU())

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        hidden = self.dropout(hidden)
        output = self.proj(hidden)
        return output
