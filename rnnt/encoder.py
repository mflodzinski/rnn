import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout, bidirectional=True):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.custom_proj = CustomProjection(hidden_size, output_size)

    def forward(self, inputs, input_lengths):
        # in case of batch size = 1
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        assert inputs.dim() == 3
        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths.cpu(), batch_first=True)
                
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.custom_proj(outputs)
        return logits, hidden

class CustomProjection(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CustomProjection, self).__init__()
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        forward_output, backward_output = x.chunk(2, dim=-1)
        forward_projected = self.linear1(forward_output)
        backward_projected = self.linear2(backward_output)
        return forward_projected + backward_projected
    
def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            n_layers=config.enc.n_layers,
            dropout=config.enc.dropout,
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError
