import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout):
        super(BaseDecoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=vocab_size-1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.embedding = batch_index_to_one_hot
        self.vocab_size = vocab_size
        self.proj =  nn.Linear(hidden_size, output_size)

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs, self.vocab_size)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.proj(outputs)
        return outputs, hidden


def build_decoder(config):
    if config.dec.type == 'lstm':
        return BaseDecoder(
            hidden_size=config.dec.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.dec.output_size,
            n_layers=config.dec.n_layers,
            dropout=config.enc.dropout,
        )
    else:
        raise NotImplementedError


def batch_index_to_one_hot(batch_indices, vocab_size, label_smoothing=0.0):
    one_hot_size = vocab_size - 1
    mask = (batch_indices == one_hot_size)
    adjusted_indices = torch.where(mask, torch.zeros_like(batch_indices), batch_indices)
    adjusted_indices = adjusted_indices.long()
    one_hot_batch = F.one_hot(adjusted_indices, one_hot_size)
    one_hot_batch[mask] = 0

    if label_smoothing > 0.0:
        confidence = 1.0 - label_smoothing
        smoothed_labels = one_hot_batch * confidence + label_smoothing / one_hot_size
        one_hot_batch = smoothed_labels

    return one_hot_batch.float()
