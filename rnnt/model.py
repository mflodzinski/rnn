import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import build_encoder
from decoder import build_decoder
from warprnnt_pytorch import RNNTLoss

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        # self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        #self.tanh = nn.Tanh()
        #self.project_layer = nn.Linear(input_size, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()
        
        outputs = enc_state + dec_state
        #concat_state = torch.cat((enc_state, dec_state), dim=-1)
        #outputs = self.tanh(outputs)
        #outputs = self.project_layer(outputs)
        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        self.config = config
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
            )

        self.crit = RNNTLoss(blank=config.blank,reduction='mean')

    def forward(self, inputs, inputs_length, targets, targets_length):

        enc_state, _ = self.encoder(inputs, inputs_length)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=self.config.blank)
        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))

        logits = self.joint(enc_state, dec_state)
        loss = self.crit(logits, targets.contiguous(), inputs_length, targets_length)
        return loss


    def recognize(self, inputs, input_lengths):
        results = list()
        blank = self.config.blank
        zero_token = torch.LongTensor([[blank]])
        if inputs.dim() == 2:
            batch_size = 1
        else:
            batch_size = inputs.dim()

        f, _ = self.encoder(inputs, input_lengths)
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(inputs, lengths):
            token_list = []
            u = 0
            t = 0
            gu, hidden = self.decoder(zero_token)
            umax = self.config.max_length

            while t < lengths and u < umax:
                h = self.joint(inputs[t].view(-1), gu.view(-1))
                out = F.log_softmax(h, dim=0)
                _, pred = torch.max(out, dim=0)
                pred = int(pred.item())

                if pred != blank:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if zero_token.is_cuda:
                        token = token.cuda()
                    gu, hidden = self.decoder(token, hidden=hidden)
                    u += 1
                else:
                    t += 1

            return token_list

        for i in range(batch_size):
            decoded_seq = decode(f[i], input_lengths[i])
            results.append(decoded_seq)

        return results