import torch
import torch.nn.functional as F
from rnnt.rest.beam_search import *

def GreedyDecode(model, inputs, input_lengths):
    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)
    blank = 30
    zero_token = torch.LongTensor([[blank]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
    results = []
    batch_size = inputs.size(0)

    def decode(inputs, lengths):
        log_prob = 0
        token_list = []
        umax = 100
        u = 0
        t = 0
        gu, hidden = model.decoder(zero_token)
        while t < lengths and u < umax:
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = F.log_softmax(h, dim=0)
            prob, pred = torch.max(out, dim=0)
            pred = int(pred.item())

            log_prob += prob.item()

            if pred != blank:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = model.decoder(token, hidden=hidden)
                u += 1
            else:
                t += 1
        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results



if __name__ =="__main__":
    import argparse
    import yaml
    from utils import AttrDict
    from model import Transducer
    from tokenizer import CharTokenizer
    from data import SentenceDataLoader
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', type=str, default='config/config.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    model_path = 'timit/rnnt/2enc1dec_model.chkpt'
    checkpoint = torch.load(model_path)
    model = Transducer(config.model)

    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer('files/tokenizer.json')

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])
    model.eval()

    batch_size = 2
    test_data = SentenceDataLoader('files/core_test_set.csv', tokenizer, config.data, batch_size)
    training_data = SentenceDataLoader('files/core_train_subset.csv', tokenizer, config.data, batch_size)
    path = "_output_transcriptions.txt"  

    for step, (inputs, inputs_length, targets, targets_length) in enumerate(test_data):
        preds = GreedyDecode(model, inputs, inputs_length)
        preds = tokenizer.ids2tokens(preds)
        targs = tokenizer.ids2tokens(targets.tolist())
        for l, t in zip(preds, targs):
            sentence = "".join(l)
            print(sentence)
            with open(path, 'a') as file:
                file.write(sentence + " -> " + "".join(t).replace('!', '') + '\n')  
