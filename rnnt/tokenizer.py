import pandas as pd


class CharTokenizer:
    def __init__(self, transcript_path):
        self.special_tokens = {
            "pad": "_",
            "sos": "<",
            "eos": ">",
            "phi": "|",
        }
        self.vocab = self.get_vocab(transcript_path)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def get_vocab(self, transcript_path):
        df = pd.read_csv(transcript_path)
        all_txt = df["transcript"].str.cat(sep="")
        vocab = sorted(list(set(all_txt)))
        vocab.insert(0, self.special_tokens["pad"])
        vocab += list(self.special_tokens.values())[1:]
        return vocab

    def ids2tokens(self, ids):
        if not ids:
            return []

        if isinstance(ids[0], list):  # Check if it's a list of lists
            return [[self.itos[i] for i in sublist] for sublist in ids]
        else:  # It's a single list
            return [self.itos[i] for i in ids]

    def tokens2ids(self, tokens):
        if not tokens:
            return []

        if isinstance(tokens[0], list):  # Check if it's a list of lists
            return [[self.stoi[s] for s in sublist] for sublist in tokens]
        else:  # It's a single list
            return [self.stoi[s] for s in tokens]