import torch
import logging
import editdistance
import pandas as pd
import os

class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def computer_cer(preds, labels):
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total



def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    return n_params, enc, dec



def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)


def save_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }
    torch.save(checkpoint, save_name)


def shuffle_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        df_shuffled.to_csv(file_path, index=False)
        return file_path
    else:
        return "File not found. Please check the file path."

def check_row_count(df):
    counter = {}

    for idx, row in df.iterrows():
        text = row["text"]
        if text not in counter:
            counter[text] = 1
        else:
            counter[text] += 1

    sorted_dict_by_values = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}

    for k, v in sorted_dict_by_values.items():
        print(k, v, '\n')


def check_num_speakers(df):
    speakers = []
    for idx, row in df.iterrows():
        speaker = row['audio_path'].split('/')[3]
        if speaker not in speakers:
            speakers.append(speaker)
    print(speakers)
    print(len(speakers))

def check_num_audios(df):
    audios = []
    for idx, row in df.iterrows():
        audio = row['audio_path'].split('/')[4]
        if audio not in audios:
            audios.append(audio)
    print(audios)
    print(len(audios))

def check_num_texts(df):
    texts = []
    for idx, row in df.iterrows():
        text = row['text']
        if text not in texts:
            print(text)
            texts.append(text)
    print(len(texts))


def sort_csv_by_age(input_file, output_file):
    df = pd.read_csv(input_file)
    df_sorted = df.sort_values(by='duration')
    df_sorted.to_csv(output_file, index=False)


def extract_core_test_set():
    df = pd.read_csv('files/original_test_set.csv')
    df['audio_identifier'] = df['audio_path'].apply(lambda x: x.split('/')[-1].split('.')[0])

    # Define the list of speakers for the core test set
    speakers = ['DAB0', 'TAS1', 'JMP0', 'LLL0', 'BPM0', 'CMJ0', 'GRT0', 'JLN0',
                'WBT0', 'WEW0', 'LNT0', 'TLS0', 'KLT0', 'JDH0', 'NJM0', 'PAM0',
                'ELC0', 'PAS0', 'PKT0', 'JLM0', 'NLP0', 'MGD0', 'DHC0', 'MLD0']

    # Filter rows
    rows = []
    for _, row in df.iterrows():
        path = row['audio_path']
        speaker, sentence = path.split('/')[3], path.split('/')[4]
        print(speaker)
        if speaker[1:] in speakers and not sentence.startswith('SA'):
            rows.append(row)

    # Create a new DataFrame from the filtered rows
    new_df = pd.DataFrame(rows, columns=df.columns)

    # Save the new DataFrame to a CSV file without the index
    new_df.to_csv('files/core_test_set.csv', index=False)



if __name__ == '__main__':
    from tensorboard import notebook
    log_dir = 'timit/rnnt/log'
    notebook.start("--logdir=" + log_dir)
