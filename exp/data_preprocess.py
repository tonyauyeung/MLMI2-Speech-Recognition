import torch
import torchaudio
import re
import json
import os
from tqdm import tqdm


def func(input):
    pattern = re.compile(r'^\w+_|\.\w+$')
    result = pattern.sub('', input)
    return result


def preprocess(mode):
    with open('exp/'+mode+'.json', 'r') as f:
        data = json.load(f)
    dics = {}
    for key in tqdm(data.keys()):
        dic = {}
        path = data[key]['wav']
        wav = torchaudio.load(path)
        fbank = torchaudio.compliance.kaldi.fbank(wav[0])
        save_path = '/rds/user/ro352/hpc-work/MLMI2/exp/TIMIT/data/{}/{}'.format(mode, data[key]['spk_id'])
        if not os.path.exists(save_path):
          os.makedirs(save_path)
        save_path += '/' + func(key) + '.pt'
        torch.save(fbank, save_path)
        dic['fbank'] = save_path
        dic['spk_id'] = data[key]['spk_id']
        dic['duration'] = data[key]['duration']
        dic['phn'] = data[key]['phn']
        dics[key] = dic
    with open('exp/{}_fbank.json'.format(mode), 'w') as json_file:
      json.dump(dics, json_file, indent=4)


if __name__ == '__main__':
    preprocess('train_origin')
    # preprocess('test')
    # preprocess('dev')