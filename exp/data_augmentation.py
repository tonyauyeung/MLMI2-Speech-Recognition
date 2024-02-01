import torch
from torchaudio.transforms import SpeedPerturbation
import torchaudio
import re
import json
import os
from tqdm import tqdm
import random


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio
    

def func(input):
    pattern = re.compile(r'^\w+_|\.\w+$')
    result = pattern.sub('', input)
    return result


def pertube(waveform, rate):
    speed_perturb = SpeedPerturbation(16000, [rate])
    # speed_perturb = RandomSpeedChange(rate)
    return speed_perturb(waveform)


def save_fbank(fbank, save_path, spk_id, duration, phn):
    dic = {}
    dic['fbank'] = save_path
    dic['spk_id'] = spk_id
    dic['duration'] = duration
    dic['phn'] = phn
    torch.save(fbank, save_path)
    return dic


def preprocess(mode):
    with open('exp/'+mode+'.json', 'r') as f:
        data = json.load(f)
    dics = {}
    for key in tqdm(data.keys()):
        dic = {}
        path = data[key]['wav']
        wav = torchaudio.load(path)
        fast_wav = pertube(wav[0], 1.1)
        slow_wav = pertube(wav[0], 0.9)
        fbank = torchaudio.compliance.kaldi.fbank(wav[0])
        fbank_fast = torchaudio.compliance.kaldi.fbank(fast_wav[0])
        fbank_slow = torchaudio.compliance.kaldi.fbank(slow_wav[0])
        save_path = '/rds/user/ro352/hpc-work/MLMI2/exp/TIMIT/data/{}/{}'.format(mode, data[key]['spk_id'])
        save_path_origin = save_path + '/' + func(key) + '.pt'
        save_path_fast = save_path + '/' + func(key) + 'fast.pt'
        save_path_slow = save_path + '/' + func(key) + 'slow.pt'
        key_fast = key + '_fast'
        key_slow = key + '_slow'
        dics[key] = save_fbank(fbank, save_path_origin, data[key]['spk_id'], 
                               data[key]['duration'], data[key]['phn'])
        dics[key_fast] = save_fbank(fbank_fast, save_path_fast, data[key]['spk_id'], 
                                    data[key]['duration'], data[key]['phn'])
        dics[key_slow] = save_fbank(fbank_slow, save_path_slow, data[key]['spk_id'], 
                                    data[key]['duration'], data[key]['phn'])
        # if not os.path.exists(save_path):
        #   os.makedirs(save_path)
        # save_path += '/' + func(key) + '.pt'
        # torch.save(fbank, save_path)
        # dic['fbank'] = save_path
        # dic['spk_id'] = data[key]['spk_id']
        # dic['duration'] = data[key]['duration']
        # dic['phn'] = data[key]['phn']
        # dics[key] = dic
    with open('exp/{}_fbank.json'.format(mode), 'w') as json_file:
        json.dump(dics, json_file, indent=4)


if __name__ == '__main__':
    preprocess('train')