from dataloader import get_dataloader
import torch
import numpy as np
import json
from tqdm import tqdm
# import matplotlib.pyplot as plt


def exersice2_1():
    with open('exp/phone_map.txt', 'r') as file:
        data_lines = file.readlines()
    values = ['_']
    for line in data_lines:
        key, value = line.strip().split(':')
        values.append(value.strip())
    values = np.unique(values)[1:]
    with open('exp/vocab_39.txt', 'w') as file:
    # Write each string on a new line
        for string in values:
            file.write(f"{string}\n")

def exercise2_2():
    with open('exp/train.json', 'r') as f:
        data = json.load(f)
    with open('exp/vocab_39.txt', 'r') as f:
        vocabs = f.readlines()
    vocabs = [i.strip() for i in vocabs]
    counts = dict(zip(vocabs, [0] * len(vocabs)))
    for key in tqdm(data.keys()):
        phns = data[key]['phn'].split(' ')
        for phn in phns:
            counts[phn] += 1
    with open('plots/excercise2_2.json', 'w') as json_file:
        json.dump(counts, json_file, indent=4)
    # plt.figure(figsize=(16, 8))
    # plt.bar(counts.keys, counts.values)
    # plt.savefig('plots/exercise2_1.jpg')


if __name__ == '__main__':
    exersice2_1()
    exercise2_2()