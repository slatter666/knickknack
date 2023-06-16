"""
  * FileName: process.py
  * Author:   Slatter
  * Date:     2023/6/11 19:47
  * Description:  
"""
import json
import os
from typing import List, Union

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def extract(file_path: str, write_dir: str):
    en = []
    zh = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            en.append(line['english'])
            zh.append(line['chinese'])

    en_save_path = os.path.join(write_dir, 'en.txt')
    zh_save_path = os.path.join(write_dir, 'zh.txt')
    with open(en_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(en))

    with open(zh_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(zh))


def build_tokenizer(files: Union[List[str], str], save_path: str, vocab_size=30000):
    if isinstance(files, str):
        files = [files]

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<SOS>', '<EOS>', '<PAD>', '<UNK>']
    )

    tokenizer.train(files, trainer=trainer)

    # And Save it
    tokenizer.save(save_path, pretty=True)

    tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)


if __name__ == '__main__':
    file_path = '../dataset/nmt/en-zh/translation2019zh_train.json'
    extract(file_path, write_dir=os.path.dirname(file_path))

    en_txt_path = '../dataset/nmt/en-zh/en.txt'
    zh_txt_path = '../dataset/nmt/en-zh/zh.txt'

    en_tokenizer_path = '../dataset/nmt/en-zh/en-tokenizer.json'
    zh_tokenizer_path = '../dataset/nmt/en-zh/zh-tokenizer.json'

    build_tokenizer(en_txt_path, en_tokenizer_path)
    build_tokenizer(zh_txt_path, zh_tokenizer_path)
