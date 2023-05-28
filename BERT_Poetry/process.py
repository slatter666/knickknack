"""
  * FileName: process.py
  * Author:   Slatter
  * Date:     2023/5/26 19:27
  * Description:
"""
import glob
import json
import os
from typing import List

from hanziconv import HanziConv
from tqdm import tqdm


def process_text(poet: List[str]) -> List[str]:
    """
    将繁体转换为简体, 丢掉有问题的诗
    :param poet: 繁体诗
    :return: 简体诗
    """
    error_char = ["□", "�"]
    conv = [HanziConv.toSimplified(piece) for piece in poet]  # 先转为简体
    res = []
    for piece in tqdm(conv):
        if len(piece) == 0:
            # 去掉空串
            continue
        else:
            # 先判断一下有没有乱码，如果有乱码直接扔掉这条句子
            flag = False
            for ch in piece:
                if ch in error_char:
                    flag = True
                    break

            if flag:
                continue
            else:
                res.append(piece)

    return res


def process(file_dir: str, write_path: str, limit=128):
    raw = []
    for file_path in tqdm(glob.glob(os.path.join(file_dir, '*.json'))):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for piece in data:
                if len(piece['paragraphs']) != 0:
                    t = "".join(piece['paragraphs'])
                    if len(t) <= limit:
                        raw.append(t)

    processed = process_text(raw)
    print(len(processed))

    with open(write_path, 'w') as f:
        f.write('\n'.join(processed))


if __name__ == '__main__':
    file_dir = '../dataset/poetry/json'
    write_path = '../dataset/poetry/poetry.txt'
    process(file_dir, write_path)
