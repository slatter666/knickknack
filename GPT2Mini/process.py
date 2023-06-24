"""
  * FileName: process.py
  * Author:   Slatter
  * Date:     2023/6/22 10:46
  * Description:  
"""
import json
from tqdm import tqdm
from typing import List


def merge_data(file_paths: List[str], store_path: str):
    rec = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as input_file:
            texts = json.load(input_file)
            print('pieces:', len(texts))
            rec += texts

    with open(store_path, 'w', encoding='utf-8') as output_file:
        json.dump(rec, output_file, indent=2, ensure_ascii=False)


def process_data(file_path: str, store_path: str, max_len: int = 1024, mode: str = 'chunk', window_size: int = 10):
    """
    Args:
        file_path: raw text data path(json format)
        store_path: processed data's store path
        max_len: maximum sequence length
        mode: process method: truncate, chunk, slide_window
        window_size: only used in mode `slide_window`
    Returns:
        None: Store new data to store_path
    """
    assert mode in ['truncate', 'chunk', 'slide_window'], "Error: mode must in ['truncate', 'chunk', 'slide_window']"
    with open(file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
        data = [piece['text'] for piece in data]

    print("Original pieces:", len(data))

    res = []
    if mode == 'truncate':
        for text in tqdm(data):
            res.append(text[:max_len])
    elif mode == 'chunk':
        for text in tqdm(data):
            if len(text) <= max_len:
                res.append(text)
            else:
                chunk_size = len(text) // max_len
                for i in range(chunk_size + 1):
                    if (i + 1) * chunk_size <= len(text):
                        res.append(text[i * max_len: (i + 1) * max_len])
                    elif len(text) - i * chunk_size > 0.15 * max_len:  # 最后部分至少要有0.15 * max_len才保留
                        res.append(text[-max_len:])
    elif mode == 'slide_window':
        for text in tqdm(data):
            left = 0
            while left <= len(text):
                piece = text[left: left + max_len]
                if len(piece) == max_len:
                    res.append(piece)
                elif len(piece) > 0.15 * max_len:
                    res.append(text[-max_len:])

                left += max_len - window_size
    else:
        raise NotImplementedError

    res = [{'text': text} for text in res]
    print("Processed pieces:", len(res))
    with open(store_path, 'w', encoding='utf-8') as output_file:
        json.dump(res, output_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    file_paths = ['../dataset/corpus/prose.json', '../dataset/corpus/prose2.json']
    merge_path = '../dataset/corpus/merge.json'
    store_path = '../dataset/corpus/processed2.json'

    merge_data(file_paths, merge_path)
    process_data(merge_path, store_path, max_len=512, mode='slide_window')
