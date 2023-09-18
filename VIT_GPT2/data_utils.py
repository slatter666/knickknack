"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/9/16 18:30
  * Description:  
"""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor


class CaptionDataset(Dataset):
    def __init__(self, mapping_file: str, image_dir: str):
        self.image_dir = image_dir
        self.mapping = pd.read_csv(mapping_file)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        piece = self.mapping.iloc[idx]
        image_name, caption = piece['image'], piece['caption']

        image = Image.open(os.path.join(self.image_dir, image_name))
        return {
            "image": image,
            "caption": caption
        }


class CaptionCollator:
    def __init__(self, tokenizer: AutoTokenizer, image_processor: AutoProcessor, max_len: int = 128):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __call__(self, instances):
        images = [example["image"] for example in instances]
        caption = [example["caption"] for example in instances]

        image_inputs = self.image_processor(images, return_tensors='pt')
        caption_inputs = self.tokenizer(caption, padding='max_length', max_length=self.max_len, return_tensors='pt')
        labels = caption_inputs['input_ids']
        labels[labels.eq(self.tokenizer.pad_token_id)] = -100
        return {
            "pixel_values": image_inputs["pixel_values"],
            "labels": labels
        }
