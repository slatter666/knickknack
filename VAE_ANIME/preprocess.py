"""
  * FileName: preprocess.py
  * Author:   Slatter
  * Date:     2023/3/16 21:35
  * Description:  
"""
import os
from tqdm import tqdm
from PIL import Image


def process(img_size: int = 64):
    origin_dir = '../dataset/anime/raw/images'
    process_dir = '../dataset/anime/processed/images'
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

    raw_img_paths = os.listdir(origin_dir)
    print('Number of Raw Images:', len(raw_img_paths))

    for path in tqdm(raw_img_paths):
        img_path = os.path.join(origin_dir, path)
        new_path = os.path.join(process_dir, path)

        # resize image
        img = Image.open(img_path)
        new_img = img.resize((img_size, img_size))
        new_img.save(new_path, quality=100)

    processed_img_paths = os.listdir(origin_dir)
    print('Number of Processed Images:', len(processed_img_paths))


if __name__ == '__main__':
    process()
