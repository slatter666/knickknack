"""
  * FileName: utils.py
  * Author:   Slatter
  * Date:     2023/5/3 18:21
  * Description:  
"""
import numpy as np
import torch
from torchvision import transforms


def convert_image_to_natural(image):
    """
    scale image pixel values to [0, 1]
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # scale to [0, 1]
    ])

    image = transform(torch.clamp(image, -1, 1))  # scale to [-1, 1] then scale to [0, 1]
    return image


def tensor_image_to_PIL(image):
    """
    show tensor images
    :param image: (c, h, w)
    :return: (h, w, c)
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # (c, h, w) to (h, w, c)
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return transform(torch.clamp(image, -1, 1))
