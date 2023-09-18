"""
  * FileName: inference.py
  * Author:   Slatter
  * Date:     2023/9/18 11:06
  * Description:  
"""
import torch.cuda
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel


def infer(ckpt_path: str, image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare processor
    image_processor = AutoImageProcessor.from_pretrained("/data2/daijincheng/pretrain/vit-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    # prepare model
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    img = Image.open(image_path)
    inputs = image_processor(img, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    out = out[:out.index(".") + 1]
    print(out)


if __name__ == '__main__':
    ckpt_path = "/data2/daijincheng/knickknack/VIT_GPT2/caption_ckpt/checkpoint-25285"
    test_path = "/data2/daijincheng/knickknack/VIT_GPT2/test_images/ride_horse.jpeg"
    infer(ckpt_path, test_path)
