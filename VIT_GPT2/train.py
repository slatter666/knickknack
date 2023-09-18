"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/8/17 19:46
  * Description:  
"""
from dataclasses import dataclass, field

from transformers import HfArgumentParser, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

from Text2Image.trainer_utils import *
from data_utils import *


@dataclass
class ModelArguments:
    encoder_path: str = field(default="/data2/daijincheng/pretrain/vit-base-patch16-224",
                              metadata={"help": "Path to the encoder model"})
    decoder_path: str = field(default="/data2/daijincheng/pretrain/gpt2-medium",
                              metadata={"help": "Path to the decoder model"})


@dataclass
class DataArguments:
    mapping_file: str = field(default="/data2/daijincheng/knickknack/dataset/flickr8k/captions.txt",
                              metadata={"help": "Path to the image2text mapping file"})
    image_dir: str = field(default="/data2/daijincheng/knickknack/dataset/flickr8k/Images",
                           metadata={"help": "Path to the image folder"})


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="checkpoint", metadata={"help": "Path to the checkpoint"})


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # prepare processor
    image_processor = AutoProcessor.from_pretrained(model_args.encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.decoder_path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    # prepare dataset
    dataset = CaptionDataset(data_args.mapping_file, data_args.image_dir)
    data_collator = CaptionCollator(tokenizer, image_processor)

    # prepare model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        model_args.encoder_path, model_args.decoder_path
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # training
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    train()
