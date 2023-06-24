"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/6/18 21:13
  * Description:  
"""

from typing import List

import pytorch_lightning as pl
import torch
from torch import optim
from transformers import BertTokenizer

from arch import *
from utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, maxlen=1024):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)  # (embed_size/2)
        pos = torch.arange(0, maxlen).view(maxlen, 1)  # (maxlen, 1)
        pos_embedding = torch.zeros(maxlen, embed_size)  # (maxlen, embed_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 偶数位置  (maxlen, embed_size/2)
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 奇数位置  (maxlen, embed_size/2)
        pos_embedding = pos_embedding.unsqueeze(dim=0)  # (1, maxlen, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        """
        Add positional embedding to token_embedding
        Args:
            token_embedding: (batch, src_len, embed_size)
        Returns:
            final_embedding: (batch, src_len, embed_size)
        """
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])  # 广播到batch


class GPT2Mini(pl.LightningModule):
    def __init__(self, tokenizer: BertTokenizer, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 num_layers: int = 6, dropout: float = 0.1, max_len=1024, activation: str = 'relu',
                 norm_first: bool = False, lr: float = 1e-4, warmup_steps: int = 4000):
        super(GPT2Mini, self).__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.vocab_size = tokenizer.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.vocab_size)
        )

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def warmup_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        else:
            return (1 - 3e-5) ** (current_step - self.warmup_steps)

    def forward(self, seq, seq_mask):
        """
        Args:
            seq: (batch, seq_len)
            seq_mask: (batch, seq_len, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        embeded = self.pos_embedding(self.embedding(seq))

        outs = self.transformer(embeded, seq_mask)
        return self.fc(outs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.warmup_lambda)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Args: batch contains seq, seq_mask
            seq: (batch, seq_len)
            seq_mask: (batch, seq_len, seq_len)
        Returns:
        """
        seq, seq_mask = batch

        seq_input = seq[:, :-1]  # (batch, seq_len - 1)
        seq_out = seq[:, 1:]  # Shift one to the right  (batch, seq_len - 1)

        seq_mask = seq_mask[:, :-1, :-1]  # (batch, seq_len - 1, seq_len - 1) cause we have to shift one token

        logits = self.forward(seq_input, seq_mask)  # (batch, seq_len - 1, vocab_size)
        train_loss = self.criterion(logits.reshape(-1, logits.size(-1)), seq_out.reshape(-1))
        self.log('train_loss', train_loss, sync_dist=True)
        return train_loss

    def training_epoch_end(self, step_output):
        output = [x['loss'].item() for x in step_output]
        loss = sum(output) / len(output)
        print('Train loss: {:.3f}'.format(loss))

    def validation_step(self, batch, batch_idx):
        seq, seq_mask = batch

        seq_input = seq[:, :-1]  # (batch, seq_len - 1)
        seq_out = seq[:, 1:]  # Shift one to the right  (batch, seq_len - 1)
        seq_mask = seq_mask[:, :-1, :-1]

        logits = self.forward(seq_input, seq_mask)  # (batch, seq_len - 1, vocab_size)

        avg_loss = 0
        avg_ppl = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            loss = self.criterion(logits[i], seq_out[i])
            avg_ppl += torch.exp(loss)
            avg_loss += loss

        avg_loss /= batch_size
        avg_ppl /= batch_size
        return avg_loss, avg_ppl

    def validation_epoch_end(self, step_output):
        loss, ppl = [], []
        for item in step_output:
            loss.append(item[0])
            ppl.append(item[1])

        avg_loss = sum(loss) / len(loss)
        avg_ppl = sum(ppl) / len(ppl)
        self.log('val_loss', avg_loss, sync_dist=True)
        self.log('val_ppl', avg_ppl, sync_dist=True)
        print('Valid loss: {:.3f}, ppl: {:.3f}'.format(avg_loss, avg_ppl))

    def generate(self, prompts: List[str], temperature: float = 0.8, top_p: float = 0.95, device=torch.device('cpu')):
        """
        Args:
            prompts: list of string, given some of the text then generate, you can also choose to generate from scratch
            temperature: temperature
            top_p: top_p
            device: cpu or gpu
        Returns:
            generation: list of string
        """
        self.eval()
        with torch.no_grad():
            bsz = len(prompts)
            prompt_tokens = [self.tokenizer.encode(x)[:-1] for x in prompts]  # remember to chop off the `[SEP]` token

            tokens = torch.full((bsz, self.max_len), self.tokenizer.pad_token_id, device=device).long()
            for idx, text in enumerate(prompt_tokens):
                tokens[idx, :len(text)] = torch.tensor(text).long()

            input_text_mask = tokens != self.tokenizer.pad_token_id  # (bsz, max_len)
            start_pos = min([len(t) for t in prompt_tokens])

            for cur_pos in range(start_pos, self.max_len):
                input_ids = tokens[:, :cur_pos]  # (batch, seq_len)
                # (1, seq_len, seq_len)
                attention_mask = self.transformer.generate_square_subsequent_mask(input_ids.size(1)).unsqueeze(dim=0).to(device)
                logits = self.forward(input_ids, attention_mask)[:, -1, :]  # (batch, vocab_size)

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = self.sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=-1)

                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token

            decoded = []
            for i, t in enumerate(tokens.tolist()):
                # cut to eos tok if any, here eos token is [SEP]
                try:
                    t = t[: t.index(self.tokenizer.sep_token_id)]
                except ValueError:
                    pass
                decoded.append(self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            return decoded

    @staticmethod
    def sample_top_p(probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    # def generation(self, prompts: List[str], temperature: float = 0.8, top_p: float = 0.95):
    #     bsz = len(prompts)
    #     cfg = self.model.config
    #     assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
    #
    #     prompt_tokens = [self.tokenizer.encode(x) for x in prompts]
    #
    #     min_prompt_size = min([len(t) for t in prompt_tokens])
    #     max_prompt_size = max([len(t) for t in prompt_tokens])
    #
    #     total_len = min(cfg.max_sequence_length, max_gen_len + max_prompt_size)
    #
    #     tokens = torch.full((bsz, total_len), cfg.pad_token_id, device=self.device).long()
    #     for k, t in enumerate(prompt_tokens):
    #         tokens[k, : len(t)] = torch.tensor(t).long()
    #
    #     input_text_mask = tokens != cfg.pad_token_id
    #     start_pos = min_prompt_size
    #     prev_pos = 0
    #     for cur_pos in range(start_pos, total_len):
    #         logits = self.model.forward(tokens[:, 0:cur_pos]).logits[:, -1]
    #         if temperature > 0:
    #             probs = torch.softmax(logits / temperature, dim=-1)
    #             next_token = sample_top_p(probs, top_p)
    #         else:
    #             next_token = torch.argmax(logits, dim=-1)
    #         next_token = next_token.reshape(-1)
    #         # only replace token if prompt has already been generated
    #         next_token = torch.where(
    #             input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    #         )
    #         tokens[:, cur_pos] = next_token
    #         prev_pos = cur_pos
    #
    #     decoded = []
    #     for i, t in enumerate(tokens.tolist()):
    #         # cut to max gen len
    #         t = t[: len(prompt_tokens[i]) + max_gen_len]
    #         # cut to eos tok if any
    #         try:
    #             t = t[: t.index(cfg.eos_token_id)]
    #         except ValueError:
    #             pass
    #         decoded.append(self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    #     return decoded
