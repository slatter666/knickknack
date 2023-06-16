"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/6/8 13:55
  * Description:  
"""
from arch import *
from utils import *
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, maxlen=512):
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


class NMT(pl.LightningModule):
    def __init__(self, src_tokenizer: SelfTokenizer, tgt_tokenizer: SelfTokenizer, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'relu', norm_first: bool = False, lr: float = 1e-4,
                 warmup_steps: int = 4000):
        super(NMT, self).__init__()
        self.save_hyperparameters()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_vocab_size = src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = tgt_tokenizer.get_vocab_size()

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )

        self.fc = nn.Linear(d_model, self.tgt_vocab_size)

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id('<PAD>'))

    def warmup_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        else:
            return (1 - 1e-6) ** (current_step - self.warmup_steps)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
            src_mask: (batch, src_len, src_len)
            tgt_mask: (batch, tgt_len, tgt_len)

        Returns:
            logits (batch, tgt_len, tgt_vocab_size)
        """
        src_embeded = self.pos_embedding(self.src_embedding(src))
        tgt_embeded = self.pos_embedding(self.tgt_embedding(tgt))

        outs = self.transformer(src_embeded, tgt_embeded, src_mask, tgt_mask)
        return self.fc(outs)

    def encode(self, src, src_mask):
        src_embeded = self.pos_embedding(self.src_embedding(src))
        return self.transformer.encoder(src_embeded, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt_embeded = self.pos_embedding(self.tgt_embedding(tgt))
        return self.transformer.decoder(tgt_embeded, memory, src_mask, tgt_mask)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.warmup_lambda)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        """
        Args: batch contains src, src_mask, tgt, tgt_mask
            src: (batch, src_len)
            src_mask: (batch, src_len, src_len)
            tgt: (batch, tgt_len)
            tgt_mask: (batch, tgt_len, tgt_len)
        Returns:

        """
        src, src_mask, tgt, tgt_mask = batch
        tgt_mask = tgt_mask[:, :-1, :-1]  # (batch, tgt_len - 1, tgt_len - 1) cause we have to shift one

        tgt_input = tgt[:, :-1]  # (batch, tgt_len - 1)
        tgt_out = tgt[:, 1:]  # Shift one to the right  (batch, tgt_len - 1)

        logits = self.forward(src, tgt_input, src_mask, tgt_mask)  # (batch, tgt_len - 1, tgt_vocab_size)
        train_loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        self.log('train loss', train_loss, sync_dist=True)
        return train_loss

    def training_epoch_end(self, step_output):
        output = [x['loss'].item() for x in step_output]
        loss = sum(output) / len(output)
        print('Train loss: {:.3f}'.format(loss))

    def validation_step(self, batch, batch_idx):
        src, src_mask, tgt, tgt_mask = batch
        tgt_mask = tgt_mask[:, :-1, :-1]  # (batch, tgt_len - 1, tgt_len - 1) cause we have to shift one

        tgt_input = tgt[:, :-1]  # (batch, tgt_len - 1)
        tgt_out = tgt[:, 1:]  # Shift one to the right  (batch, tgt_len - 1)

        logits = self.forward(src, tgt_input, src_mask, tgt_mask)  # (batch, tgt_len - 1, tgt_vocab_size)
        val_loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        return val_loss

    def validation_epoch_end(self, step_output):
        loss = sum(step_output) / len(step_output)
        self.log('val loss', loss, sync_dist=True)
        print('Valid loss: {:.3f}'.format(loss))

    def greedy_decode(self, src: List[str], max_len: int = 128, device=torch.device('cpu')):
        """
        Args:
            src: list of string
            max_len: maximum length of sequence to be generated
            device: cpu or gpu, default: cpu
        Returns:

        """
        self.eval()
        with torch.no_grad():
            bsz = len(src)

            encode_src = self.src_tokenizer.encode_batch(src)
            src, src_padding_mask = encode_src['input_ids'], encode_src['attention_mask']
            src = torch.tensor(src, dtype=torch.long, device=device)  # (batch, src_len)
            src_mask = torch.tensor(src_padding_mask, device=device).unsqueeze(dim=1)  # (batch, 1, src_len)

            enc_outputs = self.encode(src, src_mask)

            tokens = torch.full((bsz, 1), self.tgt_tokenizer.token_to_id('<SOS>'), dtype=torch.long,
                                device=device)  # (batch, 1)

            for i in range(max_len):
                tgt_mask = self.transformer.generate_square_subsequent_mask(tokens.size(1)).unsqueeze(
                    dim=0).to(device)  # (1, tgt_len, tgt_len)
                logits = self.decode(tokens, enc_outputs, src_mask, tgt_mask)  # (batch, tgt_len, d_model)
                dec_outputs = self.fc(logits[:, -1, :])  # (batch, tgt_vocab_size)

                next_token = torch.argmax(dec_outputs, dim=-1, keepdim=True)  # (batch, 1)
                tokens = torch.cat([tokens, next_token], dim=1)  # (batch, tgt_len + 1)

            decoded = []
            for i, t in enumerate(tokens.tolist()):
                try:
                    t = t[: t.index(self.tgt_tokenizer.token_to_id('<EOS>'))]
                except ValueError:
                    pass
                decoded.append(self.tgt_tokenizer.decode(t).strip())
            return decoded

    # Under development, do not use in real world scenarios
    def beam_search(self, src: List[str], beam_size: int = 5, max_len: int = 128, device=torch.device('cpu')):
        """
        beam search implementation for batch sequence
        Args:
            src: list of string
            beam_size: beam search size
            max_len: maximum length of sequence to be generated
            device: cpu or gpu, default: cpu
        Returns:

        """
        self.eval()
        with torch.no_grad():
            bsz = len(src)
            PAD_TOKEN, EOS_TOKEN = self.tgt_tokenizer.token_to_id('<PAD>'), self.tgt_tokenizer.token_to_id('<EOS>')

            encode_src = self.src_tokenizer.encode_batch(src)
            src, src_padding_mask = encode_src['input_ids'], encode_src['attention_mask']
            src = torch.tensor(src, dtype=torch.long, device=device)  # (batch, src_len)
            src_mask = torch.tensor(src_padding_mask, device=device).unsqueeze(dim=1)  # (batch, 1, src_len)

            enc_outputs = self.encode(src, src_mask)  # (batch, src_len, d_model)
            _, src_len, d_model = enc_outputs.size()
            src_mask = src_mask.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, 1, src_len)
            enc_outputs = enc_outputs.unsqueeze(dim=1).expand(-1, beam_size, -1, -1).reshape(-1, src_len,
                                                                                             d_model)  # (batch * beam_size, src_len, d_model)

            # start token, scores,
            beam_tokens = torch.full((bsz, beam_size, 1), self.tgt_tokenizer.token_to_id('<SOS>'), dtype=torch.long,
                                     device=device)  # (batch, beam_size, 1)
            beam_scores = torch.zeros((bsz, beam_size), dtype=torch.float, device=device)  # (batch, beam_size)
            beam_scores[:, 1:] = float('-inf')  # Set scores of all beams except the first one to -inf

            for i in range(max_len):
                prev_tokens = beam_tokens.view(bsz * beam_size, -1)  # (batch * beam_size, tgt_len)
                tgt_mask = self.transformer.generate_square_subsequent_mask(prev_tokens.size(1)).unsqueeze(
                    dim=0).to(device)  # (1, tgt_len, tgt_len)
                logits = self.decode(prev_tokens, enc_outputs, src_mask,
                                     tgt_mask)  # (batch * beam_size, tgt_len, d_model)

                # Get top-k token predictions and their corresponding scores
                log_probs = F.log_softmax(self.fc(logits[:, -1, :]), dim=-1)  # (batch * beam_size, tgt_vocab_size)
                scores = beam_scores.view(-1, 1) + log_probs  # (batch * beam_size, tgt_vocab_size)
                scores = scores.view(bsz, -1)  # (batch, beam_size * tgt_vocab_size)

                # topk_scores: (batch, beam_size)  topk_indices: (batch, beam_size)
                topk_scores, topk_indices = scores.topk(k=beam_size, dim=1, largest=True, sorted=True)

                # Calculate beam indices and token indices
                beam_indices = topk_indices.div(self.tgt_vocab_size, rounding_mode='floor')
                token_indices = topk_indices % self.tgt_vocab_size

                # Update beam scores and beam tokens
                beam_scores = topk_scores    # (batch, beam_size)
                beam_tokens = torch.cat([beam_tokens[torch.arange(bsz).unsqueeze(dim=1), beam_indices], token_indices.unsqueeze(dim=-1)], dim=-1)  # (batch, beam_size, tgt_len + 1)

                # # Select the best sequence from each batch
                # best_sequence_indices = beam_scores.argmax(dim=-1)
                # best_sequences = beam_tokens[torch.arange(batch_size), best_sequence_indices]

            decoded = []
            beam_tokens = beam_tokens.view(bsz * beam_size, -1)
            for i, t in enumerate(beam_tokens.tolist()):
                try:
                    t = t[: t.index(self.tgt_tokenizer.token_to_id('<EOS>'))]
                except ValueError:
                    pass
                decoded.append(self.tgt_tokenizer.decode(t).strip())

            return decoded
