"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/5/26 22:57
  * Description:  
"""
import torch
from torch import nn, optim
from transformers import BertModel
import pytorch_lightning as pl


class PoetryBert(pl.LightningModule):
    def __init__(self, pretrain_path: str, pad_token_id, lr=2e-5):
        super(PoetryBert, self).__init__()
        self.save_hyperparameters()
        self.model = BertModel.from_pretrained(pretrain_path)
        # set BertModel.config.is_decoder=True so that it will generate correct attention mask
        self.model.config.is_decoder = True

        self.fc = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)

        self.pad_token_id = pad_token_id
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        :param input_ids: input tokens' id    (batch, max_seq_len)
        :param attention_mask: encoder attention mask  (max_len, max_seq_len)
        :param token_type_ids: (batch, max_seq_len)
        :return: (batch, max_seq_len, vocab_size)
        """
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        hidden_outputs = encoder_outputs.last_hidden_state  # (batch, max_seq_len, hidden_size)
        final_output = self.fc(hidden_outputs)
        return final_output

    def generate(self, tokenizer, head='', temperature=1, top_p=0.90, max_len=128):
        device = self.model.device

        punctuation = ['，', '。', '？', '；']
        head_index = 0
        is_head_list = True if isinstance(head, list) else False
        if is_head_list:
            tokens = tokenizer.encode(head[head_index])
        else:
            tokens = tokenizer.encode(head)

        # remove [SEP] but reserve [CLS]
        tokens = tokens[0:-1]
        break_flag = True

        while len(tokens) <= max_len and break_flag:
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            attention_mask = torch.tensor([[1] * len(tokens)], device=device)
            logits = self.forward(input_ids, attention_mask)   # (1, tokens_len, vocab_size)
            logits = logits[0, -1, :]  # (vocab_size)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1).item()

            if tokenizer.convert_ids_to_tokens(next_token) in ['[UNK]', '[CLS]', '[MASK]', '[PAD]']:
                continue

            if is_head_list:
                if tokenizer.convert_ids_to_tokens(next_token) in punctuation:
                    head_index += 1
                    if head_index < len(head):
                        head_token = tokenizer.encode(head[head_index])[1]
                        new_token = [next_token, head_token]
                    else:
                        new_token = [next_token]
                        break_flag = False

                else:
                    new_token = [next_token]
            else:
                new_token = [next_token]

            if tokenizer.convert_ids_to_tokens(next_token) == '[SEP]':
                break

            tokens += new_token

        return "".join(tokenizer.convert_ids_to_tokens(tokens[1:]))    # remove [CLS]

    def training_step(self, batch, batch_idx):
        logits = self.forward(**batch)   # (batch, max_seq_len, vocab_size)
        logits = logits.permute(0, 2, 1)  # (batch, vocab_size, max_seq_len)

        label_tokens = batch['input_ids'][:, 1:]   # (batch, max_seq_len - 1)
        bsz = label_tokens.size(0)
        t = torch.full((bsz, 1), self.pad_token_id, device=label_tokens.device)
        label_tokens = torch.concat([label_tokens, t], dim=1)  # (batch, max_seq_len)

        loss = self.criterion(logits, label_tokens)
        self.log('loss', loss)
        return loss

    def training_epoch_end(self, step_output):
        output = [x['loss'].item() for x in step_output]
        loss = sum(output) / len(output)
        print('Train loss: {:.2f}'.format(loss))

    @staticmethod
    def sample_top_p(probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))   # normalize
        next_token = torch.multinomial(probs_sort, num_samples=1)   # sample
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token.item()


if __name__ == '__main__':
    from dataset import PoemLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = '../dataset/poetry/poetry.txt'
    tokenizer_path = '/data2/daijincheng/pretrain/bert-base-chinese'
    loader = PoemLoader(file_path, tokenizer_path, batch_size=8, device=device)
    train_loader = loader.train_dataloader()
    patch = next(iter(train_loader))

    model_path = '/data2/daijincheng/pretrain/bert-base-chinese'
    model = PoetryBert(model_path, loader.pad_token_id).to(device)
    out = model(**patch)
