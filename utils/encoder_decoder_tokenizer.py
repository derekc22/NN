import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # (B, S, D)

# Input preprocessor for seq2seq generation
class Seq2SeqInputPreprocessor(nn.Module):
    def __init__(self, tokenizer_name, d_model, max_len=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, source_texts, target_texts, max_length=128):
        src = self.tokenizer(source_texts, padding="max_length", truncation=True,
                             max_length=max_length, return_tensors="pt")
        tgt = self.tokenizer(target_texts, padding="max_length", truncation=True,
                             max_length=max_length + 1, return_tensors="pt")

        decoder_input_ids = tgt["input_ids"][:, :-1]                     # Shifted right
        labels = tgt["input_ids"][:, 1:]                                 # Target for loss
        decoder_attention_mask = tgt["attention_mask"][:, :-1]

        return {
            "input_ids": src["input_ids"],                               # (B, S_src)
            "attention_mask": src["attention_mask"],                     # (B, S_src)
            "decoder_input_ids": decoder_input_ids,                      # (B, S_tgt - 1)
            "decoder_attention_mask": decoder_attention_mask,            # (B, S_tgt - 1)
            "labels": labels                                              # (B, S_tgt - 1)
        }

    def forward(self, input_ids, decoder_input_ids):
        src_emb = self.positional_encoding(self.embedding(input_ids))              # (B, S_src, D)
        tgt_emb = self.positional_encoding(self.embedding(decoder_input_ids))      # (B, S_tgt, D)
        return src_emb, tgt_emb