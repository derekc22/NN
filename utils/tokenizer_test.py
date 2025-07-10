import torch
from tokenizer import Seq2SeqInputPreprocessor

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 512
preprocessor = Seq2SeqInputPreprocessor("bert-base-uncased", d_model).to(device)

# Replace with your actual dataset
source_texts = [
    "Write a short story about a dragon.",
    "Describe a rainy day."
]

target_texts = [
    "Once upon a time, a dragon lived in a forest and guarded treasure.",
    "The rain fell gently, tapping against the windowpane all morning."
]

# Tokenize and prepare inputs
batch = preprocessor.tokenize(source_texts, target_texts, max_length=64)

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
decoder_input_ids = batch["decoder_input_ids"].to(device)
decoder_attention_mask = batch["decoder_attention_mask"].to(device)
labels = batch["labels"].to(device)

# Get embedded inputs
src_emb, tgt_emb = preprocessor(input_ids, decoder_input_ids)

# Final model call
# output = model(
#     src_emb,                   # (B, S_src, D)
#     tgt_emb,                   # (B, S_tgt, D)
#     attention_mask,            # (B, S_src)
#     decoder_attention_mask,    # (B, S_tgt)
#     labels                     # (B, S_tgt)
# )