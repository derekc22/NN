import torch
from utils.data import *
from utils.logger import load_config
from models.transformer import Transformer
from utils.encoder_decoder_tokenizer import Seq2SeqInputPreprocessor  
import argparse

parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()

config = load_config(args.config)
log_id = config['log_id']

system = config["system"]
device = system["device"]
save_fpath = system["save_fpath"]

specifications = config["specifications"]
context_window = specifications["context_window"]
sequence_length = specifications["sequence_length"]

architecture = config["architecture"]
transformer_architecture = architecture["transformer_architecture"]
ff_architecture = architecture["ff_architecture"]
d_model = transformer_architecture["d_model"]

mode = args.mode
pretrained = args.pretrained


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]
    transformer_hyperparameters = hyperparameters["transformer_hyperparameters"]
    ff_hyperparameters = hyperparameters["ff_hyperparameters"]
    
    # Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = Seq2SeqInputPreprocessor("bert-base-uncased", d_model, context_window).to(device)

    # Replace with your actual dataset
    source_texts = [
        "Write a short story about a dragon.",
        "Describe a rainy day."
    ]

    target_texts = [
        "Ecris une courte histoire sur un dragon.",
        "Decris une journee pluvieuse."
    ]

    # Tokenize and prepare inputs
    batch = preprocessor.tokenize(source_texts, target_texts, max_length=sequence_length)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Get embedded inputs
    src_emb, tgt_emb = preprocessor(input_ids, decoder_input_ids)

    # Final model call
    # output = model
    #     src_emb,                   # (B, S_src, D)
    #     tgt_emb,                   # (B, S_tgt, D)
    #     attention_mask,            # (B, S_src)
    #     decoder_attention_mask,    # (B, S_tgt)
    #     labels                     # (B, S_tgt)
    # )

    specifications["encoder_padding_mask"] = attention_mask
    specifications["decoder_padding_mask"] = decoder_attention_mask
    specifications["tokenizer"] = preprocessor

    if pretrained:
        pass

    else:
        transformer = Transformer(
            pretrained=False,
            training=True,
            device=device,
            architecture=transformer_architecture,
            ff_architecture=ff_architecture,
            hyperparameters=transformer_hyperparameters,
            ff_hyperparameters=ff_hyperparameters,
            save_fpath=save_fpath,
            specifications=specifications
        )


    epoch_plt, loss_plt = transformer.train(
        src_emb=src_emb.detach(), 
        tgt_emb=tgt_emb.detach(), 
        labels=labels.detach(),
        epochs=epochs, 
        save_params=True
    )
    
    if epoch_plt and show_plot:
        plot_training_results(epoch_plt, loss_plt, log_id)

# Testing mode
# else:
