import torch
from utils.data import *
# from utils.ff_utils import *
from utils.logger import load_config
from models.transformer import Transformer
import argparse
from utils.encoder_decoder_tokenizer import Seq2SeqInputPreprocessor  


parser = argparse.ArgumentParser(description='Set run options')
parser.add_argument('--config', type=str, help='Specify config location')
parser.add_argument('--mode', type=str, help='Specify "train" or "test"')
parser.add_argument('--pretrained', action="store_true", help='Specify if model is pretrained')
args = parser.parse_args()

config = load_config(args.config)
log_id = config['log_id']

specs = config["specs"]
device_type = specs["device_type"] #torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = args.mode #specs["mode"]
pretrained = args.pretrained #specs["pretrained"]
input_feature_count = specs["input_feature_count"]

parameters = config["parameters"]
transformer_save_fpath = parameters["transformer_save_fpath"]
ff_save_fpath = parameters["ff_save_fpath"]
architecture = config["architecture"]
transformer_architecture = architecture["transformer_architecture"]
ff_architecture = architecture["ff_architecture"]


# Training mode
if mode == "train":
    train_config = config["train"]
    train_dataset_size = train_config["train_dataset_size"]
    epochs = train_config["epochs"]
    show_plot = train_config["show_loss_plot"]
    hyperparameters = config["hyperparameters"]
    transformer_hyperparameters = hyperparameters["transformer_hyperparameters"]
    ff_hyperparameters = hyperparameters["ff_hyperparameters"]
    
    # input_embedding = torch.rand(size=(5, 10, input_feature_count))
    # output_embedding = torch.rand(size=(5, 10, input_feature_count))
    

    # Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = Seq2SeqInputPreprocessor("bert-base-uncased", input_feature_count).to(device)

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
    batch = preprocessor.tokenize(source_texts, target_texts, max_length=64)

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

    transformer_architecture["encoder_padding_mask"] = attention_mask
    transformer_architecture["decoder_padding_mask"] = decoder_attention_mask
    transformer_architecture["vocab_size"] = preprocessor.vocab_size
    


    if pretrained:
        pass
        # ff = ff(
        #     pretrained=True,
        #     training=True,
        #     device_type=device_type,
        #     hyperparameters=hyperparameters,
        #     model_params=fetch_ff_params_from_file(device_type, save_fpath),
        #     save_fpath=save_fpath,
        # )
    else:
        transformer = Transformer(
            pretrained=False,
            training=True,
            device_type=device_type,
            hyperparameters=transformer_hyperparameters,
            ff_hyperparameters=ff_hyperparameters,
            architecture=transformer_architecture,
            ff_architecture=ff_architecture,
            input_feature_count=input_feature_count,
            save_fpath=transformer_save_fpath,
            ff_save_fpath=ff_save_fpath,
            tokenizer=preprocessor
        )

    # print(src_emb.shape)
    # print(tgt_emb.shape)
    # print(preprocessor.pad_token_id)
    # exit()
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
#     test_config = config["test"]
#     test_dataset_size = test_config["test_dataset_size"]
#     show_results = test_config["show_results"]

#     data_batch, label_batch = gen_matrix_stack(test_dataset_size, int(input_feature_count**(1/2)))

#     ff = ff(
#         pretrained=True,
#         training=False,
#         device_type=device_type,
#         model_params=fetch_ff_params_from_file(device_type, save_fpath),
#     )

#     prediction_batch = ff.inference(data_batch)
#     print_classification_results(test_dataset_size, prediction_batch, label_batch)

