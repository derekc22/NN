import torch
from utils.data import *
# from utils.ff_utils import *
from utils.logger import load_config
from models.transformer import Transformer
import argparse
from utils.tokenizer import Seq2SeqInputPreprocessor  


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
    # source_texts = [
    #     "Write a short story about a dragon.",
    #     "Describe a rainy day.",
    # ]

    # target_texts = [
    #     "Once upon a time, a dragon lived in a forest and guarded treasure.",
    #     "The rain fell gently, tapping against the windowpane all morning.",
    # ]

    source_texts = [
        "Write a short story about a dragon.",
        "Describe a rainy day.",
        # "My name is Derek.",
        # "Explain how a car engine works.",
        # "Translate 'Hello, how are you?' to French.",
        # "List the planets in the solar system.",
        # "Describe the feeling of loneliness.",
        # "What is the capital of Japan?",
        # "Tell me a joke about computers.",
        # "Summarize the plot of 'Hamlet'.",
        # "Give me a recipe for pancakes.",
        # "Explain the concept of gravity.",
        # "Who was Albert Einstein?",
        # "What is the tallest mountain in the world?",
        # "Describe the process of photosynthesis.",
        # "Write a motivational quote.",
        # "Name three types of clouds.",
        # "How do airplanes fly?",
        # "Define artificial intelligence.",
        # "What causes earthquakes?",
        # "What is blockchain technology?",
        # "Explain the water cycle.",
        # "Describe a sunset over the ocean.",
        # "Who wrote 'Pride and Prejudice'?",
        # "Tell a story about a lost dog.",
        # "Explain how to multiply fractions.",
        # "What is quantum computing?",
        # "List the steps to plant a tree.",
        # "What are the symptoms of the flu?",
        # "Describe a typical day at school."
    ]

    target_texts = [
        "Once upon a time, a dragon lived in a forest and guarded treasure.",
        "The rain fell gently, tapping against the windowpane all morning.",
        # "And I am the greatest machine learning engineer of all time.",
        # "A car engine works by igniting fuel in cylinders, which moves pistons to create mechanical power.",
        # "Bonjour, comment ca va ?",
        # "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        # "Loneliness feels like an empty room where no one hears your thoughts.",
        # "Tokyo is the capital city of Japan.",
        # "Why did the computer show up late? It had a hard drive.",
        # "Prince Hamlet seeks revenge after his father's ghost reveals a murder.",
        # "Mix flour, eggs, milk, and butter, then cook on a hot griddle until golden.",
        # "Gravity is the force that pulls objects toward one another, especially toward Earth.",
        # "Albert Einstein was a physicist known for the theory of relativity.",
        # "Mount Everest is the tallest mountain in the world at 8,848 meters.",
        # "Photosynthesis is how plants convert sunlight, carbon dioxide, and water into energy.",
        # "Success comes to those who persist when others give up.",
        # "Cumulus, stratus, and cirrus are three main types of clouds.",
        # "Airplanes fly by generating lift through the shape of their wings and forward thrust.",
        # "Artificial intelligence is the simulation of human intelligence in machines.",
        # "Earthquakes are caused by sudden movements along faults in Earth's crust.",
        # "Blockchain is a decentralized digital ledger that records transactions across many computers.",
        # "The water cycle includes evaporation, condensation, precipitation, and collection.",
        # "The sun dipped below the horizon, turning the sky into a canvas of gold and crimson.",
        # "Jane Austen wrote 'Pride and Prejudice'.",
        # "A dog wandered through the city until a kind stranger took it home.",
        # "To multiply fractions, multiply the numerators and multiply the denominators.",
        # "Quantum computing uses quantum bits to process information in parallel states.",
        # "Dig a hole, place the sapling, cover the roots with soil, and water it regularly.",
        # "Flu symptoms include fever, chills, fatigue, sore throat, and muscle aches.",
        # "Students attend classes, interact with teachers, and complete assignments throughout the day."
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

