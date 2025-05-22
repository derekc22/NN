import torch
torch.set_printoptions(threshold=torch.inf)
torch.set_printoptions(linewidth=200)
import glob, os, re
import numpy as np
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# def genSineWave(time_steps, freq, amp, T, phase=0, add_noise=False):
#     t = torch.linspace(0, T, time_steps)
#     X = amp * torch.sin(freq * t + phase)
#     if add_noise:
#         X += torch.from_numpy(np.random.normal(0, 0.01, t.shape)).float()
#     return t.unsqueeze(1), X.unsqueeze(1)


# def genSineWave(time_steps, freq, amp, T, add_noise=False):
#     if add_noise: t = torch.sort(torch.rand(time_steps) * T).values
#     else: t = torch.linspace(0, T, time_steps)
#     X = amp * torch.sin(freq * t)
#     return t.unsqueeze(1), X.unsqueeze(1)


# def genSineWave(time_steps, freq, amp, T, batch_size, phase=0, add_noise=False):
#     t = torch.sort(torch.rand(batch_size, time_steps, 1) * T, dim=1).values
#     X = amp * torch.sin(freq * t + phase)
#     if add_noise:
#         X += torch.from_numpy(np.random.normal(0, 0.01, t.shape)).float()
#     return t, X

def genSineWave(time_steps, freq, amp, T, batch_size, vary_dt, vary_phase, add_noise=False):
    t = torch.sort(torch.rand(batch_size, time_steps, 1) * T, dim=1).values if vary_dt else torch.linspace(0, T, time_steps).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
    arg = freq*(t)
    if vary_phase:
        arg += (torch.rand(batch_size, 1)*np.pi).unsqueeze(1).repeat(1, time_steps, 1)
    X = amp * torch.sin(arg)
    if add_noise:
        X += torch.from_numpy(np.random.normal(0, 0.01, t.shape)).float()
    return t, X



def fetchRNNParametersFromFile(device_type, directory):

    modelParams = {}

    # Use glob to get all files matching the pattern
    wxh_pattern = "layer_*_wxh*.pth"  # Pattern to match
    wxh_files = glob.glob(os.path.join(directory, wxh_pattern))
    wxh_files.sort()

    whh_pattern = "layer_*_whh_*.pth"  # Pattern to match
    whh_files = glob.glob(os.path.join(directory, whh_pattern))
    whh_files.sort()

    bh_pattern = "layer_*_bh.pth"  # Pattern to match
    bh_files = glob.glob(os.path.join(directory, bh_pattern))
    bh_files.sort()

    by_pattern = "layer_*_by.pth"  # Pattern to match
    by_files = glob.glob(os.path.join(directory, by_pattern))
    by_files.sort()

    if not all([len(wxh_files), len(whh_files), len(bh_files), len(by_files)]) :
        raise FileNotFoundError("Model parameters failed to load from file. Parameter folder may be empty")

    regex_pattern_whh = r"layer_(\d+)_whh_(.*?)\.pth"
    for (wxh_f, whh_f, bh_f) in zip(wxh_files[:-1], whh_files, bh_files):

        wxh = torch.load(wxh_f, map_location=device_type)
        whh = torch.load(whh_f, map_location=device_type)
        bh = torch.load(bh_f, map_location=device_type)

        match = re.search(regex_pattern_whh, whh_f)
        index = match.group(1)
        activation = match.group(2)

        modelParams.update({f"Layer {index}": [wxh, whh, bh, activation, index] })
    

    regex_pattern_wxh = r"layer_(\d+)_wxh_(.*?)\.pth"
    wxh_output = torch.load(wxh_files[-1], map_location=device_type)
    by = torch.load(by_files[0], map_location=device_type)
    match = re.search(regex_pattern_wxh, wxh_files[-1])
    index = match.group(1)
    activation = match.group(2)
    modelParams.update({f"Layer {index}": [wxh_output, by, activation, index] })
    
    return modelParams





# def get_string_embedding(text, embedding_dim, device='cpu'):
#     """
#     Converts a string to embeddings using a Hugging Face tokenizer and a PyTorch embedding layer.

#     Args:
#         text (str): Input string.
#         tokenizer: Hugging Face tokenizer.
#         embedding_layer (nn.Embedding): PyTorch embedding layer.
#         device (str): 'cpu' or 'cuda'.

#     Returns:
#         torch.Tensor: (sequence_length, embedding_dim)
#     """
#     # Load tokenizer (e.g., from BERT or GPT2)
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     vocab_size = tokenizer.vocab_size
#     embedding_layer = nn.Embedding(vocab_size, embedding_dim)

#     tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
#     with torch.no_grad():
#         embeddings = embedding_layer(tokens.squeeze(0))
#     return embeddings






# def decode_predictions(predictions, embedding_dim):
#     """
#     Map RNN outputs back to tokens by finding the nearest embeddings.

#     Args:
#         predictions (torch.Tensor): Tensor of shape (seq_len, embedding_dim)
#         embedding_layer (nn.Embedding): Embedding layer used during input.
#         tokenizer: Hugging Face tokenizer.

#     Returns:
#         list[str]: Decoded tokens
#     """
#     # Load tokenizer (e.g., from BERT or GPT2)
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     vocab_size = tokenizer.vocab_size
#     embedding_layer = nn.Embedding(vocab_size, embedding_dim)

#     # Get the embedding matrix (vocab_size x embedding_dim)
#     embedding_weights = embedding_layer.weight.data  # (vocab_size, embedding_dim)

#     # Cosine similarity between each output and vocab embeddings
#     similarities = F.cosine_similarity(
#         predictions.unsqueeze(1),  # (seq_len, 1, embed_dim)
#         embedding_weights.unsqueeze(0),  # (1, vocab_size, embed_dim)
#         dim=-1
#     )  # Result: (seq_len, vocab_size)

#     # Get the most similar token indices
#     token_ids = torch.argmax(similarities, dim=-1).tolist()

#     # Convert back to tokens
#     tokens = tokenizer.convert_ids_to_tokens(token_ids)
#     return tokens






def load_glove_embeddings(glove_path, embedding_dim, device='cpu'):
    vocab = {}
    vectors = []

    with open(glove_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) != embedding_dim:
                continue
            vocab[word] = idx
            vectors.append(vector)

    # Convert list of arrays to a single NumPy array for faster tensor creation
    embedding_matrix = torch.tensor(np.array(vectors))
    return vocab, embedding_matrix.to(device)



def encode_word(word, embedding_dim):
    """
    Encodes a single word to its GloVe vector.

    Args:
        word (str): The input word.
        vocab (dict): Mapping from word to index.
        embedding_matrix (torch.Tensor): Pre-loaded GloVe vectors.
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Word vector of shape (embedding_dim,)
    """

    glove_path = f"data/glove.6B/glove.6B.{embedding_dim}d.txt"
    vocab, embedding_matrix = load_glove_embeddings(glove_path, embedding_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = vocab.get(word.lower(), None)
    if index is None:
        raise ValueError(f"Word '{word}' not in GloVe vocabulary.")
    return embedding_matrix[index].to(device)





def decode_vector(predicted_vector, embedding_dim, top_k=1):
    """
    Finds the most similar words to the predicted vector using cosine similarity.

    Args:
        predicted_vector (torch.Tensor): Output from RNN, shape (embedding_dim,)
        embedding_matrix (torch.Tensor): GloVe vectors
        vocab (dict): word to index
        top_k (int): Number of closest words to return

    Returns:
        list[str]: Top-k predicted words
    """

    glove_path = f"data/glove.6B/glove.6B.{embedding_dim}d.txt"
    vocab, embedding_matrix = load_glove_embeddings(glove_path, embedding_dim)

    predicted_vector = predicted_vector.unsqueeze(0)  # shape: (1, embedding_dim)
    similarities = F.cosine_similarity(predicted_vector, embedding_matrix, dim=-1)  # shape: (vocab_size,)
    top_indices = similarities.topk(top_k).indices.tolist()

    index_to_word = {idx: word for word, idx in vocab.items()}
    return [index_to_word[i] for i in top_indices]


def encode_sentence(sentence, embedding_dim):
    words = sentence.split()  # Split by whitespace
    encoded = [encode_word(word, embedding_dim) for word in words]
    return torch.stack(encoded)


def decode_sentence(sentence_embedding, embedding_dim):
    decoded = ' '.join([decode_vector(word_embedding, embedding_dim)[0] for word_embedding in sentence_embedding])
    return decoded





if __name__ == "__main__":
#   modelParams = fetchRNNParametersFromFile("cpu", "params/paramsRNN")
#   print(modelParams)


    # embed_dim = 100

    # # w = "hello"
    # # embed_word = encode_word(w, embed_dim)
    # # print(embed_word)
    # # unembed_word = decode_vector(embed_word, embed_dim)
    # # print(unembed_word)


    # # with open("data/input.txt", "r") as f:
    # #     txt = f.read()
    # # sentence = txt

    # sentence = "hello my name is derek"
    # embed_sentence = encode_sentence(sentence, embed_dim)
    # # print(embed_sentence)
    # unembed_sentence = decode_sentence(embed_sentence, embed_dim)
    # print(unembed_sentence)


    t, X = genSineWave(100, 1, 1, 2*np.pi, 10, vary_phase=True, add_noise=False)
    print(t.shape)
    print(t[0, :, 0].shape)
    print(X.shape)
    print(X[0, :, 0].shape)
    for ti, Xi in zip(t, X):
        print(ti.shape)
        plt.plot(ti[:, 0], Xi[:, 0])
        plt.show()


    # t1, X1 = genSineWave(100, 1, 1, 2*np.pi, 10, phase=np.pi, add_noise=False)
    # t2, X2 = genSineWave(100, 1, 1, 2*np.pi, 10, phase=0,     add_noise=False)
    # print(X2.shape)
    # plt.scatter(t1, X1)
    # plt.scatter(t2, X2)
    # plt.show()

  
