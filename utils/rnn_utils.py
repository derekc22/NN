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

    glove_path = f"data/text/glove.6B/glove.6B.{embedding_dim}d.txt"
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

    glove_path = f"data/text/glove.6B/glove.6B.{embedding_dim}d.txt"
    vocab, embedding_matrix = load_glove_embeddings(glove_path, embedding_dim)

    predicted_vector = predicted_vector.unsqueeze(0)  # shape: (1, embedding_dim)
    similarities = F.cosine_similarity(predicted_vector, embedding_matrix, dim=-1)  # shape: (vocab_size,)
    top_indices = similarities.topk(top_k).indices.tolist()

    index_to_word = {idx: word for word, idx in vocab.items()}
    return [index_to_word[i] for i in top_indices]


def encodeSentence(sentence: str, embedding_dim):
    print("encoding sentence...")
    sentence_cleaned = sentence.rstrip('\n')
    words = sentence_cleaned.split() # Split by whitespace
    print(words)
    encoded = [encode_word(word, embedding_dim) for word in words]
    return torch.stack(encoded)

def encodeParagraph(paragraph: list, embedding_dim):
    print("encoding paragraph...")
    encoded = [encodeSentence(sentence, embedding_dim) for sentence in paragraph]
    return torch.stack(encoded)

def decodeSentence(sentence_embedding, embedding_dim):
    print("decoding sentence...")
    decoded = ' '.join([decode_vector(word_embedding, embedding_dim)[0] for word_embedding in sentence_embedding])
    return decoded

def decodeParagraph(paragraph_embedding, embedding_dim):
    print("decoding paragraph...")
    decoded = '\n'.join([decodeSentence(sentence_embedding, embedding_dim) for sentence_embedding in paragraph_embedding])
    return decoded


def genTextTrainingData(readlines_arr: list, embed_dim):

    lengthBools = [len(readlines_arr[0].rstrip('\n').split()) == len(line.rstrip('\n').split()) for line in readlines_arr]
    allSameLength = all(lengthBools)
    if not allSameLength:
        print(lengthBools)
        raise ValueError(f"sentence #{lengthBools.index(False)} differs in length from sentence #0")

    data_batch = encodeParagraph(readlines_arr, embed_dim)
    batch_size = data_batch.shape[0]
    label_batch = data_batch[:, 1:, :].clone()
    end_line_tensor = torch.load("data/text/embeddings/end_line.pth").unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
    label_batch = torch.cat([label_batch, end_line_tensor], dim=1)

    torch.save(data_batch, "data/text/embeddings/data_batch.pth")
    torch.save(label_batch, "data/text/embeddings/label_batch.pth")

    return data_batch, label_batch


if __name__ == "__main__":
#   modelParams = fetchRNNParametersFromFile("cpu", "params/paramsRNN")
#   print(modelParams)


    embed_dim = 100

    # w = "end"
    # embed_word = encode_word(w, embed_dim)
    # print(embed_word)
    # unembed_word = decode_vector(embed_word, embed_dim)
    # print(unembed_word)
    # torch.save(embed_word, "data/text/embeddings/end_line.pth")
    # end_line_tensor = torch.load("data/text/embeddings/end_line.pth")
    # print(end_line_tensor)


    # with open("data/text/sentence.txt", "r") as f:
    #     txt = f.read()
    # sentence = txt
    # print(sentence)

    # # sentence = "hello my name is derek"
    # embed_sentence = encodeSentence(sentence, embed_dim)
    # print(embed_sentence)
    # unembed_sentence = decodeSentence(embed_sentence, embed_dim)
    # print(unembed_sentence)




    # with open("data/text/paragraph.txt", "r") as f:
    #     txt = f.readlines()
    # paragraph = txt
    # print(paragraph)

    # embed_paragraph = encodeParagraph(paragraph, embed_dim)
    # print(embed_paragraph.shape)

    embed_paragraph = torch.load("data/text/embeddings/data_batch.pth")[:5]
    unembed_paragraph = decodeParagraph(embed_paragraph, embed_dim)
    print(unembed_paragraph)


    # genTextTrainingData(paragraph, embed_dim)



    # t, X = genSineWave(100, 1, 1, 2*np.pi, 10, vary_dt=True, vary_phase=True, add_noise=False)
    # print(t.shape)
    # print(t[0, :, 0].shape)
    # print(X.shape)
    # print(X[0, :, 0].shape)
    # for ti, Xi in zip(t, X):
    #     print(ti.shape)
    #     plt.plot(ti[:, 0], Xi[:, 0])
    #     plt.show()


    # t1, X1 = genSineWave(100, 1, 1, 2*np.pi, 10, phase=np.pi, add_noise=False)
    # t2, X2 = genSineWave(100, 1, 1, 2*np.pi, 10, phase=0,     add_noise=False)
    # print(X2.shape)
    # plt.scatter(t1, X1)
    # plt.scatter(t2, X2)
    # plt.show()

  
