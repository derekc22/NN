import torch
torch.set_printoptions(threshold=torch.inf)
import glob, os, re



def fetch_mlp_params_from_file(device, directory):

    params = {}

    # Use glob to get all files matching the pattern
    weight_pattern = "layer_*_weights_*.pth"  # Pattern to match
    weight_files = glob.glob(os.path.join(directory, weight_pattern))
    weight_files.sort()

    bias_pattern = "layer_*_biases_*.pth"  # Pattern to match
    bias_files = glob.glob(os.path.join(directory, bias_pattern))
    bias_files.sort()


    for (w_file, b_file) in zip(weight_files, bias_files):

        weights = torch.load(w_file, map_location=device)
        biases = torch.load(b_file, map_location=device)

        regex_pattern = r"layer_(\d+)_weights_(.*?)\.pth"
        match = re.search(regex_pattern, w_file)

        index = match.group(1)
        activation = match.group(2)

        params.update({f"Layer {index}": [weights, biases, activation, index] })

    return params




def gen_matrix_stack(n, d=5):
    num_a = n // 2
    num_b = n - num_a

    A = torch.randn(num_a, d, d)
    symmetric_tensors = (A + A.transpose(1, 2)) / 2
    symmetric_labels = torch.ones((num_a, ))

    B = torch.randn(num_b, d, d)
    non_symmetric_labels = torch.zeros((num_b, ))

    ds = torch.cat([symmetric_tensors, B], dim=0)
    labels = torch.cat([symmetric_labels, non_symmetric_labels], dim=0)

    shuffle_indices = torch.randperm(n)

    data_batch = ds[shuffle_indices]
    # print(data_batch)
    data_batch = torch.flatten(data_batch, start_dim=1)
    label_batch = labels[shuffle_indices]
    
    return data_batch, label_batch



if __name__ == "__main__":
    # gen_pet_img_stack(15, 64, 64, False, False, 1)

    # idk = fetch_mlp_params_from_file("cpu", "/params/paramsMLP")
    # idk = fetch_cnn_params_from_file("cpu", "./params/paramsCNN")

    idk1, idk2 = gen_matrix_stack(5, 2)
    print(idk1)
    print(idk2)