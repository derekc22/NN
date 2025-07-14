import torch
torch.set_printoptions(threshold=torch.inf)
from utils.mlp_utils import fetch_mlp_params_from_file
import glob, os, re



def fetch_transformer_params_from_file(device, directory, ff_directory):

    encoder_params = {}
    decoder_params = {}

    # Use glob to get all files matching the pattern
    WQKV_pattern = "encoder_*_WQKV.pth"  # Pattern to match
    WQKV_files = glob.glob(os.path.join(directory, WQKV_pattern))
    WQKV_files.sort()

    WO_pattern = "*coder_*_WO_h_[0-9]*.pth"  # Pattern to match
    # WO_pattern = "*coder_*_WO_h_*.pth"  # Pattern to match
    WO_files = glob.glob(os.path.join(directory, WO_pattern))
    WO_files.sort()
    
    num_layers = len(WO_files)
    num_decoders = int(num_layers/2)

    # Use glob to get all files matching the pattern
    WQKV_masked_pattern = "decoder_*_WQKV_masked.pth"  # Pattern to match
    WQKV_masked_files = glob.glob(os.path.join(directory, WQKV_masked_pattern))
    WQKV_masked_files.sort()

    WO_masked_pattern = "decoder_*_WO_masked.pth"  # Pattern to match
    WO_masked_files = glob.glob(os.path.join(directory, WO_masked_pattern))
    WO_masked_files.sort()

    WQ_pattern = "decoder_*_WQ.pth"  # Pattern to match
    WQ_files = glob.glob(os.path.join(directory, WQ_pattern))
    WQ_files.sort()

    WKV_pattern = "decoder_*_WKV.pth"  # Pattern to match
    WKV_files = glob.glob(os.path.join(directory, WKV_pattern))
    WKV_files.sort()


    regex_pattern_WO = r"(.*?)_(\d+)_WO_h_(\d+)\.pth"
    # regex_pattern_WO = r"(encoder|decoder)_(\d+)_WO_(\d+)\.pth"
        
    # load encoder params
    for (WQKV_f, WO_f) in zip(WQKV_files, WO_files[num_decoders:]):

        WQKV = torch.load(WQKV_f, map_location=device)
        WO = torch.load(WO_f, map_location=device)

        match = re.search(regex_pattern_WO, WO_f)
        component = match.group(1)
        index = match.group(2)
        num_heads = int(match.group(3))
        
        ff_params = fetch_mlp_params_from_file(device, f"{ff_directory}/encoder_{index}")

        encoder_params.update({f"encoder_{index}": [WQKV, WO, num_heads, ff_params, index] })
        

        
    # load decoder params
    for (WQKV_masked_f, WO_masked_f, WQ_f, WKV_f, WO_f) in zip(WQKV_masked_files, WO_masked_files, WQ_files, WKV_files, WO_files[:num_decoders]):

        WQKV_masked = torch.load(WQKV_masked_f, map_location=device)
        WO_masked = torch.load(WO_masked_f, map_location=device)
        WQ = torch.load(WQ_f, map_location=device)
        WKV = torch.load(WKV_f, map_location=device)
        WO = torch.load(WO_f, map_location=device)

        match = re.search(regex_pattern_WO, WO_f)
        component = match.group(1)
        index = match.group(2)
        num_heads = int(match.group(3))

        ff_params = fetch_mlp_params_from_file(device, f"{ff_directory}/decoder_{index}")
        
        decoder_params.update({f"decoder_{index}": [WQKV_masked, WO_masked, WQ, WKV, WO, num_heads, ff_params, index] })
    
    linear_pattern = f"decoder_{index}_linear.pth"  # Pattern to match
    linear_files = glob.glob(os.path.join(directory, linear_pattern))
    linear_files.sort()
    linear = torch.load(linear_files[0], map_location=device)
    decoder_params[f"decoder_{index}"][7:7] = [linear]



    params = {"encoder": encoder_params, "decoder": decoder_params}
        
    return params