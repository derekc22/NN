import torch
torch.set_printoptions(threshold=torch.inf)
torch.set_printoptions(linewidth=200)
import glob, os, re
import numpy as np



def fetch_lstm_params_from_file(device, directory):

    params = {}

    # Use glob to get all files matching the pattern
    wf_pattern = "layer_*_wf_*.pth"  # Pattern to match
    wf_files = glob.glob(os.path.join(directory, wf_pattern))
    wf_files.sort()

    wi_pattern = "layer_*_wi_*.pth"  # Pattern to match
    wi_files = glob.glob(os.path.join(directory, wi_pattern))
    wi_files.sort()

    # Use glob to get all files matching the pattern
    wc_pattern = "layer_*_wc_*.pth"  # Pattern to match
    wc_files = glob.glob(os.path.join(directory, wc_pattern))
    wc_files.sort()

    wo_pattern = "layer_*_wo_*.pth"  # Pattern to match
    wo_files = glob.glob(os.path.join(directory, wo_pattern))
    wo_files.sort()

    # Use glob to get all files matching the pattern
    bf_pattern = "layer_*_bf.pth"  # Pattern to match
    bf_files = glob.glob(os.path.join(directory, bf_pattern))
    bf_files.sort()

    bi_pattern = "layer_*_bi.pth"  # Pattern to match
    bi_files = glob.glob(os.path.join(directory, bi_pattern))
    bi_files.sort()

    # Use glob to get all files matching the pattern
    bc_pattern = "layer_*_bc.pth"  # Pattern to match
    bc_files = glob.glob(os.path.join(directory, bc_pattern))
    bc_files.sort()

    bo_pattern = "layer_*_bo.pth"  # Pattern to match
    bo_files = glob.glob(os.path.join(directory, bo_pattern))
    bo_files.sort()


    regex_pattern_wf = r"layer_(\d+)_wf_(.*?)\.pth"
    for (wf_f, wi_f, wc_f, wo_f, bf_f, bi_f, bc_f, bo_f) in zip(wf_files, wi_files, wc_files, wo_files, bf_files, bi_files, bc_files, bo_files):

        wf = torch.load(wf_f, map_location=device)
        wi = torch.load(wi_f, map_location=device)
        wc = torch.load(wc_f, map_location=device)
        wo = torch.load(wo_f, map_location=device)

        bf = torch.load(bf_f, map_location=device)
        bi = torch.load(bi_f, map_location=device)
        bc = torch.load(bc_f, map_location=device)
        bo = torch.load(bo_f, map_location=device)


        match = re.search(regex_pattern_wf, wf_f)
        index = match.group(1)
        gate_activation = match.group(2)

        params.update({f"layer_{index}": [wf, wi, wc, wo, bf, bi, bc, bo, gate_activation, index] })
    

    why_pattern = "layer_*_why_*.pth"  # Pattern to match
    why_files = glob.glob(os.path.join(directory, why_pattern))
    why_files.sort()

    by_pattern = "layer_*_by.pth"  # Pattern to match
    by_files = glob.glob(os.path.join(directory, by_pattern))
    by_files.sort()

    why = torch.load(why_files[0], map_location=device)
    by = torch.load(by_files[0], map_location=device)

    regex_pattern_why = r"layer_(\d+)_why_(.*?)\.pth"
    match = re.search(regex_pattern_why, why_files[-1])
    # index = match.group(1)
    output_activation = match.group(2)

    output_params =  [why, by, output_activation]
    params[f"layer_{index}"][9:9] = output_params
    
    return params
