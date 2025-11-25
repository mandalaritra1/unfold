
import pickle as pkl

def load_pkls(pythia_path, data_path):
    with open(pythia_path, "rb") as f:
        output_pythia = pkl.load(f)
    with open(data_path, "rb") as f:
        output_data = pkl.load(f)
    return output_pythia, output_data
