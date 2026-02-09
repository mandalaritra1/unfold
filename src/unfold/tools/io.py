
import pickle as pkl

def load_pkls(pythia_path, data_path):
    """Load two pickle files and return their contents.
    Args:
        pythia_path (str): Path to the Pythia pickle file.
        data_path (str): Path to the data pickle file.
    Returns:
        tuple: A tuple containing the contents of the Pythia and data pickle files.
    """
    with open(pythia_path, "rb") as f:
        output_pythia = pkl.load(f)
    with open(data_path, "rb") as f:
        output_data = pkl.load(f)
    return output_pythia, output_data
