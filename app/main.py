import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Example tensor
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).to(DEVICE)
    print("Tensor on device:", tensor)

    # Example numpy array
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    print("Numpy array:", array)

    # Convert numpy array to tensor
    tensor_from_numpy = torch.from_numpy(array).to(DEVICE)
    print("Converted tensor on device:", tensor_from_numpy)