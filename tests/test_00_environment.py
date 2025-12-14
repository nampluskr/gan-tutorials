import torch

def test_torch_installed():
    assert torch.__version__ is not None

def test_tensor_creation():
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)

def test_cuda_info():
    print("CUDA available:", torch.cuda.is_available())
