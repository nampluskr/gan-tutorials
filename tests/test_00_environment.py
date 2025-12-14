import os
import torch
import pytest

DATASET_DIR = "/mnt/d/deep_learning/datasets"


@pytest.mark.env
def test_torch_installed():
    assert torch.__version__ is not None


@pytest.mark.env
def test_tensor_creation():
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)


@pytest.mark.env
def test_cuda_info():
    print("CUDA available:", torch.cuda.is_available())


@pytest.mark.env
def test_dataset_dir_exists():
    data_dir = DATASET_DIR
    assert os.path.isdir(data_dir), f"dataset directory not found: {data_dir}"




