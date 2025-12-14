import os
import torch
import pytest

from src.gan_tutorials.datasets import MNIST, get_train_loader

MNIST_DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"


@pytest.mark.env
def test_mnist_dataset_exists():
    data_dir = MNIST_DATA_DIR
    assert os.path.isdir(data_dir)


@pytest.mark.dataset
def test_mnist_dataset_created():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    assert dataset is not None


@pytest.mark.dataloader
def test_mnist_dataloader_created():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    batch_size = 128
    dataloader = get_train_loader(dataset, batch_size)
    assert dataloader is not None


@pytest.mark.batch
def test_mnist_batch_ndim_shape_range_dtype():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    batch_size = 128
    dataloader = get_train_loader(dataset, batch_size)
    batch = next(iter(dataloader))
    x, y = batch["image"], batch["label"]

    print()
    print(f">> images shape: {x.shape}, ndim: {x.ndim}")
    print(f">> labels shape: {y.shape}, ndim: {y.ndim}")

    assert x.ndim == 4
    assert x.shape[1:] == (1, 32, 32)
    assert x.dtype == torch.float32
    assert x.min() >= -1.0
    assert x.max() <= 1.0

    assert y.ndim == 1
    assert y.dtype == torch.long

