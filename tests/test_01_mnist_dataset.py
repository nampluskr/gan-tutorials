import os
import torch
import pytest

from gan_tutorials.datasets import MNIST, get_train_loader

MNIST_DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
BATCH_SIZE = 64


@pytest.mark.mnist
def test_mnist_dataset_exists():
    data_dir = MNIST_DATA_DIR
    assert os.path.isdir(data_dir)


@pytest.mark.mnist
def test_mnist_train_dataset_length():
    train_dataset = MNIST(MNIST_DATA_DIR, split="train")
    assert train_dataset is not None
    assert len(train_dataset) == 60000


@pytest.mark.mnist
def test_mnist_test_dataset_length():
    test_dataset = MNIST(MNIST_DATA_DIR, split="test")
    assert test_dataset is not None
    assert len(test_dataset) == 10000


@pytest.mark.mnist
def test_mnist_dataloader_length():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    dataloader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    assert dataloader is not None
    assert len(dataloader) == 60000 // BATCH_SIZE


@pytest.mark.mnist
def test_mnist_images_valid():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    dataloader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))
    images = batch["image"]

    assert isinstance(images, torch.Tensor)
    assert images.ndim == 4
    assert images.shape == (BATCH_SIZE, 1, 32, 32)
    assert images.dtype == torch.float32
    assert images.min() >= -1.0
    assert images.max() <= 1.0


@pytest.mark.mnist
def test_mnist_labels_valid():
    dataset = MNIST(MNIST_DATA_DIR, split="train")
    dataloader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(dataloader))
    labels = batch["label"]

    assert isinstance(labels, torch.Tensor)
    assert labels.ndim == 1
    assert labels.shape == (BATCH_SIZE, )
    assert labels.dtype == torch.long
    assert labels.min() >= 0
    assert labels.max() <= 9