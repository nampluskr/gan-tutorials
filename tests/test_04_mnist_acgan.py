import os
import numpy as np
import torch
import pytest

from gan_tutorials.datasets import MNIST, get_train_loader
from gan_tutorials.models.acgan import ACGenerator, ACDiscriminator, ACGAN
from gan_tutorials.trainer import train, evaluate, fit
from gan_tutorials.utils import sample_latent, sample_labels, create_images


DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
BATCH_SIZE = 100
IMG_SIZE = 32
LATENT_DIM = 128
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE = 64

NUM_CLASSES = 10
EMBEDDING_DIM = 10
EMBEDDING_CHANNELS = 1

NUM_EPOCHS = 2
NUM_SAMPLES = 100


@pytest.mark.mnist_acgan
def test_mnist_dataloader_images_valid():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(train_loader))
    images = batch["image"]

    assert images.dtype == torch.float32
    assert images.shape == (BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.mark.mnist_acgan
def test_mnist_dataloader_labels_valid():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(train_loader))
    labels = batch["label"]

    assert labels.dtype == torch.long
    assert labels.shape == (BATCH_SIZE, )


@pytest.mark.mnist_acgan
def test_mnist_acgan_generator_forward():
    generator = ACGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    noises = torch.randn(BATCH_SIZE, LATENT_DIM)
    labels = torch.arange(NUM_CLASSES).repeat(BATCH_SIZE // NUM_CLASSES).long()
    images = generator(noises, labels)

    print()
    print(f"{noises.shape}")
    print(f"{labels.shape}")
    print(f"{images.shape}")

    assert generator is not None
    assert images.ndim == 4
    assert images.shape == (BATCH_SIZE, OUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    assert images.min() >= -1.0
    assert images.max() <= 1.0


@pytest.mark.mnist_acgan
def test_mnist_acgan_discriminator_forward():
    discriminator = ACDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
    )
    images = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).float()
    adv_logits, aux_logits = discriminator(images)

    print()
    print(f">> images: {images.shape}")
    print(f">> adv_logits: {adv_logits.shape}")
    print(f">> aux_logits: {aux_logits.shape}")

    assert discriminator is not None
    assert adv_logits.ndim == 2
    assert adv_logits.shape == (BATCH_SIZE, 1)
    assert aux_logits.ndim == 2
    assert aux_logits.shape == (BATCH_SIZE, 10)


@pytest.mark.mnist_acgan
def test_mnist_acgan_train_one_batch():
    generator = ACGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    discriminator = ACDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
    )
    gan = ACGAN(generator, discriminator)
    batch = {
        "image": torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).float(),
        "label": torch.arange(NUM_CLASSES).repeat(BATCH_SIZE // NUM_CLASSES).long(),
    }
    outputs = gan.train_step(batch)

    assert generator is not None
    assert discriminator is not None
    assert gan is not None
    assert "d_loss" in outputs
    assert "g_loss" in outputs


@pytest.mark.mnist_acgan
def test_mnist_acgan_train_one_epoch():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    generator = ACGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    discriminator = ACDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
    )
    gan = ACGAN(generator, discriminator)
    results = train(gan, train_loader)

    assert isinstance(results, dict)
    assert gan.global_epoch == 1


@pytest.mark.mnist_acgan
def test_mnist_acgan_train_num_epoch():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    generator = ACGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    discriminator = ACDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
    )
    gan = ACGAN(generator, discriminator)
    results = fit(gan, train_loader, num_epochs=NUM_EPOCHS)

    assert isinstance(results, dict)
    assert gan.global_epoch == NUM_EPOCHS


@pytest.mark.mnist_acgan
def test_mnist_acgan_create_images():
    generator = ACGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
    )
    noises = sample_latent(NUM_SAMPLES, LATENT_DIM)
    labels = sample_labels(NUM_SAMPLES, NUM_CLASSES)
    images = create_images(generator, noises, labels=labels)

    assert isinstance(images, np.ndarray)
    assert images.dtype == np.float32
    assert images.ndim == 4
    assert images.shape == (NUM_SAMPLES, IMG_SIZE, IMG_SIZE, OUT_CHANNELS)
    assert images.min() >= 0.0
    assert images.max() <= 1.0