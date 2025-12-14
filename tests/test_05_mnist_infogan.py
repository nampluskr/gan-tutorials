import os
import numpy as np
import torch
import pytest

from gan_tutorials.datasets import MNIST, get_train_loader
from gan_tutorials.models.infogan import InfoGenerator, InfoDiscriminator, InfoGAN
from gan_tutorials.trainer import train, evaluate, fit
from gan_tutorials.utils import sample_latent, sample_labels, create_images


DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
BATCH_SIZE = 100
IMG_SIZE = 32
LATENT_DIM = 128
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE = 64

NUM_DISCRETE = 10
NUM_CONTINUOUS = 2

NUM_EPOCHS = 2
NUM_SAMPLES = 100


@pytest.mark.mnist_infogan
def test_mnist_dataloader_images_valid():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(train_loader))
    images = batch["image"]

    assert images.dtype == torch.float32
    assert images.shape == (BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.mark.mnist_infogan
def test_mnist_dataloader_labels_valid():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    batch = next(iter(train_loader))
    labels = batch["label"]

    assert labels.dtype == torch.long
    assert labels.shape == (BATCH_SIZE, )


@pytest.mark.mnist_infogan
def test_mnist_infogan_generator_forward():
    generator = InfoGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    assert generator is not None

    noises = torch.randn(BATCH_SIZE, LATENT_DIM).float()
    codes_discrete = torch.randint(low=0, high=NUM_DISCRETE, size=(BATCH_SIZE,)).long()
    codes_continuous = torch.randn(BATCH_SIZE, NUM_CONTINUOUS).float()

    assert noises.shape == (BATCH_SIZE, LATENT_DIM)
    assert codes_discrete.shape == (BATCH_SIZE,)
    assert codes_continuous.shape == (BATCH_SIZE, NUM_CONTINUOUS)

    images = generator(noises, codes_discrete, codes_continuous)

    assert images.ndim == 4
    assert images.shape == (BATCH_SIZE, OUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    assert images.dtype == torch.float32
    assert images.min().item() >= -1.0
    assert images.max().item() <= 1.0


@pytest.mark.mnist_infogan
def test_mnist_infogan_discriminator_forward():
    discriminator = InfoDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    assert discriminator is not None

    images = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).float()
    adv_logits, disc_logits, cont_logits = discriminator(images)

    assert adv_logits.dtype == torch.float32
    assert adv_logits.ndim == 2
    assert adv_logits.shape == (BATCH_SIZE, 1)

    assert disc_logits.dtype == torch.float32
    assert disc_logits.ndim == 2
    assert disc_logits.shape == (BATCH_SIZE, NUM_DISCRETE)

    assert cont_logits.dtype == torch.float32
    assert cont_logits.ndim == 2
    assert cont_logits.shape == (BATCH_SIZE, NUM_CONTINUOUS)


@pytest.mark.mnist_infogan
def test_mnist_infogan_train_one_batch():
    generator = InfoGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    discriminator = InfoDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    gan = InfoGAN(generator, discriminator, 
        latent_dim=LATENT_DIM, num_discrete=NUM_DISCRETE, num_continuous=NUM_CONTINUOUS)
    batch = {"image": torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).float()}
    outputs = gan.train_step(batch)

    assert generator is not None
    assert discriminator is not None
    assert gan is not None
    assert "d_loss" in outputs
    assert "g_loss" in outputs
    assert "q_loss" in outputs
    assert "q_disc_loss" in outputs
    assert "q_cont_loss" in outputs


@pytest.mark.mnist_infogan
def test_mnist_infogan_train_one_epoch():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    generator = InfoGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    discriminator = InfoDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    gan = InfoGAN(generator, discriminator, 
        latent_dim=LATENT_DIM, num_discrete=NUM_DISCRETE, num_continuous=NUM_CONTINUOUS)
    results = train(gan, train_loader)

    assert isinstance(results, dict)
    assert gan.global_epoch == 1


@pytest.mark.mnist_infogan
def test_mnist_infogan_train_num_epoch():
    dataset = MNIST(DATA_DIR, split="train")
    train_loader = get_train_loader(dataset, batch_size=BATCH_SIZE)
    generator = InfoGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    discriminator = InfoDiscriminator(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    gan = InfoGAN(generator, discriminator, 
        latent_dim=LATENT_DIM, num_discrete=NUM_DISCRETE, num_continuous=NUM_CONTINUOUS)
    results = fit(gan, train_loader, num_epochs=NUM_EPOCHS)

    assert isinstance(results, dict)
    assert gan.global_epoch == NUM_EPOCHS


@pytest.mark.mnist_infogan
def test_mnist_infogan_create_images():
    generator = InfoGenerator(
        img_size=IMG_SIZE,
        latent_dim=LATENT_DIM,
        out_channels=OUT_CHANNELS,
        base=BASE,
        num_discrete=NUM_DISCRETE,
        num_continuous=NUM_CONTINUOUS,
    )
    assert generator is not None

    noises = torch.randn(BATCH_SIZE, LATENT_DIM).float()
    codes_discrete = torch.randint(low=0, high=NUM_DISCRETE, size=(BATCH_SIZE,)).long()
    codes_continuous = torch.randn(BATCH_SIZE, NUM_CONTINUOUS).float()

    images = create_images(generator, noises, 
        codes_discrete=codes_discrete, codes_continuous=codes_continuous)

    assert isinstance(images, np.ndarray)
    assert images.dtype == np.float32
    assert images.ndim == 4
    assert images.shape == (NUM_SAMPLES, IMG_SIZE, IMG_SIZE, OUT_CHANNELS)
    assert images.min() >= 0.0
    assert images.max() <= 1.0