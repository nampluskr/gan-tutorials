```
src/gan_tutorials/
├── __init__.py
│
├── datasets/
│   ├── __init__.py
│   ├── mnist.py
│   ├── cifar10.py
│   ├── celeba.py
│   └── loaders.py
│
├── models/
│   ├── __init__.py
│   │
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_generator.py
│   │   ├── base_discriminator.py
│   │   └── base_gan.py
│   │
│   ├── gan/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── gan.py
│   │
│   ├── cgan/
│   │   ├── __init__.py
│   │   ├── c_generator.py
│   │   ├── c_discriminator.py
│   │   └── cgan.py
│   │
│   ├── acgan/
│   │   ├── __init__.py
│   │   ├── ac_generator.py
│   │   ├── ac_discriminator.py
│   │   └── acgan.py
│   │
│   └── infogan/
│       ├── __init__.py
│       ├── info_generator.py
│       ├── info_discriminator.py
│       ├── q_network.py
│       └── infogan.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
│
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── sampling.py
│   ├── visualization.py
│   └── history.py
│
└── experiments/
    ├── __init__.py
    ├── mnist_gan.py
    ├── mnist_cgan.py
    ├── mnist_acgan.py
    └── mnist_infogan.py
```