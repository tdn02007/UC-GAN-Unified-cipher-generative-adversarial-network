# UC-GAN

Automated Classical Cipher Emulation Attacks via Unified Unsupervised Generative Adversarial Networks

Torch implementation for learning a mapping between Plain text to Cipher Text

<img src="img/img.PNG" width="900px"/>

## Setup

### Prerequisites

- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Getting Started

- Clone this repo:

```bash
git clone git@github.com:tdn02007/UC-GAN-Unified-cipher-generative-adversarial-network.git
cd UC-GAN-Unified-cipher-generative-adversarial-network
```

### Lib

The code to make cipher text.

- caeser.py: Caeser
- substitution.py: Substitution
- vigenere.py: Vigenere
- cipher.py: Encryption Text code

### data

Example of dataset.

## Train

```bash
python main.py
```
