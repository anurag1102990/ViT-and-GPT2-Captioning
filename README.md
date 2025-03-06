# Vision Transformer and GPT-2 Image Captioning

This repository contains an implementation of an advanced image captioning system that leverages the power of Vision Transformers (ViT) and GPT-2. The project demonstrates how to integrate cutting-edge machine learning models for image understanding and natural language processing to generate descriptive captions for images.

## Project Overview

The system uses a Vision Transformer to process images and extract features, which are then fed into a GPT-2 model to generate coherent and contextually relevant captions. This approach combines the strengths of state-of-the-art models in computer vision and text generation, making it a robust solution for automated image captioning.

### Features

- **Vision Transformer (ViT):** Utilizes ViT for efficient and effective image feature extraction.
- **GPT-2:** Employs the GPT-2 model for generating descriptive text based on the image features.
- **Flickr8k Dataset:** Trains and evaluates the model using the Flickr8k dataset, a benchmark collection in the image captioning community.
- **Customizable Training Loop:** Includes a training script with adjustable parameters for experimentations.
- **BLEU Score Evaluation:** Measures the quality of captions using the BLEU score metric to ensure the captions' relevance and accuracy.

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch 1.8+
- Transformers 4.5+
- NLTK

### Installation

1. Clone the repository:
