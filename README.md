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

1. Clone the repository: git clone https://github.com/anurag1102990/ViT-and-GPT2-Captioning.git
2. Install the dependencies as mentioned above.
   
## Dataset

This project utilizes the **Flickr8k dataset**, which is well-known in the field of image captioning. Flickr8k contains 8,000 photographs annotated with descriptions, making it an ideal dataset for training and evaluating image captioning models due to its manageable size and diverse imagery.

### Dataset Structure

- **Images**: Each image in the dataset is paired with five different captions that provide diverse descriptive texts.
- **Annotations**: Captions are provided by human annotators and cover a broad range of language uses and descriptive styles.

### Usage

The dataset is split into training, validation, and testing sets. This separation allows for the comprehensive training of models and robust evaluation of their captioning capabilities. For setup:

1. Download the dataset from the designated repository or website.
2. Place it in the `data/` directory within this project's structure.
3. Ensure the paths in the configuration file are set correctly to point to the location of the dataset.

Training and evaluation scripts automatically load data from the specified paths and preprocess images and captions using the provided tokenizers and image processors.

For more detailed information or to access the dataset, visit the official [Flickr8k Dataset page](https://www.kaggle.com/datasets/dibyansudiptiman/flickr-8k).

## Model Architecture

This project integrates two powerful architectures, the Vision Transformer (ViT) and OpenAI's GPT-2, to create a state-of-the-art image captioning system. Here’s how each component contributes to the model:

### Vision Transformer (ViT)

- **Purpose**: Extracts rich feature representations from input images.
- **Configuration**: Utilizes patches of images, which are then embedded into a sequence of vectors. These vectors are processed through multiple layers of transformer blocks that learn to understand various aspects of the visual input.
- **Benefits**: Unlike traditional CNNs, ViT can capture more global context and relationships between different parts of the image due to its self-attention mechanism.

### GPT-2

- **Purpose**: Generates descriptive text based on the features provided by ViT.
- **Configuration**: Receives the encoded image features as initial prompts and generates a sequence of words to form a coherent caption. This is facilitated by leveraging the transformer's ability to predict the next word in a sequence given the previous words and the image context.
- **Benefits**: GPT-2's powerful language model, pretrained on a diverse corpus of text, enables the generation of natural, contextually appropriate captions.

### Integration

- **Vision Encoder-Decoder Model**: The integration involves passing the output from ViT directly into GPT-2. This coupling allows the system to directly translate visual information into textual descriptions, harnessing the strengths of both models.
- **Performance Impact**: The combination of ViT for visual understanding and GPT-2 for text generation allows the model to produce relevant and accurate captions that are contextually tied to the visual cues.

### Training and Evaluation

- **Training**: The model is trained end-to-end with the objective of minimizing the difference between the generated captions and the ground-truth annotations, using techniques such as teacher forcing during the early phases of training.
- **Evaluation**: Performance is evaluated using the BLEU score, which measures the linguistic quality of the captions.

This architecture not only leverages the advancements in both image processing and natural language processing technologies but also sets a foundation for further exploration into more integrated and complex multimodal systems.

## Results and Performance

The integration of Vision Transformer (ViT) and GPT-2 for image captioning was extensively trained and evaluated using the Flickr8k dataset. Below are the summarized results reflecting the performance of our model:

### Key Metrics
- **Vision Transformer Accuracy**: 80%
  - This metric indicates the effectiveness of the ViT model in extracting relevant features from images, which significantly contributes to the overall performance of the caption generation.
- **BLEU Score for Image Captioning**: 0.06
  - The BLEU score, while seemingly low, is typical in the context of complex tasks like image captioning where the diversity of possible correct captions is very high. This score reflects the linguistic accuracy of the generated captions relative to human-annotated captions.


