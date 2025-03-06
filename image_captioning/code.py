from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator

from functools import partial
import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer, default_data_collator, ViTImageProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, AutoTokenizer, Seq2SeqTrainingArguments
import os
# ##########
# TODO: Add more imports

# ##########

class Args:
    """Configuration class storing hyperparameters and model configuration."""
    # Encoder-Decoder for captioning
    encoder = "google/vit-base-patch16-224"
    decoder = "gpt2"

    # Dataset path
    root_dir = "./flickr8k"

    # Hyperparameters
    batch_size = 8
    lr = 5e-5
    epochs = 2

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45     # TODO: Can play around

    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50

class FlickrDataset(Dataset):
    """Custom dataset class to load the Flickr8k dataset."""

    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer
        self.processor = processor
        self.tokenizer = tokenizer
        
        img_paths_captions = self.load_flickr8k_data(args.root_dir, mode)
        if img_paths_captions is not None:
            self.img_paths, self.captions = img_paths_captions
        else:
            raise ValueError("Failed to load dataset. Check the data file format.")
        
        # self.img_paths, self.captions = None, None
        
    def load_flickr8k_data(self, root_dir, mode):
        """Loads the image paths and captions from a specified dataset split."""

        img_paths = []
        captions = []
        with open(f"{root_dir}/{mode}.txt", "r") as file:
            for line in file:
                img_path, caption = line.strip().split(';')
                if (img_path == "image"):
                    continue
                img_paths.append(f"{root_dir}/images/{img_path}")
                captions.append(caption)
        return (img_paths, captions) if img_paths and captions else None

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Fetches an image-caption pair from the dataset based on index."""

        image = Image.open(self.img_paths[idx]).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.tokenizer(self.captions[idx], return_tensors="pt", padding="max_length", max_length=50).input_ids.squeeze()
        
        encoding = {
            "pixel_values": pixel_values,       # Return processed image as a tensor
            "labels": labels,             # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding

    
def train_cap_model(args):
    """Sets up and trains the image captioning model."""
    processor = ViTImageProcessor.from_pretrained(args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    tokenizer.add_special_tokens({'pad_token': '<|pad|>', 'bos_token': '<|beginoftext|>', 'eos_token': '<|endoftext|>'})

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = args.max_length
    model.generation_config.num_beams = args.num_beams
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")    # NOTE: Send your model to GPU

    # Load train/val dataset
    train_dataset = FlickrDataset(args, mode="train", processor=processor, tokenizer=tokenizer)
    val_dataset = FlickrDataset(args, mode="val", processor=processor, tokenizer=tokenizer)

    # Model configuration. 
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_steps=500,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    data_collator = default_data_collator  # Use DataCollatorForSeq2Seq for better handling of Seq2Seq models
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    trainer.save_model(args.name)
    processor.save_pretrained(args.name)
    model.save_pretrained(args.name)
    tokenizer.save_pretrained(args.name)
    

def load_trained_model(
    ckpt_dir: str,
    ):
    """TODO: Load your best trained model, processor and tokenizer.
    """
    # Load your model configuration
    config = VisionEncoderDecoderConfig.from_pretrained(ckpt_dir)

    # Load encoder processor
    processor = ViTImageProcessor.from_pretrained(ckpt_dir)

    # Load decoder tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    # Load your best trained model
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir, config=config)
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer

def inference(img_path,model, processor,tokenizer):
    """
    Generates captions for an image using a trained Vision Encoder-Decoder model.
    
    Args:
    img_path (str): Path to the image file.
    model (VisionEncoderDecoderModel): Trained Vision Encoder-Decoder model.
    processor (ViTImageProcessor): Processor associated with the Vision Transformer.
    tokenizer (AutoTokenizer): Tokenizer for decoding generated tokens to text.
    
    Returns:
    str: Generated caption for the image.
    """
    # Load and process the image
    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(images=image, return_tensors="pt").pixel_values   # Preprocess the image

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(img_tensor, max_length=50, num_beams=5, early_stopping=True)

    # Tokens -> Str
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """Computes the BLEU score for model predictions.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }
