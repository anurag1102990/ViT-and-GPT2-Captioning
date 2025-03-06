from typing import Tuple
import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Args:
    
    """Configuration class for model hyperparameters."""
    num_classes = 10
    epochs = 25  
    batch_size = 64
    lr = 5e-4
    weight_decay = 1e-5
    input_resolution = 32 # CIFAR 
    in_channels = 3 # RGB
    patch_size = 6
    hidden_size = 64
    layers = 6
    heads = 8


class PatchEmbeddings(nn.Module):
    
    """Module to extract patches from input images and project them into a latent space."""
    def __init__( self, input_resolution: int, patch_size: int, hidden_size: int, in_channels: int = 3):
        # Convolutional layer to project flattened patches into a hidden space
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the patches and transpose to fit the transformer input requirement

        batch_size = x.shape[0]
        x = self.projection(x)  # Shape: (batch_size, hidden_size, num_patches_height, num_patches_width)
        x = x.flatten(2)  # Flatten height and width into a single dimension (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)  # Transpose to (batch_size, num_patches, hidden_size). So we basically reshaped according to the transformer.

        embeddings = x

        return embeddings

class PositionEmbedding(nn.Module):
    """Module to add position embeddings to the encoded patches."""
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))

    def forward(
        self,
        embeddings: torch.Tensor
        ) -> torch.Tensor:

        batch_size = embeddings.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, hidden_size)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # Concatenate CLS token to the patch embeddings
        embeddings += self.position_embeddings

        return embeddings


class TransformerEncoderBlock(nn.Module):
    """Transformer block for processing sequences of embedded patches."""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):

        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]  # Residual connection after attention
        x = x + self.mlp(self.ln_2(x))  # Residual connection after MLP

        return x


class ViT(nn.Module):
    """Vision Transformer for classifying images based on the encoded and transformed patch representations."""

    def __init__(
        self, 
        num_classes: int,
        input_resolution: int, 
        patch_size: int, 
        in_channels: int,
        hidden_size: int, 
        layers: int, 
        heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size

        self.patch_embed = PatchEmbeddings(input_resolution, patch_size, hidden_size, in_channels)
        num_patches = (input_resolution // patch_size) ** 2
        self.pos_embed = PositionEmbedding(num_patches, hidden_size)
        self.ln_pre = nn.LayerNorm(hidden_size)
        
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(d_model=hidden_size, n_head=heads) for _ in range(layers)]
        )
        
        self.ln_post = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0])  # Use the output corresponding to the CLS token
        x = self.classifier(x)

        return x


def transform(
    input_resolution: int,
    mode: str = "train",
    mean: Tuple[float] = (0.5, 0.5, 0.5),   # NOTE: Modify this as you see fit
    std: Tuple[float] = (0.5, 0.5, 0.5),    # NOTE: Modify this as you see fit
    ):
    """Apply transformations and data augmentation to the input images."""

    if mode == "train":
        
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_resolution, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return tfm

def inverse_transform(
    img_tensor: torch.Tensor,
    mean: Tuple[float] = (-0.5/0.5, -0.5/0.5, -0.5/0.5),    
    std: Tuple[float] = (1/0.5, 1/0.5, 1/0.5),              
    ) -> np.ndarray:
    """Reverse the normalization process to convert tensors back to images for visualization."""
    
    inv_normalize = transforms.Normalize(mean=mean, std=std)
    img_tensor = inv_normalize(img_tensor).permute(1, 2, 0)
    img = np.uint8(255 * img_tensor.numpy())

    return img


def train_vit_model(args):
    """Training loop for the Vision Transformer model."""

    # Dataset for train / test
    tfm_train = transform(
        input_resolution=args.input_resolution, 
        mode="train",
    )

    tfm_test = transform(
        input_resolution=args.input_resolution, 
        mode="test",
    )

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ViT( num_classes=args.num_classes, input_resolution=args.input_resolution, patch_size=args.patch_size, in_channels=args.in_channels, hidden_size=args.hidden_size, layers=args.layers, heads=args.heads )
    # print(model)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.epochs}")

        for i, (x, labels) in enumerate(pbar):
            model.train()
            if torch.cuda.is_available():
                x, labels = x.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # NOTE: Show train loss at the end of epoch
            pbar.set_postfix({'loss': '{:.4f}'.format(loss.item())})

        # Evaluate at the end
        test_acc = test_classification_model(model, test_loader)

        # Save the model
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = {
                "model": model.state_dict(),
                "acc": best_acc,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(state_dict, "{}.pt".format(args.name))
            print("Best test acc:", best_acc)
        else:
            print("Test acc:", test_acc)
        print()

def test_classification_model(model: nn.Module,test_loader):
    """Evaluate the model on the test dataset and return the accuracy."""
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
