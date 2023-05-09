# utils to transform images, create a dataloader to compute the embeddings, and actually compute the embeddings

from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import torch
from typing import Tuple, List, Type, Union
from PIL import Image
import torch.nn as nn
from tqdm import tqdm

class CustomDataSet(Dataset):
    """
    Custom dataset class for loading images from a directory (no labels).
    """
    def __init__(self, all_imgs: list, transform: transforms, abs_path: str = 'images/') -> None:
        """
        Initialize the dataset.

        Args:
            all_imgs (list): List of image names.
            transform (transforms): Transformations to be applied on the images.
            abs_path (str): Absolute path to the directory containing the images.
        """
        self.abs_path = abs_path
        self.all_imgs = all_imgs
        self.total_imgs = len(all_imgs)
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return self.total_imgs

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return the image at the given index.

        Args:
            idx (int): Index of the image to be returned.

        Returns:
            torch.Tensor: Image at the given index.
        """
        image = Image.open(self.abs_path + self.all_imgs[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image


def transform_image(image_list: List[Image.Image]) -> torch.Tensor:
    """
    Transform a list of images to a tensor applying normalize and resize transforms
    Args:
        image_list (List[Image.Image]): list of images
    Returns:
        torch.Tensor: tensor of images
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.Resize((224, 224))])
    image_list = [transform(image) for image in image_list]
    image_tensor = torch.stack(image_list)
    return image_tensor


def load_images_in_dl(image_paths: list, batch_size: int = 256, abs_path: str = 'images/') -> torch.utils.data.DataLoader:
    """
    Loads images from a list of paths and returns a dataloader.
    
    Args:
        image_paths (list): list of paths to images
        batch_size (int): batch size for the dataloader
        abs_path (str): absolute path to the images
        
    Returns:
        torch.utils.data.DataLoader: dataloader with the images
    """
    # load the images
    preprocess = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        transforms.Resize((224, 224))])
    dataset = CustomDataSet(image_paths, transform=preprocess, abs_path=abs_path)
    # create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


def embeddings_from_dl(im_encoder: nn.Module, dl: torch.utils.data.DataLoader, dl_with_labels: bool = False) -> torch.Tensor:
    """
    Computes embeddings for all images in a dataloader.
    
    Args:
        im_encoder: an ImageEncoder
        dl: a dataloader that returns images, and labels if labels=True
        labels: whether to collect and return labels as well
    
    Returns:
        im_embeds: a tensor of shape (num_images, embedding_size)
        all_labels: a list of lists of labels, if labels=True
    """    
    im_embeds = torch.zeros(len(dl.dataset), im_encoder.embedding_size).to('cuda')
    all_labels = []
    for i, images in tqdm(enumerate(dl)):
        if dl_with_labels:
            images, labels = images
            all_labels += list(labels)
        temp_embeds = im_encoder.encode(images, alr_preprocessed=True).cpu()
        im_embeds[i * dl.batch_size : (i + 1) * dl.batch_size] = temp_embeds
    if dl_with_labels:
        im_embeds = im_embeds, all_labels
    return im_embeds