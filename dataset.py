import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split
import cv2

class FlowerDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.categories = ["bougainvillea", "daisies", "garden_roses", "gardenias", "hibiscus",
                           "hydrangeas", "lilies", "orchids", "peonies", "tulip"]

        image_filenames = os.listdir(root)
        self.image_paths = [os.path.join(root, file) for file in image_filenames]
        self.labels = [self.categories.index(file.rsplit("_", 1)[0]) for file in image_filenames]

        train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(self.image_paths,
                                                                        self.labels, test_size=0.2, random_state=42)
        if train:
            self.image_paths = train_image_paths
            self.labels = train_labels
        else:
            self.image_paths = test_image_paths
            self.labels = test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

if __name__ == '__main__':
    transform = Compose([
        # Resize((224, 224)),
        # ToTensor(),
    ])
    dataset = FlowerDataset(root="flowers", train=True, transform=transform)
    image, label = dataset.__getitem__(100)
    print(image.size)
    image.show()
    print(label)
