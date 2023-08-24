import torch.optim
from dataset import FlowerDataset
from models import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter, ToPILImage
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import cv2
import numpy as np

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="flowers")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            scale=(0.85, 1.15),
            shear=2.5
        ),
        ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.05
        ),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = FlowerDataset(args.root, train=True, transform=train_transform)

    # image, _ = train_dataset.__getitem__(300)
    # image_numpy = (image * 255).byte().numpy()
    # image_cv2 = cv2.cvtColor(image_numpy.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    # cv2.imshow("test image", image_cv2)
    # cv2.waitKey(0)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    test_dataset = FlowerDataset(args.root, train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    iters = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progess_bar = tqdm(train_dataloader, colour="green")
        for iter, (image_batch, label_batch) in enumerate(progess_bar):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            # forword
            outputs = model(image_batch)
            loss_value = criterion(outputs, label_batch)
            progess_bar.set_description(f"Epoch [{epoch+1}/{args.epochs}]. Iteration [{iter+1}/{iters}]. Loss [{loss_value:.3f}]")
            writer.add_scalar("Train/Loss. ", loss_value, epoch*iters + iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()    # update paramater

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (image_batch, label_batch) in enumerate(test_dataloader):
            all_labels.extend(label_batch)
            if torch.cuda.is_available():
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
            with torch.no_grad():
                predictions = model(image_batch)
                indices = torch.argmax(predictions, dim=1)
                all_predictions.extend(indices.cpu())
                # loss_value = criterion(predictions, label_batch)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"\nEpoch {epoch + 1}. Accuracy: {accuracy}\n")
        writer.add_scalar("Val/Accuracy. ", accuracy, epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, f"{args.trained_models}/last_model.pt")
        if accuracy > best_acc:
            checkpoint = {
                # "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, f"{args.trained_models}/best_model.pt")
            best_acc = accuracy



