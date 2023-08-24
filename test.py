from argparse import ArgumentParser
from models import SimpleCNN
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

def get_args():
    parser = ArgumentParser(description="Testing")
    parser.add_argument("--image-path", "-p", type=str, default=None)
    parser.add_argument("--image-size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_model.pt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    categories = ["Bougainvillea", "Daisies", "Garden Roses", "Gardenias", "Hibiscus",
                           "Hydrangeas", "Lilies", "Orchids", "Peonies", "Tulips"]
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SimpleCNN(num_classes=10).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        accuracy = checkpoint["best_acc"]
        print(f"Accuracy of model: {accuracy}")
    else:
        print("No checkpoint found!")
        exit(0)

    model.eval()
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    ori_image = Image.open(args.image_path)
    image = transform(ori_image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    predicted_class = categories[predicted.item()]
    confidence = torch.max(probs).item()
    print(f"The test image is about {predicted_class} with confident score of {confidence*100:.2f}%")
    ori_image.show()
