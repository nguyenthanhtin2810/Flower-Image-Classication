from argparse import ArgumentParser
from models import SimpleCNN
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
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

    draw = ImageDraw.Draw(ori_image)

    width, height = ori_image.size
    x = int(width * 0.01)
    y = int(height * 0.01)

    text = f"{predicted_class} ({confidence*100:.2f})"
    text_color = (255, 255, 255)

    font_size = int(min(width, height) * 0.1)
    font = ImageFont.truetype("C:\Windows\Fonts/arial.ttf", font_size)

    x, y, text_width, text_height = draw.textbbox((x, y), text, font=font)
    background_for_text = (x, y, x + text_width, y + text_height)
    background_color = (0, 0, 124)
    draw.rectangle(background_for_text, fill=background_color)

    draw.text((x, y), text=text, font=font, fill=text_color)

    image_filename = args.image_path.split("\\")[2]
    ori_image.save(f"test_image/predicted_{image_filename}")
    ori_image.show()
