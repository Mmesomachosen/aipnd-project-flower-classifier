import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint)
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])

    # Class to index mapping
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Image processing
    image = Image.open(args.input)
    image_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = image_transform(image)
    image = image.unsqueeze(0)

    # Use GPU for inference if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = torch.exp(model(image))

    # Top K probabilities and classes
    top_probs, top_classes = output.topk(args.top_k, dim=1)

    # Convert indices to classes
    top_classes = [idx_to_class[idx.item()] for idx in top_classes[0]]

    # Display the result
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    class_names = [cat_to_name[cls] for cls in top_classes]

    for i in range(len(top_probs[0])):
        print(f"Class: {class_names[i]}, Probability: {top_probs[0][i].item():.4f}")

if __name__ == "__main__":
    main()
