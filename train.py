import argparse
from torchvision import datasets, transforms
import torch
from torch import nn, optim
from torchvision.models import vgg13
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    parser.add_argument('data_directory', type=str, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', choices=['vgg13'], help='Choose architecture (only vgg13 is supported)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Data loading and transformation
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(args.data_directory, transform=data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)

    # Model architecture
    model = vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    # Training on GPU if available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    print("Training started Please do not terminate")
    for epoch in range(args.epochs):
        for inputs, labels in dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Training loss: {loss:.4f}")

    # Save the checkpoint
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_units': args.hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets.class_to_idx,
                  'arch': 'vgg13'}

    torch.save(checkpoint, args.save_dir)

if __name__ == "__main__":
    main()
