import torch
import ssl
import certifi
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer

# Properly configure the SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Create a function to replace the default SSL handler in urllib
import urllib

https_handler = urllib.request.HTTPSHandler(context=ssl_context)
opener = urllib.request.build_opener(https_handler)
urllib.request.install_opener(opener)

# Load the MNIST dataset
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = autoencoderMLP4Layer(N_input=784, N_bottleneck=8, N_output=784)
model.load_state_dict(torch.load('MLP.8.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()  # Set the model to evaluation mode


def add_noise(image, noise_factor=0.25):
    noise = torch.rand_like(image) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)


def display_reconstruction(index):
    if 0 <= index < len(train_set.data):
        # Get the original image and normalize it
        original_img = train_set.data[index].float() / 255.0

        # Add noise to the image
        noisy_img = add_noise(original_img)
        input_img = noisy_img.view(1, 784).to(device)

        # Get the reconstruction
        with torch.no_grad():  # Disable gradient calculations
            reconstructed_img = model(input_img).cpu().view(28, 28)

        # Display the images
        f = plt.figure(figsize=(15, 5))

        # Original image
        f.add_subplot(1, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Original (Label: {train_set.targets[index].item()})')

        # Noisy image
        f.add_subplot(1, 3, 2)
        plt.imshow(noisy_img.view(28, 28), cmap='gray')
        plt.title('Noisy Input')

        # Reconstructed image
        f.add_subplot(1, 3, 3)
        plt.imshow(reconstructed_img, cmap='gray')
        plt.title('Reconstructed')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Index out of range! Please enter a value between 0 and {len(train_set.data) - 1}")


# Main loop
while True:
    try:
        idx = int(input(f"Enter an index between 0 and {len(train_set.data) - 1} (or -1 to exit): "))
        if idx == -1:
            break

        display_reconstruction(idx)
    except ValueError:
        print("Invalid input! Please enter an integer for the index.")

print("Program ended.")