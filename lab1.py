from time import process_time_ns

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
import matplotlib.pyplot as plt

# Import functions from existing files
# from lab1step4 import display_reconstruction as display_reconstruction_step4

# from lab1step5 import add_noise

from lab1step6 import main as step6_main

def load_model_and_data():
    # Load the MNIST dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    # Load the trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=8, N_output=784)
    model.load_state_dict(torch.load('MLP.8.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, train_set, device

def run_step4(model, train_set, device):
    while True:
        try:
            idx = int(input(f"Step 4 - Enter an index between 0 and {len(train_set.data) - 1} (or -1 to return to main menu): "))
            if idx == -1:
                break
            if 0 <= idx < len(train_set.data):
                # Get the original image and normalize it
                original_img = train_set.data[idx].float() / 255.0

                # Prepare the input for the model (1x784 dimensional tensor)
                input_img = original_img.view(1, 784).to(device)

                # Get the reconstruction
                with torch.no_grad():  # Disable gradient calculations
                    reconstructed_img = model(input_img).cpu().view(28, 28)

                # Display the images side-by-side using the specified matplotlib code
                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(original_img, cmap='gray')
                plt.title(f'Original (Label: {train_set.targets[idx].item()})')
                f.add_subplot(1, 2, 2)
                plt.imshow(reconstructed_img, cmap='gray')
                plt.title('Reconstructed')
                plt.show()
            else:
                print(f"Index out of range! Please enter a value between 0 and {len(train_set.data) - 1}")
        except ValueError:
            print("Invalid input! Please enter an integer.")

def run_step5(model, train_set, device):
    while True:
        try:
            idx = int(input(f"Step 5 - Enter an index between 0 and {len(train_set.data) - 1} (or -1 to return to main menu): "))
            if idx == -1:
                break
            if 0 <= idx < len(train_set.data):
                # Get the original image and normalize it
                original_img = train_set.data[idx].float() / 255.0

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
                plt.title(f'Original (Label: {train_set.targets[idx].item()})')

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
        except ValueError:
            print("Invalid input! Please enter an integer.")

def add_noise(image, noise_factor=0.25):
    noise = torch.rand_like(image) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def main():
    model, train_set, device = load_model_and_data()

    while True:
        try:
            step = int(input("Enter a step number (4, 5, or 6) or -1 to quit: "))
            if step == -1:
                break
            elif step == 4:
                run_step4(model, train_set, device)
            elif step == 5:
                run_step5(model, train_set, device)
            elif step == 6:
                step6_main()
            else:
                print("Invalid step number. Please enter 4, 5, 6, or -1 to quit.")
        except ValueError:
            print("Invalid input! Please enter an integer.")

    print("Program ended.")

if __name__ == "__main__":
    main()