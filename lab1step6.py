import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer


def load_model_and_data():
    # Load the trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=8, N_output=784)
    model.load_state_dict(torch.load('MLP.8.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load the MNIST dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    return model, train_set, device


def encode_images(model, img1, img2, device):
    # Encode the two images
    with torch.no_grad():
        bottleneck1 = model.encode(img1.view(1, -1).to(device))
        bottleneck2 = model.encode(img2.view(1, -1).to(device))
    return bottleneck1, bottleneck2


def interpolate_bottlenecks(bottleneck1, bottleneck2, n_steps):
    # Linearly interpolate between the two bottleneck tensors
    alphas = torch.linspace(0, 1, n_steps)
    interpolated = torch.zeros(n_steps, bottleneck1.shape[1])
    for i, alpha in enumerate(alphas):
        interpolated[i] = (1 - alpha) * bottleneck1 + alpha * bottleneck2
    return interpolated


def decode_interpolations(model, interpolated, device):
    # Decode the interpolated bottleneck tensors
    with torch.no_grad():
        decoded = model.decode(interpolated.to(device)).cpu()
    return decoded


def plot_interpolation(img1, img2, decoded, n_steps):
    # Plot the original images and the interpolations in a single row
    fig, axes = plt.subplots(1, n_steps + 2, figsize=((n_steps + 2) * 2, 4))

    # Plot the first original image
    axes[0].imshow(img1.squeeze(), cmap='gray')
    axes[0].set_title('Image 1')

    # Plot interpolations
    for i in range(n_steps):
        axes[i + 1].imshow(decoded[i].view(28, 28), cmap='gray')
        axes[i + 1].set_title(f'Step {i + 1}')

    # Plot the second original image
    axes[-1].imshow(img2.squeeze(), cmap='gray')
    axes[-1].set_title('Image 2')

    # Remove axis ticks
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    model, train_set, device = load_model_and_data()

    # Get two image indices from the user
    idx1 = int(input(f"Enter the index of the first image (0-{len(train_set) - 1}): "))
    idx2 = int(input(f"Enter the index of the second image (0-{len(train_set) - 1}): "))

    # Get number of interpolation steps
    n_steps = int(input("Enter the number of interpolation steps: "))

    # Get the images
    img1 = train_set.data[idx1].float() / 255.0
    img2 = train_set.data[idx2].float() / 255.0

    # Encode the images
    bottleneck1, bottleneck2 = encode_images(model, img1, img2, device)

    # Interpolate between the bottlenecks
    interpolated = interpolate_bottlenecks(bottleneck1, bottleneck2, n_steps)

    # Decode the interpolations
    decoded = decode_interpolations(model, interpolated, device)

    # Plot the results
    plot_interpolation(img1, img2, decoded, n_steps)


if __name__ == "__main__":
    main()