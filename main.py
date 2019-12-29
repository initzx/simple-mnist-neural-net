import numpy as np

from src.mnist_loader import load_data_wrapper
from src.network import Network
from PIL import Image


def train_1():
    training_data, validation_data, test_data = [[*k] for k in load_data_wrapper('./data/mnist.pkl.gz')]
    dimensions = [784, 64, 32, 16, 10]
    network = Network(dimensions)
    network.train(training_data, 1, 30, 3.0, test_data)
    network.export_net("./models")


def train_import(model):
    training_data, validation_data, test_data = [[*k] for k in load_data_wrapper('./data/mnist.pkl.gz')]
    network = Network.load_net(model)
    network.train(training_data, 20, 30, 2.0, test_data)
    network.export_net("./models")


def test():
    img = np.invert(Image.open("data/digit.png").convert('L')).ravel()/255
    network = Network.load_net("models/network_v1577543917.npy")
    print(f"I think it's a {network.test_image(img)}?")


if __name__ == "__main__":
    test()
    # train_1()
    # train_import("models/network_v1577543917.npy")