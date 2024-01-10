import torch

from cookie.models.model import Network


def test_model():
    model = Network(784, 10, [128, 64])

    # test that it accepts the correct input
    sample = torch.randn(64, 784)
    assert model(sample).shape == torch.Size([64, 10])

    # test that it has the correct output
    sample = torch.randn(1, 784)
    assert model(sample).shape == torch.Size([1, 10])


if __name__ == "__main__":
    test_model()
