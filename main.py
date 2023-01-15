import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import functools
import operator

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: list,
            pool_every: int,
            hidden_dims: list,
            conv_params=None,
            activation_type: str = "relu",
            activation_params=None,
            pooling_type: str = "max",
            pooling_params=None,
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        if pooling_params is None:
            pooling_params = {}
        if activation_params is None:
            activation_params = {}
        if conv_params is None:
            conv_params = {}
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_c, _, _ = tuple(self.in_size)
        if "kernel_size" not in self.conv_params:
            self.conv_params["kernel_size"] = 1
        if "padding" not in self.conv_params:
            self.conv_params["padding"] = 0
        if "stride" not in self.conv_params:
            self.conv_params["stride"] = 1
        if "kernel_size" not in self.pooling_params:
            self.pooling_params["kernel_size"] = 1

        layers = []
        first_features = [in_c] + self.channels
        last_features = self.channels
        tuples = zip(first_features, last_features)

        for i, (in_num, out_num) in enumerate(tuples):
            layers.append(
                nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=self.conv_params['kernel_size'],
                          stride=self.conv_params['stride'],
                          padding=self.conv_params['padding']))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            if (i + 1) % self.pool_every == 0:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        features = functools.reduce(operator.mul, list(self.feature_extractor(torch.zeros(1, *self.in_size)).shape))
        layers = []

        first_features = [features] + self.hidden_dims
        last_features = self.hidden_dims
        tuples = zip(first_features, last_features)

        for in_num, out_num in tuples:
            layers.append(nn.Linear(in_num, out_num))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        xtemp = self.feature_extractor(x)
        xtemp = xtemp.reshape(xtemp.shape[0], -1)
        return self.classifier(xtemp)


def main():
    test_params = [
        dict(
            in_size=(3, 100, 100), out_classes=10,
            channels=[32] * 4, pool_every=2, hidden_dims=[100] * 2,
            conv_params=dict(kernel_size=3, stride=1, padding=1),
            activation_type='relu', activation_params=dict(),
            pooling_type='max', pooling_params=dict(kernel_size=2),
        ),
        dict(
            in_size=(3, 100, 100), out_classes=10,
            channels=[32] * 4, pool_every=2, hidden_dims=[100] * 2,
            conv_params=dict(kernel_size=5, stride=2, padding=3),
            activation_type='lrelu', activation_params=dict(negative_slope=0.05),
            pooling_type='avg', pooling_params=dict(kernel_size=3),
        ),
    ]

    for i, params in enumerate(test_params):
        net = ConvClassifier(**params)
        print(f"\n=== test {i=} ===")
        print(net)

        test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
        test_out = net(test_image)
        print(f'{test_out=}')

        expected_out = torch.load(f'tests/assets/expected_conv_out_{i:02d}.pt')
        diff = torch.norm(test_out - expected_out).item()
        print(f'{diff=:.3f}')


if __name__ == "__main__":
    main()
