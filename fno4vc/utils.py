# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022
# Partially based on the code by Zongyi Li at:
# https://github.com/zongyi-li/fourier_neural_operator

import h5py
import numpy as np
import torch
from typing import Optional, Tuple, Sequence
import os
from fno4vc.project_path import datadir


class Normalizer(object):
    """Normalizer a tensor image with training mean and standard deviation.

    Extracts the mean and standard deviation from the training dataset, and uses
    them to normalize an input image.

    Attributes:
        mean: A torch.Tensor containing the mean over the training dataset.
        std: A torch.Tensor containing the standard deviation over the training.
        eps: A small float to avoid dividing by 0.
    """

    def __init__(self, dataset: torch.Tensor, eps: Optional[int] = 0.00001):
        """Initializes a Normalizer object.

        Args:
            dataset: A torch.Tensor that first dimension is the batch dimension.
            eps: A optional small float to avoid dividing by 0.
        """
        super().__init__()

        # Compute the training dataset mean and standard deviation over the
        # batch dimensions.
        self.mean = torch.mean(dataset, 0)
        self.std = torch.std(dataset, 0)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input sample.

        Args:
            x: A torch.Tensor with the same dimension organization as `dataset`.

        Returns:
            A torch.Tensor with the same dimension organization as `x` but
            normalized with the mean and standard deviation of the training
            dataset.
        """
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Restore the normalization from the input sample.

        Args:
            x: A normalized torch.Tensor with the same dimension organization as
            `dataset`.

        Returns:
            A torch.Tensor with the same dimension organization as `x` that has
            been unnormalized.
        """
        return x * (self.std + self.eps) + self.mean


class CustomLRScheduler(object):
    """A custom learning rate scheduler.

    The learning rate is computed as `a * (b + t) ** gamma`, where `t` is the
    iteration number, `gamma` is the decay rate, and  `a, b` are chosen to
    control the initial and final learning rate.

    Attributes:
        optimizer: A torch.optim.Optimizer to update its learning rate.
        initial_lr: A float for the initial learning rate.
        final_lr: A float for the final learning rate.
        gamma: A negative float indicating the decay rate.
        a: A float for `a` according to the initial and final learning rate
        b: A float for `b` according to the initial and final learning rate
        count: An integer for the number of steps.
    """

    def __init__(self,
                 optim: torch.optim.Optimizer,
                 initial_lr: float,
                 final_lr: float,
                 max_step: int,
                 gamma: Optional[float] = -1 / 3):
        """A custom learning rate scheduler.

        The learning rate is computed as `a * (b + k) ** gamma`, where `k` is
        the step number, `gamma` is the decay rate, and  `a,
        b` are chosen to control the initial and final learning rate.

        Args:
            optimizer: A torch.optim.Optimizer to update its learning rate.
            initial_lr: A float for the initial learning rate.
            final_lr: A float for the final learning rate.
            max_step: An integer for the maximum number of steps.
            gamma: An optional negative float indicating the decay rate.

        Raises:
            ValueError: If `final_lr` is larger than `initial_lr`.
            ValueError: If `gamma` is larger than 0.0.
        """
        super().__init__()
        if final_lr > initial_lr:
            raise ValueError('The final learning rate must be smaller than the'
                             ' initial learning rate.')
        if gamma > 0.0:
            raise ValueError('The decay rate must be negative.')

        self.optim = optim
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.gamma = gamma

        # Compute the a and b values for according to `initial_lr`, `final_lr`.
        self.b = max_step / ((final_lr / initial_lr)**(1 / gamma) - 1.0)
        self.a = initial_lr / (self.b**gamma)

        # Initialize the step count.
        self.count = 0

    def compute_lr(self) -> float:
        """Computes the learning rate for the current step.

        Returns:
            A float for the learning rate.
        """
        if self.initial_lr == self.final_lr:
            return self.initial_lr
        else:
            return self.a * (self.b + self.count)**self.gamma

    def step(self):
        """Updates the optimizer learning rate.
        """
        # Obtain the current learning rate.
        lr = self.compute_lr()
        # Update the optimizer learning rate.
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        # Increment the step count.
        self.count += 1


def read_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reads the training data and normalizes it.

    Returns:
        init_image: A torch.Tensor with the initial image of size `(1, height,
            width, 1)`.
        images_dataset: A torch.Tensor of images in the dataset of size `(N,
            height, width, 1)`.
        models_dataset: A torch.Tensor of models in the dataset of size `(N,
            height, width, 1)`.
    """
    # Read the training HDF5 data.
    data_file_path = os.path.join(datadir('training_pairs'), 'data-pairs.h5')
    data_file = h5py.File(data_file_path, 'r')

    # The HDF5 data contains 'images', 'models', and 'image-base' datasets.
    images_dataset = torch.from_numpy(data_file['images'][...])
    models_dataset = torch.from_numpy(data_file['models'][...])
    init_image = torch.from_numpy(data_file['image-base'][...])
    data_file.close()

    # Normalize the seismic images in the training data.
    image_normalizer = Normalizer(images_dataset)
    images_dataset = image_normalizer.normalize(images_dataset)
    init_image = image_normalizer.normalize(init_image)

    # Normalize the background models in the training data.
    model_normalizer = Normalizer(models_dataset)
    models_dataset = model_normalizer.normalize(models_dataset)

    # Adding an extra dimension as the last dimension.
    init_image = init_image.unsqueeze(-1)
    images_dataset = images_dataset.unsqueeze(-1)
    models_dataset = models_dataset.unsqueeze(-1)

    return init_image, images_dataset, models_dataset


def make_grid(spatial_dim: Sequence[int]) -> torch.Tensor:
    """Make the grid of coordinates for the Fourier neural operator input.

    Args:
        spatial_dim: A sequence of spatial deimensions `(height, width)`.

    Returns:
        A torch.Tensor with the grid of coordinates of size `(1, height, width,
            2)`.
    """
    grids = []
    grids.append(np.linspace(0, 1, spatial_dim[0]))
    grids.append(np.linspace(0, 1, spatial_dim[1]))
    grid = np.vstack([u.ravel() for u in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, spatial_dim[0], spatial_dim[1], 2)
    grid = grid.astype(np.float32)
    return torch.tensor(grid)
