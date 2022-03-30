# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from fno4vc.fourier_neural_operator import FourierNeuralOperator
from fno4vc.project_path import checkpointsdir
from fno4vc.utils import CustomLRScheduler, read_dataset, make_grid

torch.manual_seed(0)
np.random.seed(0)


def train(args: argparse.Namespace):
    """Main training loop.

    Args:
        args: An argparse.Namespace, containing command line hyperparameters
            and data paths.
    """
    # Read training data pairs.
    init_image, images_dataset, models_dataset = read_dataset()

    # Move initial image and grid to `device` and repeat for concatenation.
    init_image = init_image.to(args.device).repeat(args.batch_size, 1, 1, 1)
    grid = make_grid(images_dataset.shape[1:3]).to(args.device).repeat(
        args.batch_size, 1, 1, 1)

    # Setup the Fourier neural operator and move to `device`.
    g = FourierNeuralOperator(args.modes, args.lifted_dim).to(args.device)

    # Setup the optimization algorithm.
    optim = torch.optim.Adam(g.parameters(), args.lr, weight_decay=args.wd)

    # Setup the learning rate scheduler.
    scheduler = CustomLRScheduler(optim, args.lr, args.lr_final,
                                  args.max_epoch)

    # Setup the batch index generator.
    data_loader = torch.utils.data.DataLoader(range(images_dataset.shape[0]),
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=True)
    # Setup the objective function logger.
    obj_log = []

    # Training loop, run for `args.max_epoch` epochs.
    with tqdm(range(args.max_epoch), unit='epoch', colour='#B5F2A9') as pb:
        for epoch in pb:
            # Reset gradient attributes.
            optim.zero_grad()
            # Update learning rate.
            scheduler.step()

            # Loop over batches.
            for itr, idx in enumerate(data_loader):
                with torch.no_grad():
                    # Extract the batch.
                    image = images_dataset[idx, ...]
                    model = models_dataset[idx, ...]
                    # Move to `device`.
                    image = image.to(args.device)
                    model = model.to(args.device)

                # Feed the batch to the Fourier neural operator.
                pred = g.forward(torch.cat([model, init_image, grid], dim=3))

                # Compute the objective function.
                obj = torch.norm(pred - image)**2

                # Compute the gradient.
                obj.backward()

                # Update network parameters.
                optim.step()

                # Print current objective value.
                pb.set_postfix(itr="{:02d}/{:2d}".format(
                    itr + 1, len(data_loader)),
                               obj="{:.2}".format(obj.item()))

                # Keep a log of objective values.
                obj_log.append(obj.item())

            # Save the current network parameters, optimizer state variables,
            # current epoch, and objective log.
            if epoch % 50 == 0 or epoch == args.max_epoch - 1:
                torch.save(
                    {
                        'model_state_dict': g.state_dict(),
                        'optim_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'obj_log': obj_log
                    },
                    os.path.join(checkpointsdir(args.experiment),
                                 'checkpoint_' + str(epoch) + '.pth'))
