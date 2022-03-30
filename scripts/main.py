# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: March 2022
# Copyright: Georgia Institute of Technology, 2022

import argparse
import os
import torch
from fno4vc.project_path import datadir
from scripts.training import train


def main(args: argparse.Namespace):
    """Main function to call training or prediction.

    Args:
        args: An argparse.Namespace, containing command line hyperparameters
            and data paths.
    """
    # Experiment name according to input arguments.
    args.experiment = ('{}_lr-{}_lr-final-{}_wd-{}_modes-{}_lifted_dim-{}'
                       '_batch_size-{}'.format(args.experiment, args.lr,
                                               args.lr_final, args.wd,
                                               args.modes, args.lifted_dim,
                                               args.batch_size))

    if args.phase == 'training':
        train(args)

    # elif args.phase == 'prediction':
    #     predict(args)

    # elif args.phase == 'plotting':
    #     plot(args)


def download_data():
    """Download the training and testing data.

    Args:
        args: An argparse.Namespace, containing command line hyperparameters
            and data paths.
    """
    training_data_file = os.path.join(datadir('training_pairs'),
                                      'data-pairs.h5')
    if not os.path.isfile(training_data_file):
        os.system('wget https://www.dropbox.com/s/yj8n35qglol66db/'
                  'training-pairs.h5 -O' + training_data_file)
    testing_data_file = os.path.join(datadir('testing_pairs'), 'data-pairs.h5')
    if not os.path.isfile(testing_data_file):
        os.system('wget https://www.dropbox.com/s/3c5siiqshwxjd6k/'
                  'data-pairs.h5 -O' + testing_data_file)


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max_epoch',
                        dest='max_epoch',
                        type=int,
                        default=500,
                        help='maximum number of epochs')
    parser.add_argument('--lr',
                        dest='lr',
                        type=float,
                        default=0.002,
                        help='initial learning rate')
    parser.add_argument('--lr_final',
                        dest='lr_final',
                        type=float,
                        default=0.0005,
                        help='final learning rate')
    parser.add_argument('--wd',
                        dest='wd',
                        type=float,
                        default=1e-4,
                        help='weight decay coefficient')
    parser.add_argument('--experiment',
                        dest='experiment',
                        default='VC_with_FNO',
                        help='experiment name')
    parser.add_argument('--cuda',
                        dest='cuda',
                        type=int,
                        default=1,
                        help='set itto 1 for running on GPU, 0 for CPU')
    parser.add_argument('--modes',
                        dest='modes',
                        type=int,
                        default=24,
                        help='number of fouerier modes')
    parser.add_argument('--lifted_dim',
                        dest='lifted_dim',
                        type=int,
                        default=32,
                        help='lifted domain dimension')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=16,
                        help='batch_size')
    parser.add_argument('--phase',
                        dest='phase',
                        default='training',
                        help='train or prediction')
    parsed_args = parser.parse_args()

    # Download data if it is not already downloaded.
    download_data()

    # Setting default device (cpu/cuda) depending on CUDA availability and input
    # arguments.
    if torch.cuda.is_available() and parsed_args.cuda:
        parsed_args.device = torch.device('cuda')
    else:
        parsed_args.device = torch.device('cpu')

    # Run the main function, which calls training or prediction.
    main(parsed_args)
