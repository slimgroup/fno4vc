import git
import os
from typing import Optional


def gitdir() -> str:
    """Find the absolute path to the GitHub repository root.
    """
    git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
    git_root = git_repo.git.rev_parse('--show-toplevel')
    return git_root


def datadir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to a directory at the data directory.

    Data directory, located at the GitHub repository root, is for training and
    testing data. Here the path is created if it does not exist upon call if
    `mkdir` is True.

    Args:
        path: A string for directory name located at the data directory.
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(gitdir(), 'data/', path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def plotsdir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to a directory at the plot directory.

    Plot directory, located at the GitHub repository root, is storing figure of
    experiment results. Here the path is created if it does not exist upon call
    if `mkdir` is True.

    Args:
        path: A string for directory name located at the plot directory.
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(gitdir(), 'plots/', path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path


def checkpointsdir(path: str, mkdir: Optional[bool] = True) -> str:
    """The absolute path to a directory at the checkpoint directory.

    Checkpoint directory, located at the GitHub repository root, is storing
    intermediate training checkpoints, e.g., network weights. Here the path is
    created if it does not exist upon call if `mkdir` is True.

    Args:
        path: A string for directory name located at the checkpoint directory.
        mkdir: An optional boolean for whether to create the directory if it
            does not exist.
    """
    path = os.path.join(datadir('checkpoints'), path)
    if (not os.path.exists(path)) and mkdir:
        os.makedirs(path)
    return path
