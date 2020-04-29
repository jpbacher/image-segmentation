import os
import argparse
from pathlib import Path
import numpy as np

from project import Project
from data.loaders import get_generator
from losses.dice import dice_coeff, dice_bce_loss
from models.unet import unet, unet_large


arg = argparse.ArgumentParser()
arg.add_argument('-i', '--input', required=True, help='path to image folder')
arg.add_argument('-t', '--target', required=True, help='path to target folder')
arg.add_argument('-m', '--model', required=True, help='model to train')
arg.add_argument('-l', '--lr_schedule', required=False, help='type of lr callback')
arg.add_argument('-s', '--split', type=float, help='percentage of data used for training')

if __name__ == "__main__":
    project = Project()
    train_imgs = Path(project.data_dir / 'train')
    mask_imgs = Path(project.data_dir / 'train_masks')
