import os
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from project import Project
from data.loaders import get_generator
from losses.dice import dice_coeff, dice_bce_loss
from models.unet import unet, unet_large
from callbacks.learning_rate import SGDRSchedule, step_decay_lr


def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-s', '--split', type=float, help='percentage of data used for training')
    arg.add_argument('-m', '--model', required=True, help='model to train')
    arg.add_argument('-bs', '--batch_size', type=int, help='batch size')
    arg.add_argument('e', '--epochs', type=int, help='number of epochs')
    arg.add_argument('-is', '--img_size', type=list, help='image size')
    arg.add_argument('lrs', '--lr_schedule', type=str, help='learning rate schedule')
    args = vars(arg.parse_args())
    return args

def main():
    args = parse_args()
    project = Project()
    train_folder = Path(project.data_dir / 'train')
    mask_folder = Path(project.data_dir / 'train_masks')
    train_files = os.listdir(train_folder)
    train_masks = os.listdir(mask_folder)
    train_files_sorted = sorted(train_files)
    train_masks_sorted = sorted(train_masks)
    train_ids, val_ids, train_masks, val_masks = train_test_split(
        train_files_sorted, train_masks_sorted, train_size=args['split'], random_state=21
    )
    val_ids, test_ids, val_masks, test_masks = train_test_split(
        val_ids, val_masks, test_size=0.5, random_state=18
    )
    train_gen = get_generator(
        train_folder, mask_folder, train_ids, train_masks, batch_size=args['batch_size'],
        img_size=args['img_size']
    )
    val_gen = get_generator(
        train_folder, mask_folder, val_ids, val_masks, batch_size=args['batch_size'],
        img_size=args['img_size']
    )
    test_gen = get_generator(
        train_folder, mask_folder, test_ids, test_masks, batch_size=args['batch_size'],
        img_size=args['img_size']
    )
    model = args['model']
    model = model(input_size=tuple(args['img_size']))
    lr_scheduler = args['lr_schedule']
    if lr_scheduler == 'sgdr':
        schedule = SGDRSchedule(min_lr=1e-6, max_lr=1e-2, lr_decay=0.9, cycle_length=3, mult_factor=1.5)
    elif lr_scheduler == 'step_decay':
        schedule = step_decay_lr(1e-3, factor=0.2, step_size=2)
    elif lr_scheduler == 'plateau':
        schedule = ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.00001)
    else:
        raise ValueError("Must be 'sgdr', 'step_decay', or 'plateau'")
    model.compile(optimizer=Adam, loss=[dice_bce_loss], metrics=[dice_coeff])
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=int(len(train_ids) / args['batch_size']),
                                  epochs=args['epochs'],
                                  validation_data=val_gen,
                                  validation_steps=int(len(val_ids) / args['batch_size']),
                                  callbacks=


