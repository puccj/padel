"""Train YOLOv8 with Albumentations for data augmentation and normalization
This script sets up a YOLOv8 training pipeline using the Ultralytics YOLO library.
It includes custom data augmentation using Albumentations, computes dataset mean and standard deviation,
and trains the model with specified hyperparameters."""

from ultralytics import YOLO
from dataset import YOLOv8Dataset
from trainer import compute_mean_std

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = YOLO('models/yolo11x.pt')

TRAIN_IMG_DIR = "dataset_original/train/images/"
TRAIN_LABEL_DIR = "dataset_original/train/labels/"
VALID_IMG_DIR = "dataset_original/valid/images/"
VALID_LABEL_DIR = "dataset_original/valid/labels/"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
batch = 16
workers = 16
NUM_EPOCHS = 900
optimizer="SGD",    # robusto e generalizza bene
momentum=0.937,     # default per SGD
weight_decay=0.0005,
SAVE_INTERVAL = 10
SAVE_VAL_IMG = True
SAVE_TRAIN_IMG = True
PATIENCE = 20
TRAIN_LOSS_FILE = "train_loss.npy"
VAL_LOSS_FILE = "val_loss.npy"
TRAIN_ACC_FILE = "val_accuracy.npy"
VAL_DICE_FILE = "val_dice.npy"
LOSS_PLOT_PATH = "loss_plot.png"
ACC_PLOT_PATH = "accuracy_plot.png"
DICE_PLOT_PATH = "dice_plot.png"

basic_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    # No ToTensorV2() here!
    # No augmentations
    # No normalization
])
temp_dataset = YOLOv8Dataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=basic_transform)

MEAN, STD = compute_mean_std(temp_dataset)

def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A         

            # Insert required transformation here
            self.transform = A.Compose([
                A.RandomRain(p=0.0, slant_lower=-10, slant_upper=10,        # Apparently, without this everything crashes
                              drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
                              blur_value=5, brightness_coefficient=0.9, rain_type=None),
                A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
                A.HorizontalFlip(p=0.5),                # court symmetry
                A.Perspective(scale=(0.01, 0.05), p=0.2),
                A.Affine(scale=(0.95, 1.05),                # minor size changes
                        translate_percent=(-0.05, 0.05),   # small translations
                        rotate=(-5, 5),                    # mild rotation (balls are rotationally symmetric, but field is not)
                        shear=(-1, 1),                     # minor shearing
                        p=0.5),
                
                A.ColorJitter(brightness=0.2,   # lighting variation
                            contrast=0.2,     
                            saturation=0.1,   # balls are often bright-colored
                            hue=0.05,         # keep ball colors recognizable
                            p=0.5),
                A.RandomGamma(p=0.3),           # simulate over/underexposure

                A.ImageCompression(quality_range=(70, 100), p=0.5), # simulate streaming/recording compression
                
                A.Normalize(mean=MEAN, 
                            std=STD, 
                            max_pixel_value=255.0), # 255 (default) indicates that the pixel values on original images are in the range [0, 255]
                ToTensorV2()
                ],

                bbox_params=A.BboxParams(format='yolo', 
                                        label_fields=['class_labels'],
                                        # min_visibility=0.1,
                                        # filter_invalid_bboxes=True)  # Clips bboxes to [0, 1] range if they go out of bounds
                )
            )

            # LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            print(f"{prefix}albumentations package not found, skipping augmentation")
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

Albumentations.__init__ = __init__

results = model.train(data='dataset_original/data.yaml', epochs=900, imgsz=640, augment = True)