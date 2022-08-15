""" Utility functions related to image reading. """

import os
import random
import csv
import math
from shutil import rmtree
from PIL import Image
from torch import load
from features import preprocessing as prep


def initialize_processed_dirs(config, labels):
    """ Create processed subfolders. """

    # Initialize a "processed" folder.
    path_lab = os.path.join(f"data/processed/{config['dataset_name']}")
    if not os.path.exists(path_lab):
        os.mkdir(path_lab)

    # Initialize two subfolders "Train" and "Val".
    for split_type in ["Train", "Val"]:
        path_split = os.path.join(path_lab, split_type)
        if not os.path.exists(path_split):
            os.mkdir(path_split)

        # Within "Train" and "Val", initialize the class subfolders.
        for label in labels.keys():
            path = os.path.join(path_split, label)
            if os.path.exists(path):
                rmtree(path)
            os.mkdir(path)


def prepare_val(config, labels):
    """ Initializes train and validation subfolders."""

    initialize_processed_dirs(config=config, labels=labels)

    # Loop over unlabeled class folders to get image count per class.
    img_count_per_class = [len(os.listdir(f"data/raw/{config['dataset_name']}"
                                          f"/{label}"))
                           for label in list(labels.keys())]

    # The number of validation images per class is 50% of the size
    # of the smallest class, ceiled by config["max_val_per_class"].
    max_per_class = int(min(config["max_val_per_class"],
                            min(img_count_per_class) * 0.5))

    # Delegate a number of validation images to the validation folder.
    for label in labels.keys():
        num_val = 0
        while num_val < max_per_class:
            unlab_path = os.path.join(
                "data/raw", config['dataset_name'], label)
            lab_path = os.path.join(
                "data/processed", config['dataset_name'], "Val", label)
            while True:
                imgname = random.choice(os.listdir(unlab_path))
                if is_image_file(imgname):
                    break

            # Preprocess the image and store it in "Val".
            img = prep.preprocess_image(
                config=config,
                image=Image.open(os.path.join(
                    unlab_path, imgname)).convert("L"),
                labels=labels)

            # Don't add duplicate images
            if imgname in os.listdir(lab_path):
                continue

            img.save(os.path.join(lab_path, imgname))
            num_val += 1

            # To prevent data leakage, make sure that if "imagename_20.png"
            # is included in the validation, then so is "imagename_50.png"
            # and vice versa.

            img_types = ["_20.png", "_50.png"]
            for idx, img_type in enumerate(img_types):
                if img_type in imgname:
                    possible_dupe = imgname.replace(
                        img_type, img_types[1 - idx])
                    if possible_dupe in os.listdir(unlab_path) and \
                            possible_dupe not in os.listdir(lab_path):
                        img = prep.preprocess_image(
                            config=config,
                            image=Image.open(os.path.join(
                                unlab_path, possible_dupe)).convert("L"),
                            labels=labels)
                        img.save(os.path.join(lab_path, possible_dupe))
                        num_val += 1

    return labels


def resume_model(model):
    """Loads a previous model from a checkpoint."""
    try:
        checkpoint = load("models/savedmodel.pt")
        first_epoch = checkpoint['epoch'] + 1
        labels = checkpoint['labels']
        metric_tracker = checkpoint['metric_tracker']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except RuntimeError:
        print("Error: model found at checkpoint has different layout.")
    except KeyError:
        print("Error: couldn't load all requested keys in model "
              "checkpoint.")
    return first_epoch, labels, metric_tracker


def get_labels(config):
    """Read the labels from the file system."""
    labels = os.listdir(f"data/raw/{config['dataset_name']}")
    labels.sort()

    if not os.path.exists(f"data/processed/{config['dataset_name']}"):
        os.mkdir(f"data/processed/{config['dataset_name']}")
    for tv_type in ["Train", "Val"]:
        if not os.path.exists(f"data/processed/{config['dataset_name']}"
                              f"/{tv_type}"):
            os.mkdir(f"data/processed/{config['dataset_name']}/{tv_type}")
        for label in labels:
            if not os.path.exists(f"data/processed/{config['dataset_name']}"
                                  f"/{tv_type}/{label}"):
                os.mkdir(f"data/processed/{config['dataset_name']}/{tv_type}"
                         f"/{label}")

    # Assign integer values to the sorted class names.
    labels = dict((key, val) for val, key in enumerate(labels))

    return labels


def count_images(config, subdir, labels):
    """ Counts the images in a subfolder."""
    num = 0
    for label in labels.keys():
        num += len(os.listdir(os.path.join(
            f"data/{subdir}", label)))
    return num


def is_image_file(imgname):
    """ Check if the image path really refers to an image.
    Used to filter out .db files and such.
    """
    return imgname.endswith(("png", "jpg", "jpeg"))


def read_metadata(imgname, config):
    """Read the metadata that belongs to an image name."""
    def convert_to_float(field):
        """Placeholder docstring."""
        if isinstance(field, float):
            return field
        if isinstance(field, str) and field.isnumeric():
            return float(field)
        vals = ["ABSENT", "BIPED", "QUADRUPED"]
        if isinstance(field, str) and field in vals:
            return float(vals.index(field))

        return 0.0
    if config["metadata_filename"]:
        with open(f"references/{config['metadata_filename']}") as csvfile:
            reader = csv.reader(csvfile,
                                delimiter=',')
            row = None
            for row in reader:
                if imgname and row[-1] in imgname:
                    return [1 / (1 + math.exp(-convert_to_float(r)))
                            for r in row]
            # Didn't find meta for `imgname'.
            return [1 / (1 + math.exp(-convert_to_float(r)))
                    for r in row]

    return list()


def read_imgs(config, labels):
    """ Read all images that have not yet been labeled from the subdir
    and that aren't validation images.
    Returns a tuple with the image path, vector, and the label.
    """

    imgs = list()

    # List all labeled images.
    try:
        old = [os.listdir(os.path.join("data/processed",
                                       config['dataset_name'],
                                       tv_type,
                                       label))
               for label in labels.keys() for tv_type in ["Train", "Val"]]
        old = sum(old, [])  # Concatenate
    except FileNotFoundError:
        print(f"Error: Unknown class found in checkpoint.")

    # Return a list of all unseen images' paths and labels.
    for label in labels.keys():
        path = os.path.join("data/raw", config["dataset_name"], label)
        for imgname in os.listdir(path):
            if is_image_file(imgname) and imgname not in old:
                imgs.append((os.path.join(path, imgname), label))
    return imgs
