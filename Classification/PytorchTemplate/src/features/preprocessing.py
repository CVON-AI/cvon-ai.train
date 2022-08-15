""" Utility functions related to image preprocessing. """

import os
from math import ceil
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage import img_as_float, filters, morphology
from skimage.measure import label
from skimage.filters import scharr
from skimage.transform import rotate
import numpy as np


def get_plot_info(config):
    """ Calculates the subplot layout in the axis. """
    num_subplots = (
        2 +  # Original and ROI crop
        6 * int(config["crop_to_mask"]["mode"]) +  # otsu|open|dil|lrg|pad|mask
        int(config["enhance_contrast"]["mode"]) +  # contrast
        int(config["rotate_to_similar"]["mode"]) +  # rotate
        int(config["edge_detect"]["mode"]))  # edge detect

    # Find a good row/column layout.
    # Find the largest number that can be modulod.
    num_rows = 1
    for i in range(2, int(num_subplots / 2)):
        if num_subplots % i == 0:
            num_rows = i

    # If num_subplots is prime, then approximate with square grid to
    # prevent a flattened grid.
    if num_rows == 1:
        num_rows = int(np.sqrt(num_subplots))

    num_cols = ceil(num_subplots / num_rows)

    return num_subplots, num_rows, num_cols


def crop_to_mask(np_img, config, show, axarr=None, current_plots=None):
    """ Subcrops to a heart-centered mask image. """

    # Mask out bright parts of the image.
    val = filters.threshold_otsu(np_img)
    mask = (np_img >= val * config["crop_to_mask"]["thr"]).astype(float)
    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=mask,
            title="Otsu")

    # Filter out small bright parts.
    ball = morphology.disk(radius=config["crop_to_mask"]["minArea"])
    mask = morphology.binary_opening(image=mask,
                                     selem=ball).astype(float)

    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=mask,
            title="Opening")

    # Pad the remaining bright parts (to connect heart chambers).
    ball = morphology.disk(radius=config["crop_to_mask"]["maxGap"])
    mask = morphology.dilation(image=mask,
                               selem=ball).astype(float)
    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=mask,
            title="Dilation")

    # Take only largest connected component (to get the heart).
    labels = label(mask)
    bins = np.bincount(labels.flat)[1:]

    # If the mask is empty, increase the sensitivity and restart.
    try:
        mask = labels == np.argmax(bins) + 1

        if show:
            current_plots, axarr = add_subplot_to_fig(
                axarr=axarr,
                config=config,
                curplot=current_plots,
                np_img=mask.astype(float),
                title="Largest")

        # Pad the mask to extract a larger image.
        ball = morphology.disk(radius=config["crop_to_mask"]["pad"])
        mask = morphology.dilation(image=mask,
                                   selem=ball).astype(float)

        if show:
            current_plots, axarr = add_subplot_to_fig(
                axarr=axarr,
                config=config,
                curplot=current_plots,
                np_img=mask.astype(float),
                title="Padding")

        # Extract the part of the heart under the mask.
        masked = np.ma.masked_where(1 - mask, np_img)
        masked_grey_values = (np.ma.median(masked), np.ma.std(masked))

        # Trim surrounding zeros to really crop to heart region.
        np_img = np_img[:, ~np.all(masked == 0, axis=0)]
        np_img = np_img[~np.all(masked == 0, axis=1)]

    except ValueError:  # Couldn't find appropriate mask.
        masked_grey_values = (None, None)

    # Normalize [0,1]
    np_img = normalize_np(np_img)

    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=np_img,
            title="mask crop")

    return np_img, axarr, current_plots, masked_grey_values


def add_subplot_to_fig(curplot, axarr, config, image=None, title="",
                       np_img=None):
    """ Adds a subplot to an axis array. """

    num_subplots, num_rows, num_cols = get_plot_info(config)
    cur_row = int(curplot / num_subplots * num_rows)
    cur_col = int(curplot % num_cols)

    curplot += 1

    if image is None:  # No PIL provided, read `np_img' instead.

        # Read grayscale PIL from numpy array.
        image = normalize_np(np_img, upper=255.0)
        image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')

    # Add image subplot to plot.
    axarr[cur_row, cur_col].imshow(image, cmap='gray')
    axarr[cur_row, cur_col].set_title(f"{curplot}: {title}", fontsize=10)
    axarr[cur_row, cur_col].axis('off')

    return curplot, axarr


def normalize_np(np_img, upper=1.0):
    """ Normalizes a numpy array between [0,upper]. """
    np_img -= np.min(np_img)
    return np_img * upper / np_img.max()


def enhance_contrast(np_img, config, show, masked_grey_values, axarr=None,
                     current_plots=None):
    """ Emphasizes the contrast in a value range in an array. """

    # Normalize in advance
    np_img = normalize_np(np_img)

    # Emphasize the value region between `left' and `right'.
    if masked_grey_values[0] is not None:
        # If a mask of the heart was used, base contrast values on that.
        left = max(0, masked_grey_values[0] - masked_grey_values[1])
        right = min(1, masked_grey_values[0] + masked_grey_values[1])
    else:  # Use the current crop value information.
        left = max(0, np.ma.median(np_img) - np.ma.std(np_img))
        right = min(1, np.ma.median(np_img) + np.ma.std(np_img))

    # Stretch the part within the left-right gray value range.
    factor = config["enhance_contrast"]["factor"]
    np_img = np.where(np_img < left,
                      np_img,
                      np_img + factor * (left - np_img))
    np_img = np.where(np_img > right,
                      np_img,
                      np_img - factor * (np_img - right))

    # Normalize again [0,1].
    np_img = normalize_np(np_img)

    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=np_img,
            title="Contrast+")

    return np_img, axarr, current_plots


def rotate_to_similar(np_img, config, show, labels, axarr, current_plots):
    """ Rotates the image to the orientation that has the least MSE
    with the images in the training directory.
    """

    # Find "average" image in train.
    imgpaths = list()
    for lab in labels.keys():
        for item in os.listdir(f"data/processed/{config['dataset_name']}"
                               f"/Train/{lab}"):
            imgpaths.append(f"data/processed/{config['dataset_name']}/Train/"
                            f"{lab}/{item}")

    # Computational complexity safeguard: subsample images.
    if config["rotate_to_similar"]["sample"]:
        imgpaths = random.sample(imgpaths,
                                 k=min(len(imgpaths),
                                       config["rotate_to_similar"]["sample"]))

    # Get the current training images, and obtain the average image to
    # compare the different rotations with.
    all_train = list()
    for imgpath in imgpaths:
        image = Image.open(imgpath).convert("L")
        image = image.resize(
            (config["img_size"][0], config["img_size"][1]), Image.BILINEAR)
        image = img_as_float(image)
        all_train.append(image)

    # Rotate this image iff there already exist some training images.
    if all_train:
        all_train = np.asarray(all_train)

        mean_img = np.mean(all_train, axis=0)

        best_deg = 0
        best_mse = np.inf

        # Try various orientations, and save the one that matches best
        # to the mean training image.
        for deg in range(0, 360, config["rotate_to_similar"]["deg_interval"]):

            rot = rotate(image=np_img, angle=deg, mode="reflect")

            # Convert to Image to upscale it bilinearly.
            rot = normalize_np(rot, upper=255.0)
            rot = Image.fromarray(rot)
            if rot.mode != 'L':
                rot = rot.convert('L')
            rot = rot.resize((config["img_size"][0], config["img_size"][1]),
                             Image.BILINEAR)

            # Convert both to same array.
            rot = normalize_np(rot, upper=1.0)
            mean_img = normalize_np(mean_img, upper=1.0)

            # Get mean squared error.
            mse = ((rot - mean_img) ** 2).mean()

            # Save degrees if lowest error so far.
            if mse < best_mse:
                best_mse = mse
                best_deg = deg

        # Final rotation.
        np_img = rotate(image=np_img, angle=best_deg, mode="reflect")

    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=np_img,
            title="Rotate")

    return np_img, axarr, current_plots


def edge_detect(np_img, config, show, axarr=None, current_plots=None):
    """ Performs Scharr edge detection on an array. """

    # Edge detect
    np_img = scharr(np_img)

    # Binarize image: all edges exceeding threshold get 1, else 0.
    np_img[np_img < np.quantile(np_img,
                                config["edge_detect"]["threshold"])] = 0
    np_img[np_img > 0] = 1

    # Skeletonize: edges become pixel-wide.
    np_img = morphology.skeletonize(np_img).astype(float)

    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=np_img,
            title="Edges")

    return np_img, axarr, current_plots


def preprocess_image(config, image, labels, show=False):
    """ Preprocess a PIL image."""

    # Initialize subplots.
    if show:
        current_plots = 0

        _, num_cols, num_rows = get_plot_info(config)
        _, axarr = plt.subplots(num_cols, num_rows, squeeze=False)
        for axi in axarr.ravel():
            axi.set_axis_off()

    else:
        axarr = None
        current_plots = None

    np_img = img_as_float(image)
    height, width = np_img.shape

    # Crop the image
    startx = int(width * config["crop_dims"][0])
    starty = int(height * config["crop_dims"][1])
    endx = int(width * config["crop_dims"][2])
    endy = int(height * config["crop_dims"][3])

    np_img = np_img[starty:endy, startx:endx]

    # Add original and ROI to preprocessing visualization.
    if show:
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            image=image,
            title="Original")
        current_plots, axarr = add_subplot_to_fig(
            axarr=axarr,
            config=config,
            curplot=current_plots,
            np_img=np_img,
            title="ROI crop")

    # Crop to mask.
    if config["crop_to_mask"]["mode"]:
        np_img, axarr, current_plots, masked_grey_values = crop_to_mask(
            np_img=np_img,
            show=show,
            config=config,
            axarr=axarr,
            current_plots=current_plots)
    else:  # Enhance function needs masked grey values from cropping.
        masked_grey_values = (None, None)

    # Enhance contrast
    if config["enhance_contrast"]["mode"]:
        np_img, axarr, current_plots = enhance_contrast(
            np_img=np_img,
            show=show,
            config=config,
            axarr=axarr,
            current_plots=current_plots,
            masked_grey_values=masked_grey_values)

    # Edge detection.
    if config["edge_detect"]["mode"]:
        np_img, axarr, current_plots = edge_detect(
            np_img=np_img,
            show=show,
            config=config,
            axarr=axarr,
            current_plots=current_plots)

    if config["rotate_to_similar"]["mode"]:  # Must be last in prp pipe
        np_img, axarr, current_plots = rotate_to_similar(
            np_img=np_img,
            show=show,
            config=config,
            axarr=axarr,
            labels=labels,
            current_plots=current_plots)

    # Convert the array to an image.
    np_img = normalize_np(np_img, upper=255.0)
    processed_image = Image.fromarray(np_img)
    if processed_image.mode != 'L':
        processed_image = processed_image.convert('L')

    # Save the preprocessing steps
    if show:
        plt.savefig("reports/figures/preprocessing.pdf")
        plt.close()

    return processed_image
