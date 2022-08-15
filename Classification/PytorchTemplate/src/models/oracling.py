""" Utility functions related to querying and oracling. """

import os
import random
import tkinter
from statistics import mean
from PIL import Image, ImageTk
from tqdm import tqdm
from features import preprocessing as prep
from data import reading
from models import feeding
from visualization import logging


def ask_user_input(config, imgpath, labels, label=None, smax=None,
                   counter=None):
    """Prompt the oracle to label an image."""

    num_classes = len(list(labels.keys()))

    # Set up a window screen.
    root = tkinter.Tk()
    root.title("<YourTitle> Active Learning")

    # Place an image canvas on the screen (for the full image).
    win_full = tkinter.Canvas(root, width=500, height=500)
    win_full.grid(row=0, column=0, padx=5)

    # Draw the image on the canvas and add a rectangle to indicate the
    # crop size.
    img2 = prep.preprocess_image(
        config=config,
        image=Image.open(imgpath),
        labels=labels)

    img1 = Image.open(imgpath)

    img1 = img1.resize((500, 500), Image.LANCZOS)
    img1 = ImageTk.PhotoImage(img1)
    win_full.create_image(0, 0, anchor=tkinter.NW, image=img1)

    # Place another image canvas on the screen (for the crop).
    win_cropped = tkinter.Canvas(root, width=500, height=500)
    win_cropped.grid(row=0, column=num_classes-1, padx=5)

    # Draw the image on the canvas.
    img2 = img2.resize((500, 500), Image.NEAREST)
    img2 = ImageTk.PhotoImage(img2)
    win_cropped.create_image(0, 0, anchor=tkinter.NW, image=img2)

    # If there is a text with an image index counter, display it.
    if counter:
        counter_label = tkinter.Label(master=root, text=counter)
        counter_label.grid(row=1, column=0, columnspan=num_classes)

    # Insert a query text on the screen.
    query_label = tkinter.Label(
        master=root,
        text="Which class does this image belong to?" +
        (f" (hint: it's {label})" if label else ""))
    query_label.grid(row=2, column=0, columnspan=num_classes)

    class Answer():
        """Helper class to process the user's answer. An answer can only
        be stored as a class member.
        """
        def __init__(self, answer=None):
            self.answer = answer

    user_answer = Answer()

    # Define the text fields on the UI buttons.
    query_button_texts = dict()
    for lab in labels.keys():
        query_button_texts[lab] = lab + (f"({100*smax[labels[lab]]:.0f}"
                                         "% sure)"
                                         if smax is not None else "")

    def button(user_answer, string):
        """Function to define a button's behavior. It sets the member in
        the Answer class. This is why a class is used: a local variable
        would go out of scope.
        """
        user_answer.answer = string
        root.destroy()  # destroy the window.

    query_buttons = list()

    for lab in labels.keys():
        # Construct the two buttons.
        query_buttons.append(tkinter.Button(
            root,
            text=query_button_texts[lab],
            command=lambda: button(user_answer, lab)))

    for idx, query_button in enumerate(query_buttons):
        query_button.grid(row=3, column=idx, sticky=tkinter.N)

    tkinter.mainloop()

    return user_answer.answer


def may_oracle(config, metric_tracker, epoch):
    """ Checks if all sampling preconditions are met."""
    loss_list = metric_tracker["T-loss"]

    recent_loss = loss_list[-config["oracle_loss_check"][0]:]
    if recent_loss and recent_loss[0] is None:
        recent_loss = recent_loss[1:]

    if config["min_epochs_after_plateau_oracle"]:
        loss_since_oracle = loss_list[metric_tracker["OracledEpochs"][-1]:]
        len_loss = len(loss_since_oracle)

        # Don't resample if too few epochs have passed since last.
        if len_loss <= config["min_epochs_after_plateau_oracle"]:
            return False

        # Do sample if there's a plateau.
        if mean(loss_since_oracle[-int(len_loss * 0.5):
                                  -int(len_loss * 0.25)]) * .5 <= \
                mean(loss_since_oracle[-int(len_loss * 0.25):]):
            return True

    # May sample if T-loss dipping too much
    if config["oracle_loss_check"][0]:
        if not recent_loss or \
                mean(recent_loss) < config["oracle_loss_check"][1]:
            return True

    # If not using special checks, then just interval check
    if not config["oracle_loss_check"][0] and \
            not config["min_epochs_after_plateau_oracle"]:
        # Sample every `oracle_interval' epochs
        if config["oracle_interval"] and config["oracle_size"] and \
                epoch % config["oracle_interval"]:
            return False
        return True

    # Passed all continuity checks -- may not oracle.
    return False


def oracle_initial(config, labels):
    """Ask the user to label the first set of images."""

    imgnames = reading.read_imgs(
        config=config,
        labels=labels)

    # Sample randomly from the unlabeled images.
    if config["initially_labeled"] <= 0:
        sample_size = len(imgnames)
    else:
        sample_size = min(config["initially_labeled"], len(imgnames))

    print(f"Labeling initial dataset ({sample_size} of {len(imgnames)} "
          "total)...")

    imgnames = random.sample(imgnames, sample_size)

    idx = 0
    for imgpath, label in tqdm(imgnames):
        if not config["auto_label"]:

            # Ask the user to define the label.
            label = ask_user_input(
                labels=labels,
                config=config,
                imgpath=imgpath,
                label=label,
                counter=f"Initial image {idx + 1} out of {len(imgnames)}")

        # Preprocess the image before saving it.
        img = prep.preprocess_image(config=config,
                                    image=Image.open(imgpath).convert("L"),
                                    labels=labels)

        # Save the image.
        img.save(os.path.join("data/processed",
                              config['dataset_name'],
                              "Train",
                              label,
                              os.path.basename(imgpath)))
        idx += 1

    # Save a plot showing the t-SNE of the processed dataset.
    if config["visualize_manifold"]:
        logging.visualize_manifold(
            imgpaths=imgnames,
            labels=labels,
            config=config)

    # Save a plot depicting the enabled stages of image preprocessing.
    if config["visualize_preprocessing"]:
        prep.preprocess_image(
            config=config,
            image=Image.open(random.choice(imgnames)[0]).convert("L"),
            show=True,
            labels=labels)


def oracle(config, labels, model, device, transform, epoch):
    """ Sample high-entropy images and query the user to label them."""

    if not config["initially_labeled"]:
        # If this if-statement passes, then there's nothing to query.
        return list()

    # Read all unlabeled images
    imgpaths = reading.read_imgs(
        config=config,
        labels=labels)

    n_add_at_epoch = config["oracle_size"]
    if config["max_labeled"]:
        n_train = reading.count_images(
            config=config,
            subdir=f"processed/{config['dataset_name']}/Train",
            labels=labels)

        n_add_at_epoch = min(config["oracle_size"],
                             config["max_labeled"] - n_train)
        n_add_at_epoch = max(0, n_add_at_epoch)

    # If no new images available, return an empty oracle batch.
    if not imgpaths or not n_add_at_epoch:
        return list()

    print(f"Sampling {n_add_at_epoch} new image(s) at epoch {epoch}.")

    # Optionally sample N images from the unlabeled images
    if config["entropy_sample"]:
        sample_size = min(config["entropy_sample"], len(imgpaths))
        imgpaths = random.sample(imgpaths, k=sample_size)

    # Get predictive entropy of all samples in the N unlabeled images
    imglist = [prep.preprocess_image(config=config,
                                     image=Image.open(img).convert("L"),
                                     labels=labels)
               for (img, _) in imgpaths]

    # Obtain the entropy and softmax for each image.
    pred_entropies = feeding.predictive_entropy(
        config=config,
        imgs=imglist,
        device=device,
        model=model,
        transform=transform)

    # Add the entropy field to the image path list.
    imgpaths = [(path, label, pred_entropies[idx]) for idx, (path, label)
                in enumerate(imgpaths)]

    # Take only the M images with the highest entropy.
    imgpaths = sorted(imgpaths,
                      key=lambda tup: tup[2][0],
                      reverse=True)

    sample_size = min(len(imgpaths), n_add_at_epoch)
    imgpaths = random.sample([(x, y, z) for (x, y, (_, z)) in imgpaths],
                             sample_size)

    if not config["auto_label"]:
        # Ask oracle to label the images.
        for idx, (imgpath, label, smax) in enumerate(imgpaths):
            oracled_label = ask_user_input(
                config=config,
                imgpath=imgpath,
                smax=smax,
                counter=f"Image {idx + 1} of {len(imgpaths)}",
                label=label,
                labels=labels)

            imgpaths[idx] = (imgpath, oracled_label)
    else:
        # Auto-labeling the images.
        imgpaths = [(path, label) for (path, label, _) in imgpaths]

    # Commit the oracled images to the train directory and return the
    # batch for training
    ret = list()
    for imgpath, label in imgpaths:
        img = prep.preprocess_image(
            config=config,
            image=Image.open(imgpath).convert("L"),
            labels=labels)
        ret.append((img, label, os.path.basename(imgpath)))
        img.save(os.path.join(f"data/processed/{config['dataset_name']}/Train",
                              label,
                              os.path.basename(imgpath)))

    return ret
