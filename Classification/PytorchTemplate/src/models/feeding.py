""" Utility functions related to Pytorch. """

import os
import sys
import time as time_
from math import log2
from statistics import mean
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from data import reading
from visualization import logging
from models import oracling


def sample_batch(tv_type, config, metric_tracker, epoch, model, device,
                 transform, labels):
    """ Sample a batch with balanced classes, optionally from oracle."""

    batch_size = config["batch_size"] + config["oracle_size"] \
        if tv_type == "Train" else config["batch_size_val"]
    min_img_per_class = int(batch_size / len(labels))

    # Oracle batch, track class distribution.
    if tv_type == "Train" and oracling.may_oracle(config,
                                                  metric_tracker,
                                                  epoch):
        newly_labeled = oracling.oracle(config=config,
                                        model=model,
                                        device=device,
                                        transform=transform,
                                        labels=labels,
                                        epoch=epoch)
    else:
        newly_labeled = list()

    # If oracle() returned empty imglist, it didn't oracle after all.
    n_oracled = len(newly_labeled)

    # Get previously labeled images.
    for label in labels.keys():
        path = os.path.join(f"data/processed/{config['dataset_name']}"
                            f"/{tv_type}/{label}")
        imgs = os.listdir(path)
        if not imgs:  # Class not yet found in labeled
            continue
        # All classes must be equally represented.
        while len([lab for (_, lab, _) in newly_labeled if lab == label])  \
                < min_img_per_class:
            randimg = random.choice(imgs)
            img = Image.open(os.path.join(path, randimg))
            newly_labeled.append((img, label, randimg))

    # Augment the images.
    newly_labeled = [(transform(img), torch.tensor(labels[lab]), imgname)
                     for (img, lab, imgname) in newly_labeled]
    # Get the corresponding metadata.
    vectors = [reading.read_metadata(imgname=imgname, config=config)
               for (_, _, imgname) in newly_labeled]
    vectors = [torch.tensor(vec) for vec in vectors]

    data = [img for (img, _, _) in newly_labeled]
    target = [lab for (_, lab, _) in newly_labeled]

    # Shuffle the three lists in the same order.
    temp = list(zip(data, target, vectors))
    random.shuffle(temp)
    data, target, vectors = zip(*temp)

    data = torch.stack(data)
    vectors = torch.stack(vectors)
    target = torch.stack(target)

    return data, vectors, target, n_oracled


def train(config, model, device, epoch, transform, labels, lrate,
          metric_tracker):
    """ Peform a training epoch."""

    loss, accuracy, _, _, _, _, n_oracled = feed_batch(
        config=config,
        tv_type="Train",
        metric_tracker=metric_tracker,
        epoch=epoch,
        model=model,
        device=device,
        transform=transform,
        labels=labels)

    n_train = reading.count_images(
        config=config,
        subdir=f"processed/{config['dataset_name']}/Train",
        labels=labels)
    n_val = reading.count_images(
        config=config,
        subdir=f"processed/{config['dataset_name']}/Val",
        labels=labels)
    n_total = reading.count_images(
        config=config,
        subdir=f"raw/{config['dataset_name']}",
        labels=labels)
    n_total -= n_val

    # Print log regularly
    if config["log_interval"] and (epoch % config["log_interval"] == 0
                                   or epoch == 1):
        print(f"Train Epoch: {epoch:>4}"
              f" | Loss: {loss.item():>9.3E}"
              f" | Accuracy: {accuracy:>3.0f}%"
              f" | n_train = {n_train:>4}/{n_total}"
              f" | lr = {lrate[0]:>7.2E}")

    metric_tracker["T-loss"].append(loss.item())
    metric_tracker["T-acc"].append(accuracy)
    metric_tracker["Time"].append(time_.gmtime())

    if n_oracled:
        metric_tracker["OracledEpochs"].append(epoch)

    # Step the learning rate scheduler.
    if config["scheduler"] != "ReduceLROnPlateau":
        model.scheduler.step()

    if config["scheduler"] == "ReduceLROnPlateau":
        model.scheduler.step(mean(
            metric_tracker["T-loss"][1:][-config["val_interval"]:]))

    return metric_tracker, n_oracled


def feed_batch(config, tv_type, model, device, transform, labels=None,
               metric_tracker=None, epoch=None, pred_imgs=None):
    """Feed a batch through the network.

    `tv_type' can be "Train", "Val", or "Predict".
    """

    if tv_type in ["Train", "Val"]:
        data, vectors, target, n_oracled = sample_batch(
            tv_type=tv_type,
            config=config,
            metric_tracker=metric_tracker,
            epoch=epoch,
            model=model,
            device=device,
            transform=transform,
            labels=labels)

        # Allocate the tensor to CPU or GPU.
        target = target.to(device)

    else:  # Predicting unlabeled, so target/loss/accuracy unknown.
        pred_imgs = [transform(img) for img in pred_imgs]
        vectors = [torch.tensor(reading.read_metadata(imgname=None,
                                                      config=config))
                   for img in pred_imgs]  # Random vector

        data = torch.stack(pred_imgs)
        vectors = torch.stack(vectors)
        target = None
        n_oracled = 0

    # Allocate the tensors to CPU or GPU.
    data = data.to(device)
    vectors = vectors.to(device)

    if tv_type == "Train":
        model.train()
        output = model(data, vectors)
        model.optimizer.zero_grad()

    else:  # Validate and predict
        model.eval()
        with torch.no_grad():
            output = model(data, vectors)

    # If prediction, no need to return loss or target.
    if tv_type == "Predict":
        return output

    loss = F.cross_entropy(output, target, reduction='mean')

    if tv_type == "Train":
        loss.backward()
        try:
            model.optimizer.step()
        except KeyError:
            print(f"Error: could not correctly load {config['optimizer']} "
                  "optimizer. Did you resume with a different one?")
            sys.exit()

    prediction = output.argmax(dim=1, keepdim=True)
    correct = prediction.eq(target.view_as(prediction)).sum().item()
    loss /= len(data)
    accuracy = 100. * correct / len(data)

    return loss, accuracy, data, output, target, prediction, n_oracled


def validate(config, model, device, transform, labels, metric_tracker):
    """ Perform a validation epoch."""

    loss, accuracy, data, output, target, prediction, _ = feed_batch(
        tv_type="Val",
        config=config,
        model=model,
        device=device,
        transform=transform,
        labels=labels)

    if config["visualize_val_batch"]:
        smax = F.softmax(output, dim=1)
        logging.plot_val_batch(config=config,
                               data=data,
                               target=target,
                               prediction=prediction,
                               labels=labels,
                               smax=smax)

    if config["certainty_threshold"]:
        certainties = smax.cpu().numpy().max(axis=1)
        mask = np.asarray(
            certainties >= config["certainty_threshold"])
        target_masked = target.cpu().numpy()[mask]
        if target_masked.size:
            pred_masked = np.squeeze(prediction.cpu().numpy())[mask]

            tpos = np.extract(target_masked * pred_masked, pred_masked).size
            tneg = np.extract(target_masked + pred_masked, pred_masked).size
            fpos = np.extract(target_masked < pred_masked, pred_masked).size
            fneg = np.extract(target_masked > pred_masked, pred_masked).size

            print(f"Certainty filter: {target_masked.size}/{certainties.size}"
                  f" ({target_masked.size/certainties.size*100:.0f}%) left, "
                  f"TPR: {0 if not tpos+fneg else tpos/(tpos+fneg):.2f}, "
                  f"TNR: {0 if not tneg+fpos else tneg/(tneg+fpos):.2f}, "
                  f"PPV: {0 if not tpos+fpos else tpos/(tpos+fpos):.2f}, "
                  f"NPV: {0 if not tneg+fneg else tneg/(tneg+fneg):.2f}")

    # Save metrics to print and plot training and validation
    # information.
    metric_tracker["V-loss"].append(loss)
    metric_tracker["V-acc"].append(accuracy)

    return metric_tracker


def predictive_entropy(config, imgs, model, device, transform):
    """ Calculates predictive entropy list for unlabeled images."""
    entropies = list()

    while imgs:

        output = feed_batch(pred_imgs=imgs[:config["entropy_pred_batch"]],
                            model=model,
                            tv_type="Predict",
                            device=device,
                            transform=transform,
                            config=config)

        imgs = imgs[config["entropy_pred_batch"]:]

        smax = F.softmax(output, dim=1)
        output = torch.max(smax, dim=1)[0]
        for idx, max_prob in enumerate(output.tolist()):
            entropies.append((-max_prob * log2(max_prob) - (1 - max_prob) *
                              log2(1 - max_prob) if max_prob < 1 else 0,
                              smax[idx]))
    return entropies
