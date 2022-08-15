""" Utility functions related to image processing and plotting. """

import os
import random
import time as time_
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from skimage import img_as_float
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from data import reading

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def plot_val_batch(config, data, target, prediction, labels, smax, num=8):
    """ Visualizes sample from validation batch with results. """

    num_imgs = min(num, data.shape[0])
    fig = plt.figure(constrained_layout=True)
    gspec = fig.add_gridspec(2, num_imgs // 2)

    for idx in range(num_imgs // 2 * 2):
        axis = fig.add_subplot(gspec[0 if idx < num_imgs // 2 else 1,
                                     idx % (num_imgs // 2)])
        image = data[idx][0].cpu() if config["cuda"] else data[idx][0]
        image = image.numpy()
        image = image - np.min(image)
        image = image * 255 / image.max()
        image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')
        axis.imshow(image, cmap='gray')
        lab = list(labels)[target[idx].item()]
        out = list(labels)[prediction[idx].item()]
        text = [f"L = {lab}",
                f"Y = {out}",
                f"    ({100*max(smax[idx]).item():.0f}% sure)",
                "correct" if lab == out else "WRONG"]
        for line_idx, txt in enumerate(text):
            axis.text(0,
                      image.height+((1+line_idx)*image.height//10),
                      txt,
                      fontsize=7)
        axis.set_axis_off()

    plt.savefig("reports/figures/val_batch_sample.pdf")
    plt.close()


def plot_metrics(config, metric_tracker, labels, lrate, n_train):
    """Visualize the metric tracker in a multiplot."""

    # Initialize the figure layout
    figsize = (9, 14)
    sns.set(style="whitegrid", rc={'figure.figsize': figsize,
                                   'font.family': ['serif']})

    fig = plt.figure(constrained_layout=True)
    gspec = fig.add_gridspec(3, 2)

    ####
    # Plot the accuracy line plot.
    ####
    def impute(vals):
        """ Linearly imputes None values."""
        track = list()
        ret = list()
        latest = None
        for val in vals:
            if val is not None:
                latest = val
                new_track = track
                for idx, _ in enumerate(track):
                    diff = val - ret[-1]
                    new_track[idx] = ret[-1] if not diff else \
                        ret[-1] + (idx + 1) / ((1 + len(track)) / diff)
                new_track.append(latest)
                ret.extend(new_track)
                track = list()
            elif not ret:
                continue
            else:
                track.append(latest)
        return ret

    acc_ax = fig.add_subplot(gspec[0, 0])

    data = impute(metric_tracker['V-acc'])

    acc_ax.plot([1 + x for x in range(len(data))],
                data,
                linewidth=1.2,
                label='V-acc',
                color="black")

    # Baseline validation acc: random selection.
    baseline = 100 / len(list(labels.keys()))
    acc_ax.axhline(y=baseline,
                   c='black',
                   lw=0.5,
                   linestyle="dashed",
                   alpha=0.7)

    acc_ax.set(xlabel="Epoch",
               ylabel="Validation accuracy (%)",
               title="Validation accuracy")
    acc_ax.grid(True, which="both")

    acc_ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 10))
    acc_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    acc_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ####
    # Plot line_axishe loss line plot.
    ####

    loss_ax = fig.add_subplot(gspec[0, 1])

    for metric in ["T-loss", "V-loss"]:
        data = impute(metric_tracker[metric])

        loss_ax.plot([1 + x for x in range(len(data))],
                     data,
                     linewidth=1.2,
                     label=metric,
                     color="orange" if metric.startswith("V") else "blue",
                     alpha=1 if metric.startswith("V") else 0.4,
                     linestyle="solid" if metric.startswith("V") else "dashed")

    loss_ax.set(xlabel="Epoch",
                ylabel="Cross-entropy loss",
                title="Loss")
    loss_ax.set_yscale('log')
    loss_ax.grid(True, which="both")

    loss_ax.xaxis.set_major_locator(ticker.MultipleLocator(len(data) / 10))
    loss_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    loss_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    locmin = ticker.LogLocator(base=10,
                               subs=np.arange(0.1, 1, 0.1),
                               numticks=10)

    loss_ax.yaxis.set_minor_locator(locmin)
    loss_ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    for oracled_epoch in metric_tracker["OracledEpochs"]:
        acc_ax.axvline(x=oracled_epoch,
                       c='gray',
                       linewidth=0.5,
                       alpha=0.8)
        loss_ax.axvline(x=oracled_epoch,
                        linewidth=0.5,
                        c='gray',
                        alpha=0.8)

    loss_ax.legend()

    ####
    # Print the metrics in text.
    ####
    text_ax = fig.add_subplot(gspec[1, 0])

    best_acc = metric_tracker['BestAccs'][-1]

    text_ax.axis('off')

    text_general = (
        f"Latest timestamp: "
        f"{time_.strftime('%Y-%m-%d %H:%M:%S', time_.gmtime())}.\n"
        f"The best validation accuracy measured\n"
        f"  so far is {best_acc['val_acc']:>6.2f}% at epoch "
        f"{best_acc['epoch']}\n"
        f"  with {best_acc['n_train']} image(s).\n"
        f"Classes: {', '.join(list(labels.keys()))}\n"
        f"Validation baseline: {baseline:>3.0f}%\n"
        f"\nRunning values:\n"
        f"Epoch: {len(metric_tracker['T-loss'])}\n"
        f"Training set size: {n_train}\n"
        f"Learning rate: {lrate[0]:>7.2E}\n"
        f"Training loss: {metric_tracker['T-loss'][-1]:>9.3E}\n"
        f"Validation loss: {metric_tracker['V-loss'][-1]:>9.3E}\n"
        f"Training accuracy: {metric_tracker['T-acc'][-1]:>3.0f}%\n"
        f"Validation accuracy: {metric_tracker['V-acc'][-1]:>3.0f}%"
        )
    text_ax.text(x=0,
                 y=0,
                 s=text_general,
                 fontsize=12,
                 bbox=dict(boxstyle="square,pad=0.3",
                           fc="none",
                           ec="black"))

    ####
    # Print n_samples / max_accuracy tradeoff
    ####

    # Don't plot if using all samples from the start.
    if config["initially_labeled"]:

        tradeoff_ax = fig.add_subplot(gspec[1, 1])
        data_x = list()
        data_y = list()
        data_labels = list()

        for best_acc in metric_tracker["BestAccs"]:
            if not data_x or best_acc['n_train'] != data_x[-1]:
                # No previous best with this number of samples
                data_x.append(best_acc['n_train'])
                data_y.append(best_acc['val_acc'])
                data_labels.append(best_acc['epoch'])
            else:
                # Overwrite previous best with this number of samples
                data_x[-1] = best_acc['n_train']
                data_y[-1] = best_acc['val_acc']
                data_labels[-1] = best_acc['epoch']

        tradeoff_ax = sns.lineplot(x=data_x,
                                   y=data_y,
                                   color='black',
                                   linewidth=1,
                                   ax=tradeoff_ax)

        tradeoff_ax.set(xlabel="Number of samples",
                        ylabel="Best validation accuracy",
                        title="Samples v. accuracy (label = epoch)")

        for idx, label in enumerate(data_labels):
            tradeoff_ax.text(data_x[idx], data_y[idx], label)
        tradeoff_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ####
    # Dump config settings.
    ####
    config_ax = fig.add_subplot(gspec[2, :])
    config_ax.axis('off')
    text_config = [f"{par}: {val}\n" for (par, val) in config.items()]
    mid = len(text_config) // 2
    config_ax.text(x=0,
                   y=0,
                   s="System hyperparameters:\n\n"+''.join(text_config[:mid]),
                   fontsize=10)
    config_ax.text(x=0.4,
                   y=0,
                   s=''.join(text_config[mid:]),
                   fontsize=10)

    ########

    fig.savefig("reports/figures/metrics.pdf")
    plt.close()


def visualize_manifold(imgpaths, config, labels, pca_components=128,
                       perplexity=156, dim=2, sample=0, display_3d=False):
    """ Plots PCA + t-SNE embedding of images.

    Accepts list of (label, imgpath).
    """
    print("Visualizing embedding manifold...")
    imgpaths = list()
    for label in labels.keys():
        basepath = os.path.join(
            f"data/processed/{config['dataset_name']}/Train", label)
        subpaths = os.listdir(basepath)
        for subpath in subpaths:
            if reading.is_image_file(subpath):
                imgpaths.append(os.path.join(basepath, subpath))
    if sample:
        imgpaths = random.sample(imgpaths, k=min(len(imgpaths), sample))

    imglist = list()
    for path in imgpaths:
        for label in labels.keys():
            if label in path:
                img = Image.open(path).convert("L")
                img = img.resize(config["img_size"], Image.LANCZOS)
                img = img_as_float(img)
                img = np.ndarray.flatten(img)
                imglist.append((img, label))

    labellist = [label for (_, label) in imglist]
    imglist = np.stack([img for (img, _) in imglist])

    if pca_components:
        imglist = PCA(n_components=pca_components).fit_transform(imglist)

    embed = TSNE(n_components=dim,
                 perplexity=perplexity,
                 verbose=1).fit_transform(imglist)

    if dim == 2:
        sns.set()

        plt.subplots(figsize=(5, 5))
        for label in labels.keys():
            startpos = labellist.index(label)
            endpos = len(labellist) - list(reversed(labellist)).index(label)
            axis = sns.kdeplot(data=embed[startpos:endpos, 0].tolist(),
                               data2=embed[startpos:endpos, 1].tolist(),
                               shade=True,
                               shade_lowest=False,
                               n_levels=1,
                               alpha=1 / len(list(labels.keys())))

        axis = sns.scatterplot(x=embed[:, 0].tolist(),
                               y=embed[:, 1].tolist(),
                               hue=labellist,
                               markers="circle",
                               s=10,
                               alpha=1)

        fig = axis.get_figure()

    elif dim == 3:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        labellist = [labels[lab] for lab in labellist]
        axis.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c=labellist)

        if display_3d:
            # Rotate the axes and update
            for angle in range(0, 360):
                axis.view_init(30, angle)
                plt.draw()
                plt.pause(.001)

    fig = axis.get_figure()
    fig.savefig(f"reports/figures/embedding{dim}D.pdf")
    plt.close()
