""" Utility functions related to parameter sweeping and plotting their
results.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


def plot_sweep(config, filename, par1=None, par2=None):
    """ Plot a heat map of the best validation accuracies of a
    two-variable parameter grid sweep.
    """
    par1_idx = 0
    par2_idx = 0
    vals = list()
    data = dict()

    # Read the csv file
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')

        for row in csv_reader:
            if par1_idx == par2_idx:  # Only initially; parses header.
                column_names = row

                # If parameters undefined, take first two parameters.
                if not par1 or par1 not in column_names:
                    par1 = column_names[0]
                if not par2 or par2 not in column_names:
                    par2 = column_names[1]

                par1_idx = row.index(par1)
                par2_idx = row.index(par2)
                acc_idx = row.index("Best%")
                continue

            # Read this row's parameters and accuracy.
            par1_val = row[par1_idx]
            par2_val = row[par2_idx]
            acc = row[acc_idx]

            # If multiple of these value combinations exist (e.g. multi-
            # var grid search), then take the best accuracy.
            for row_idx, (oldpar1, oldpar2, old_acc) in enumerate(vals):
                if oldpar1 == par1_val and oldpar2 == par2_val:
                    acc = max(acc, old_acc)
                    vals[row_idx] = (par1_val, par2_val, acc)
                    break
            else:
                vals.append((par1_val, par2_val, acc))

    # Transform the data.
    data = {
        par1: [x for (x, _, _) in vals],
        par2: [x for (_, x, _) in vals],
        "Best%": [float(x) for (_, _, x) in vals]
    }

    frame = pd.DataFrame(data, columns=[par1, par2, 'Best%'])
    frame = frame.pivot(index=par1, columns=par2, values='Best%')

    # Draw a heatmap with the numeric values in each cell
    _, axis = plt.subplots(figsize=(9, 6))

    axis = sns.heatmap(frame, annot=True, linewidths=.5, ax=axis)

    # Fix for mpl bug that cuts off top/bottom of seaborn viz
    bot, top = plt.ylim()
    plt.ylim(bot + .5, top - .5)

    fig = axis.get_figure()

    fig.savefig(os.path.join(
        config["code_root"], f"sweep_{par1}_{par2}.pdf"))
    plt.close()


def plot_sweep_multi(config, acc_multiplot_data):
    """For a `simple'-method parameter sweep, plot multiple
    sample-accuracy trade-offs in one plot."""

    figsize = (9, 9)
    sns.set(style="whitegrid", rc={'figure.figsize': figsize})
    fig = plt.figure(constrained_layout=True)
    gspec = fig.add_gridspec(2, 1)

    legend = list()
    line_axis = fig.add_subplot(gspec[0, :])

    # Loop over the trade-off lines.
    for param, value, data in acc_multiplot_data:
        data_x = list()
        data_y = list()
        data_labels = list()

        # Only append the best data to the plot data lists.
        for best_acc in data:
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

        line_axis = sns.lineplot(x=data_x,
                                 y=data_y,
                                 linewidth=1.5,
                                 label=f"{param}: {value}",
                                 ax=line_axis)

        for lab_idx, label in enumerate(data_labels):
            line_axis.text(x=data_x[lab_idx],
                           y=data_y[lab_idx],
                           s=label,
                           color=line_axis.get_lines()[-1].get_color())

        legend.append(f"{param}={value}")

    line_axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    line_axis.set(xlabel="Number of samples",
                  ylabel="Best validation accuracy",
                  title="Samples v. accuracy (label = epoch)")

    legend_axis = fig.add_subplot(gspec[1, :])
    legend_axis.axis('off')
    legend = '\n'.join([f"{idx}: {text}" for idx, text in enumerate(legend)])

    legend_axis.text(x=0, y=0, s=legend)

    fig.savefig(os.path.join(config["code_root"], "multisweep.pdf"))
    plt.close()
