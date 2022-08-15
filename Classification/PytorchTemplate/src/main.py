""" The main active learning system pipeline. Also used for sweeping."""

from __future__ import print_function
import os
import json
import time as time_
from itertools import product, combinations
from copy import deepcopy
from operator import mul
from functools import reduce
import torch
import torch.optim.lr_scheduler
from torchvision import transforms
from data import reading
from models import feeding, oracling, sweeping
from visualization import logging
from models.model import Model


def main(config):
    """ Initializes the model and runs a training/validation loop. """

    ####################################################################
    # Read file structure
    ####################################################################
    labels = reading.get_labels(config=config)
    n_classes = len(list(labels.keys()))

    ####################################################################
    # Load model
    ####################################################################

    # Use CUDA on GPU if it is available. Else, run the network on CPU
    use_cuda = config["cuda"] and torch.cuda.is_available()

    model = Model(config=config,
                  n_classes=n_classes,
                  use_cuda=use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Starting network on {device} at "
          f"{time_.strftime('%Y-%m-%d %H:%M:%S', time_.gmtime())}")

    # Define the augmentation transform for the training images
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5,
                                scale=(0.95, 1.05),
                                shear=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((config["img_size"][0], config["img_size"][1])),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # Define the augmentation transform for the testing images
    test_transform = transforms.Compose([
        transforms.Resize((config["img_size"][0], config["img_size"][1])),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    ####################################################################
    # Initialize metrics baseline
    ####################################################################

    if config["resume"]:
        # It is assumed that the data has previously been split.
        first_epoch, labels, metric_tracker = reading.resume_model(
            config=config,
            model=model)
    else:  # Not loading previous checkpoint.

        # Prepare the validation set. `labels' are the class labels.
        reading.prepare_val(config=config, labels=labels)

        # Oracle the first images.
        oracling.oracle_initial(config=config, labels=labels)

        first_epoch = 1

        # Initialize metrics tracker.
        metric_tracker = {
            'T-loss': [None],
            'V-loss': list(),
            'T-acc': [None],
            'V-acc': list(),
            'Time': [time_.gmtime()],
            'OracledEpochs': [first_epoch],
            'BestAccs': list()
        }

        # Get initial/baseline validation performance.
        metric_tracker = feeding.validate(config=config,
                                          model=model,
                                          metric_tracker=metric_tracker,
                                          device=device,
                                          transform=test_transform,
                                          labels=labels)

        # Count the number of labeled images for the tracker
        n_train = reading.count_images(
            config=config,
            subdir=f"processed/{config['dataset_name']}/Train",
            labels=labels)

        # Save intial performance as best performance so far.
        metric_tracker['BestAccs'].append(
            {
                'epoch': 0,
                'n_train': n_train,
                'val_acc': metric_tracker['V-acc'][-1]
            })

    ####################################################################
    # Main training and testing loop
    ####################################################################

    for epoch in range(first_epoch, config["epochs"] + 1):

        # Get the current learning rate from the optimizer
        curr_lr = [group['lr'] for group in model.optimizer.param_groups]

        # Train and obtain metrics
        metric_tracker, n_oracled = feeding.train(
            config=config,
            model=model,
            device=device,
            epoch=epoch,
            metric_tracker=metric_tracker,
            transform=train_transform,
            labels=labels,
            lrate=curr_lr)

        # Recount the number of labeled images
        n_train += n_oracled

        # Validation runs occur every `val_interval' epochs.
        if config["val_interval"] and epoch % config["val_interval"] == 0:

            # Validate on a batch from the validation images.
            metric_tracker = feeding.validate(
                config=config,
                model=model,
                metric_tracker=metric_tracker,
                device=device,
                transform=test_transform,
                labels=labels)

            # Save a line plot of some metric information.
            logging.plot_metrics(config=config,
                                 metric_tracker=metric_tracker,
                                 labels=labels,
                                 lrate=curr_lr,
                                 n_train=n_train)

            print(f"E{epoch}: \tValidated and plotted metrics... "
                  f"(V-acc = {metric_tracker['V-acc'][-1]:>2.0f}%)")

            # Backup best-performing model. Saves only if last V-loss is
            # lowest thus far.
            if metric_tracker['BestAccs'][-1]['val_acc'] \
                    < metric_tracker['V-acc'][-1]:
                metric_tracker['BestAccs'].append(
                    {
                        'epoch': epoch,
                        'n_train': n_train,
                        'val_acc': metric_tracker['V-acc'][-1]
                    })
                max_acc = metric_tracker['BestAccs'][-1]['val_acc']
                torch.save(
                    obj={
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'metric_tracker': metric_tracker,
                        'labels': labels},
                    f=f"models/savedmodel-{max_acc:.0f}.pt")
                print(f"* * * BEST MODEL SAVED! (V-acc = "
                      f"{max_acc:>6.2f}%) * * *")

        else:
            # Didn't validate this epoch; append Nones to validation metrics.
            metric_tracker["V-loss"].append(None)
            metric_tracker["V-acc"].append(None)

        # Save model regularly.
        if config["save_interval"] and epoch % config["save_interval"] == 0 \
                or epoch == config['epochs']:

            torch.save(
                obj={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'metric_tracker': metric_tracker,
                    'labels': labels},
                f="models/savedmodel.pt")

            print("* * * MODEL SAVED * * *")

        # Stop if there's a `stop at max image limit'.
        if config["stop_at_max_labeled"] and config["max_labeled"] and \
                config["max_labeled"] <= n_train:
            print("* * * REACHED IMAGE LIMIT! TRAINING HALTED... * * *")
            break

    # The main function returns the validation metrics after a main run.
    # Used in hyperparameter tuning.
    metric_tracker = feeding.validate(
        config=config,
        model=model,
        device=device,
        metric_tracker=metric_tracker,
        transform=test_transform,
        labels=labels)

    return metric_tracker


def tune_sweep(config, settings, filename, method):
    """ Sweeps over a range of hyperparameters and saves results.

    The `settings' parameter is a dictionary of parameter names and
    value lists.

    The `method' parameter defines the search algorithm. If it is set to
    `simple', then it will test the default settings while only changing
    every value for every parameter in the `settings'. If it is set to
    `grid', then it will try every possible combination of the values
    and parameters.
    """

    # Force network re-initialization between tuning runs.
    config["resume"] = False

    # List to store results of different runs for final multiplot.
    acc_multiplot_data = list()

    # The `simple' sweeping method loops over one parameter while
    # keeping all others at their default values.
    if method == 'simple':
        for (param, values) in settings.items():

            # Backup the default configuration to restore between
            # different parameters.
            new_config = deepcopy(config)

            # Loop over the given values.
            for value in values:
                print(f"* * * NOW TUNING WITH \"{param}\" = {value} * * *")

                # Set temporary configuration with a changed parameter.
                new_config[param] = value

                # Obtain the final validation metrics.
                metrics = main(new_config)
                best_accs = metrics['BestAccs']
                val_acc = metrics['V-acc'][-1]

                # Format the used parameter configuration and metrics.
                settings_list = '\t'.join([str(new_config[skey])
                                           for skey in settings.keys()])

                # Write the configuration and metrics to a file.
                with open(filename, "a") as file:
                    file.write(f"{settings_list}\t"
                               f"{best_accs[-1]['val_acc']:.3f}"
                               f"\t{val_acc:.3f}\n")

                # Save to multiplot.
                acc_multiplot_data.append((param, value, best_accs))
                sweeping.plot_sweep_multi(
                    config=config,
                    acc_multiplot_data=acc_multiplot_data)

    # The `grid' sweeping method is plain old grid search.
    elif method == 'grid':
        n_total = reduce(mul, [len(val) for val in settings.values()], 1)

        # Generate a list of all combinations of the sweeping settings.
        settings_prods = (dict(zip(settings, x)) for x in product(
            *settings.values()))

        # Loop over all combinations and validate.
        for subsetting_idx, subsetting in enumerate(settings_prods):
            print(f"* * * GRID TUNING {subsetting} "
                  f"({subsetting_idx + 1}/{n_total}) * * *")

            # Set temporary configuration with some changed parameters.
            new_config = deepcopy(config)
            for (param, value) in subsetting.items():
                new_config[param] = value

            # Obtain the final validation metrics.
            metrics = main(new_config)
            best_accs = metrics['BestAccs']
            val_acc = metrics['V-acc'][-1]

            # Format the used parameter configuration and metrics.
            settings_list = '\t'.join([str(new_config[skey])
                                       for skey in settings.keys()])

            # Write the configuration and metrics to a file.
            with open(filename, "a") as file:
                file.write(f"{settings_list}\t"
                           f"{best_accs[-1]['val_acc']:.3f}\t{val_acc:.3f}\n")

        # Obtain and test on all possible pairs in the settings.
        all_combinations = combinations(settings.keys(), r=2)
        for comb in all_combinations:
            sweeping.plot_sweep(config=config,
                                filename=filename,
                                par1=comb[0],
                                par2=comb[1])


if __name__ == '__main__':

    # Load in the config JSON.
    with open('references/config.json') as config_file:
        CONFIG = json.load(config_file)

    TUNE = False  # `False' to do single runs, `True' to perform sweeps.

    if TUNE:
        # Destination to store performance table.
        FILENAME = "references/tuning"

        # Enter the sweeping parameters below.
        SETTINGS = {
            "weight_decay": [2e-1, 5e-3, 1e-3, 5e-4],
            "lr": [1e-2, 1e-3, 1e-4]
            }

        # Remove old table if one exists.
        try:
            os.remove(FILENAME)
        except OSError:
            pass

        # Write header.
        with open(FILENAME, "a") as tunefile:
            TAB = '\t'
            tunefile.write(f"{TAB.join(SETTINGS.keys())}\tBest%\tFinal%\n")

        # Sweep over the settings and add to table file.
        tune_sweep(config=CONFIG,
                   settings=SETTINGS,
                   filename=FILENAME,
                   method='grid')

    else:  # Single run
        main(CONFIG)

# TODO: Finish ReadMe & write requirements
# TODO: Final run for report performance.
# TODO: Write report
# TODO: auto mkdir processed
