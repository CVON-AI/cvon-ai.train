DALPAC 1.0
==============================

Deep Active Learning Pulmonary Artery Classifier (DALPAC) is an integrated system for training an active learning n-ary classifier, designed for user-friendly research purposes and flexibility. Supported functionalities include reading and processing images from the file system, querying the user for classification, training and validating a deep neural network model, metric visualization, and automated parameter sweeping.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

Overview of configuration settings
------------


##### ```auto_label```
Let the system label queried samples automatically if possible.

##### ```resume```
Let the system resume training from a previously saved checkpoint. This restores, including Train/Val splits, optimizer and schedulers. This is helpful for continuing an interrupted session, or for loading a best-performing model. Please note that for valid conclusions, hyperparameters may generally not be modified in-between resumptions.

##### ```initially_labeled```
The number of initially labeled samples. These images are randomly selected to stimulate a good random initial data distribution. Setting this value to 0 labels all samples, effectively causing fully-supervised training.

##### ```max_labeled```
The maximum number of labeled samples in the dataset. No queries after reaching this number.

##### ```stop_at_max_labeled```
Terminate training when reaching ```max_labeled```. Helpful in automatic parameter sweeping.

##### ```visualize_manifold```
Visualizes t-SNE data embedding to ```code_root```. See the
function in logging.py for additional parameters.

##### ```visualize_preprocessing```
Illustrate the preprocessing steps for a sample.

##### ```visualize_val_batch```
Visualize a validation batch subsample and output at every validation step.

##### ```certainty_threshold```
Log TRP/FPR/PPV/NPV for subset of samples with a certainty above this threshold.

##### ```weight_decay```
Regularization: L2 weight penalty.

##### ```n_dense```
Number of nodes in the two dense layers between the ResNet18 and the logits layer.

##### ```use_pretrained```
Use the ImageNet-pretrained ResNet18.

##### ```trainable_resnet```
Unfreezes the ResNet18.

##### ```optimizer```
Optimizer function. Select from ```SGD``` and ```AdamW```.

##### ```lr```
Learning rate.

##### ```scheduler```
Learning rate scheduler. Select from ```ReduceLROnPlateau``` and ```StepLR```.

##### ```gamma```
Learning rate multiplicative reduce factor.

##### ```gamma_step```
Learning rate scheduler step or minimum plateau size.

##### ```save_interval```
Network saving interval.

##### ```oracle_interval```
Standard oracle interval if T-loss preconditions disabled.

##### ```min_epochs_after_plateau_oracle```
Oracle precondition: oracle if T-loss plateaus after this many epochs. 0 = disable plateau.

##### ```oracle_loss_check```
Oracle precondition: Oracle if T-loss dips betlow 2nd field for number of epochs in 1st field. Set 1st field to 0 to disable.

##### ```log_interval```
Log information to terminal every X epochs.

##### ```val_interval```
Log validation information and update metrics every X epochs.

##### ```epochs```
Max number of epochs. Can extend using resume option.

##### ```cuda```
Use GPU CUDA cores.

##### ```entropy_sample```
Number of samples to test for in entropy subsampling.

##### ```entropy_pred_batch```
Minibatch size used in finding the entropies of all unlabeled samples.

##### ```oracle_size```
Number of samples added after every query.

##### ```batch_size```
Training minibatch size. Excluded ```oracle_size```, so this is only the number of previously labeled samples.

##### ```batch_size_val```
Validation minibatch size.

##### ```max_val_per_class```
Maximum number of validation samples per class.

##### ```mmt```
Momentum (used for SGD).

##### ```crop_dims```
Preprocessing: crop to ROI frame.

##### ```crop_to_mask```
Preprocessing: crop to a mask. See ```preprocessing.py```.

##### ```enhance_contrast```
Preprocessing: enhance the contract of the input image.

##### ```rotate_to_similar```
Preprocessing: rotate all images to MSE-minimized orientation.

##### ```edge_detect```
Preprocessing: Scharr edge detection and subsequent skeletonizing.

##### ```img_size```
Image input size to network.

##### ```dataset_name```
The name of the dataset directory in ```data/raw/```.

##### ```metadata_filename```
Metadata filename in ```references/```.


Running the code
------------

1. Place your dataset in ```data/raw``` with the following directory tree:
```
    └── raw
        └── MyDatasetName
            ├── Class1Name
            │   ├── Image1Name.png
            │   ├── Image2Name.png
            │   └── ...
            ├── Class2Name
            │   ├── Image3Name.png
            │   ├── Image4Name.png
            │   └── ...
            └── ...


```
Image data may be formatted as ```.png```, ```.jpg```, or ```.jpeg```.

2. If not using metadata, make sure the ```metadata_filename``` parameter is set to ```false```. If using metadata, then place a comma-delimited ```.csv``` file with the following format:

| Variable1  | Variable2  | ... | FileName   |
|------------|------------|-----|------------|
| Image1Val1 | Image1Val2 | ... | Image1Name |
| Image2Val1 | Image2Val2 | ... | Image2Name |
| ...        | ...        | ... | ...        |

such that the last column contains is a (sub)string of the corresponding image name. Place this file in ```references/``` and change the value of the ```metadata_filename``` correspondingly.

3. Install the required packages by running ```pip install requirements.txt```.

4. Optionally: change the hyperparameters in ```references/config.json```.

5. Run the code using ```python src/main.py```.

6. Model output will be stored and continually updated in ```reports/figures/```. The terminal will display intermediate training results.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
