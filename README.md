## SuperHugeAAD
An all-in-one choice for preparing your EEG data and training your Auditory Attention Decoding models, both linear and nonlinear!

# Features
* Pytorch Lightning-based developement, facilitating the advantage of (one of) the most popular Deep Learning framework and its derivative(s).
* A unified entry for loading your EEG data. Prepare your EEG, audio, and other data modalities in numpy files (`.npy` or `.npy`). Write metadata for each file, link syncronized EEG and audio files, and load them in one dataset.
* User-defined preprocessing and transformations. Writing your own codes, or use our provided presets, including resample, sample- or subject-wise zscore, filtering, and other popular methods.
* Apply widely admitted leave-one-out cross-validation out of box. Don't worry about how to implement the complex experiment setups. Experienced users can also write their own codes to split data into training/validation/test folds.
* Bring any Pytorch model to our package, and use them freely without changing any code! We also provide many popular DNN models and a template to develope your own.
* Even training a linear regression or classification model with our toolbox. Lighting helps you with loading data, and controling the workflow. Write a `fit` and `predict` method is all you need based on our elaborated abstract base class (ABC).
* Log and monitor your training progress with predefined and customized metrics to csv, tensorboard, and other loggers. Decoding accuracy? F1 score? Pearson's correlation coefficients? Anything else?
* Enjoying the convinient of Lightning CLI. Start your training by simplily filling the config files.
* Share your codes with maximized reproductibility. Automatically seed your experiments based on your experiment configs, or use your lucky number to seed everything. Configuration and training progress will be saved to local files and can be restarted anytime.

# Requirements
* Our toolbox is developed on Windows 10, but it should works on other platforms.
* A computer to run python is expected to start training. If you want to start with your own datasets, you should also be accessible to MATLAB. Otherwise, you need to write a Python pipeline to prepare your data.
* CUDA devices are prefered to speed up your training. Lightning will handle everything such as DDP, `tensor.to(device)`.
* Our toolbox does not require additional system resources, so the requirements to CPU, RAM, and DISK are basically the same as you train a model in native Pytorch.
* Required packages are provided in `pyproject.yaml` file.

# Installation
* This package is still under rapid developement, and we recommemd a direct installation (symbolic link) to your anaconda environments.
To get start:
1. clone the repo
2. create a new conda environment (`$ conda create -n superhuge python=3.12`), activate it `$ conda activate superhuge`.
3. During our developement, we find Pylance unable to recognized packages installed in non-compatible editable mode. So here is our solution: Direct to `your\path\to\the\clone\repo\`, run `$ pip install -e . --config-settings editable_mode=compat`. Please feel free to start an issue helping us solving this problem.
4. [Alter] If you do not want to make any modification to our codes, you can directly run `pip install .` In this case, you can still write your own codes in the `.\train` folder. But any changes to `.\superhuge` would not take effects until you reinstall the package. You can still use our abc class and other templates to write your codes.

# Start training
1. Go to `.\train\main.py`
2. Change the model config to the one you like (if you want to use predefined models), or write your own models and a config file (simply following our existing `yaml` files).
3. Check that everthing else stated in the remaining configuration files are as you expected, such as `optimizers`, `lr_scheduler`, `num_epochs`.
4. Run `$ python .\train\main.py`. If you are in VS code, just `Run Python File in Terminal`, or press `F5` to start a debug session.

# Prepare your own data
Go to another repo [SuperPreparePublic] to download matlab files. Follow instructions there, and once everything is done, go back here and change the path to your data in the `data.yaml` file. Remember to check everything else correct.
Raise a PR to merge your preprocessing and preperation codes, and we are happy to make your scripts available for other users. Make sure to publish your raw or preprocess data in other places, such as `zenodo`, for others download.

# Write your own model
Just write model as you are in native Pytorch. That is to say, write a subclass to `torch.nn.Module`. Just follow the examples of our out-of-box models.
Hyperparameters can be adjusted by writing a config files, as you see in the `.\train\config\model` folder.
We recommend you implement your codes with maximum effort to separate the model structure and its hyperparameters. Write `def __init__(self, num_kernels, **kwargs)` instead of `def __init__(self, **kwargs)` and then define `num_kernels = 16` in your script.
Control as much hyperparameters as possible by yaml files.

Also, we kindly remind you that
* Remember to write a `**kwargs` in your `__init__` method. You can retrieve `num_channels` and other commonly used arguments by `kwargs["num_channels"]`. Take a look at the simple CNN example and you will find it well.
* If your model take both EEG and audio (envelope, mel) as inputs, make sure to properly define your `forward` method by type hinting. A simple example is `from superhuge.data import EEG_TYPE, AUDIO_TYPE ... def forward(self, eeg: EEG_TYPE, audio: AUDIO_TYPE)`.

# Other comments from authors
There are so many things to write a doc when we want to publish our toolbox. We are trying our best to cover as many corners as we can. If you find something not documented, please check the type annotations, go to their definitions and references, here from Internet and AI assistants, and feel free to write an issue.
This package is developed based on the idea of [https://github.com/miracleyoo/pytorch-lightning-template/tree/master].
