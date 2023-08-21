# Material-NN


## Getting Started
Create a new virtual environment called "jax" that automatically installs the necessary dependencies.
```sh
conda env create -f environment.yml
```

To activate this environment,
```sh
conda activate jax
```


## Training

In the Conda environment,

Unzip `tqc-filtered.7z` and make sure the extracted file (`tqc-filtered.csv`) is in the root folder of this repo.

Now with any of the five neural network types:
* ccnn: crystal convolutional neural network
* ccnn2: another crystal convolutional neural network
* cgnn: crystal graph neural network
* csnn: crystal set neural network
* naive: multilayer perceptron

Prepare the data to train the model by running
```sh
python [neural_network_type]-data-preprocessing.py tqc-filtered [symmetry_type] [class_type]
```

where the following are necessary and must be replaced:
* `[neural_network_type]`: one of the four neural network types listed above
* `      [symmetry_type]`: the type of symmetry to predict. Must be one of `f`, `c`, or `n`
* `         [class_type]`: the class of material to predict. Must be one of `f` or `c`

Run one of the `[neural_network_type]-data-preprocessing.py` files with no parameters for more details. The output is stored in a pickle at `data/tqc-filtered-[symmetry_type]-[class_type]-[neural_network_type]-data.obj`

When the data has been prepared, begin training by running the following command:
```sh
python [neural_network_type]-ml.py data/tqc-filtered-[symmetry_type]-[class_type]-[neural_network_type]-data.obj
```

and training should magically happen without any issues at all.

If any issues do arise, let us know by submitting bug reports with the logs and stack trace on https://github.com/gnnop/material-NN/issues.