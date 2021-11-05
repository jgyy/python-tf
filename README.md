# python-tf

Learn how to use Google's Deep Learning Framework - TensorFlow with Python! Solve problems with cutting edge techniques!

## Tensorflow notes

This repo using tensorflow 2.6 and the methods used will be based of tf.compat.v1

## Conda commands

As this repository uses tensorflow 1.15, a virtual environment is created.

```sh
conda env create --file requirements.yaml
conda activate tf1
conda env update --name tf1 --file requirements.yaml
```

copy the libraries if there is an error creating the environment (only happens in windows OS)

```sh
cp /mnt/c/tools/miniconda3/Library/bin/libcrypto-1_1-x64.* /mnt/c/tools/miniconda3/DLLs
cp /mnt/c/tools/miniconda3/Library/bin/libssl-1_1-x64.* /mnt/c/tools/miniconda3/DLLs
```
