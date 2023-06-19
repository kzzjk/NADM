# NAMD

## Installation

See [installation instructions](https://xmodaler.readthedocs.io/en/latest/tutorials/installation.html).

## Datasets

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) .

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) .

Please refer to the tools file for data processing and [xmodaler](https://github.com/YehLi/xmodaler) for data file location.

### Training

```
python train_net.py --num-gpus 1 \
 	--config-file configs/xray/mimic.yaml

```

## Acknowledgment

Our project references the codes in the following repos. Thanks for thier works and sharing.

- [xmodaler](https://github.com/YehLi/xmodaler)



