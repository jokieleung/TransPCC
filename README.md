# TransPCC

This repository implements the algorithms described in our paper [TransPCC: Towards Deep Point Cloud Compression via Transformers]().

## How to get started (using Docker)

### Dependenices nvida-docker

Install nvida-docker and follow [these](https://stackoverflow.com/a/61737404)
instructions

## Data
You can download the dataset from [here](http://www.ipb.uni-bonn.de/html/projects/depoco/submaps.zip) povided by [depoco](https://github.com/PRBonn/deep-point-map-compression) and link the dataset to the docker container by configuring the Makefile

```sh
DATASETS=<path-to-your-data>
```

## Building the docker container

For building the Docker Container simply run 

```sh
make build
```

in the root directory.

## Running the Code

The first step is to run the docker container:

```sh
make run
```

The following commands assume to be run inside the docker container.

### Training

For training a network we first have to create the config file with all the parameters.
An example of this can be found in `/depoco/config/transPCC_demo.yaml`. 
Make sure to give each config file a unique `experiment_id: ...` to not override previous models.
To train the network simply run

```sh
python3 transPCC_trainer.py -cfg <path-to-your-config>
```

### Evaluation

Evaluating the network on the test set can be done by:

```sh
python3 evaluate.py -cfg <path-to-your-config>
```

All results will be saved in a dictonary.


## Citation

If you find this paper helps your research, please kindly consider citing our paper in your publications.

```bibtex
@inproceedings{liang2022transpcc,
  title={TransPCC: Towards Deep Point Cloud Compression via Transformers},
  author={Liang, Zujie and Liang, Fan},
  booktitle={Proceedings of the 2022 International Conference on Multimedia Retrieval (ICMR)},
  year={2022}
}
```
## Acknowledgment

This repo contains code modified from [depoco](https://github.com/PRBonn/deep-point-map-compression), Many thanks for their efforts.