# RATT Generalization Bound
Code and results accompanying our paper titled RATT: Leveraging Unlabeled Data to guarantee generalization at ICML 2021 (Long Talk). If you find this repository useful or use this code in your research, please cite the following paper: 

> Garg, S., Balakrishnan, S., Kolter, Z., Lipton, Z. (2021). RATT: Leveraging Unlabeled Data to Guarantee Generalization. arxiv preprint arXiv:2105.00303.
```
@article{garg2021RATT,
    title={ {RATT}: Leveraging Unlabeled Data to Guarantee Generalization }
    author={Garg, Saurabh and Balakrishnan, Sivaraman and Kolter, Zico and Lipton, Zachary},
    year={2021},
    journal={arXiv preprint arXiv:2105.00303}
}
```
## Requirements

The code is written in Python and uses [PyTorch](https://pytorch.org/). To install requirements, setup a conda enviornment using the following command:

```setup
conda create --file requirements.txt
```

## Quick Experiments 

We simulate the setup for labeled and unlabeled data with the training sets of CIFAR-10, CIFAR-100, MNIST and IMDb. `train.py` file is the main entry point for training the model and run the code with the following command:

```setup
python train_PU.py --data-type="cifar_DogCat" --train-method="TEDn" --net-type="ResNet" --epochs=1000 --warm-start --warm-start-epochs=100 --alpha=0.5
```

Change the parameters to your liking and run the experiment. For example, change dataset with varying --data-type and vary algorithm with varying --train-method. We implement the BBE estimator in `estimator.py` and CVIR algorithm in `algorithm.py`.

## License
This repository is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Questions?

For more details, refer to the accompanying ICML 2021 paper (Long talk): [RATT: Leveraging Unlabeled Data to Guarantee Generalization](https://arxiv.org/abs/2105.00303). If you have questions, please feel free to reach us at sgarg2@andrew.cmu.edu or open an issue.  