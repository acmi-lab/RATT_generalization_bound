import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms

from datasets import * 

def custom_dataset(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, true_target, mask
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        true_target = self.true_target
        mask = self.mask
        return data, target, true_target[index], mask[index]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def random_label_noise(data, targets, clean_size, noise_size, num_classes=10): 

    targets = np.array(targets)
    tot_size = len(targets)
    subset_size = clean_size + noise_size

    assert subset_size <= tot_size, "Clean size + noise size must be less than total size"

    subset_idx = np.random.choice(tot_size, subset_size, replace=False)


    data = data[subset_idx]
    targets = targets[subset_idx]

    true_targets = np.copy(targets)
    new_arr = np.random.randint(0,num_classes,len(targets))


    indices = np.random.choice(len(targets), size=noise_size, replace=False)
    bool_mask = np.zeros(len(targets))
    bool_mask[indices] = 1.0

    targets = np.multiply(new_arr, bool_mask) + np.multiply(targets, 1- bool_mask)
    targets = targets.astype(np.long)
    
    return data, targets, bool_mask, true_targets

def get_train_data(data_dir, dataset_name, clean_size, noise_size, transform= None):

    assert noise_size > 0, "Number of random samplies must be greater than 0"

    if dataset_name.lower() == "cifar10":
        custom_CIFAR10 = custom_dataset(CIFAR10)
        dataset = custom_CIFAR10(root = data_dir, train=True, transform=transform, download=True)

        dataset.data, dataset.targets, dataset.mask, dataset.true_target =\
             random_label_noise(dataset.data, dataset.targets, clean_size, noise_size, 10)

    elif dataset_name.lower() == "cifar100":
        custom_CIFAR100 = custom_dataset(CIFAR100)
        dataset = custom_CIFAR100(root = data_dir, train=True, transform=transform, download=True)

        dataset.data, dataset.targets, dataset.mask, dataset.true_target =\
                random_label_noise(dataset.data, dataset.targets, clean_size, noise_size, 100)

    elif dataset_name.lower() == "mnist":
        custom_MNIST = custom_dataset(MNIST)
        dataset = custom_MNIST(root = data_dir, train=True, transform=transform, download=True)
        
        dataset.data, dataset.targets, dataset.mask, dataset.true_target =\
                random_label_noise(dataset.data, dataset.targets, clean_size, noise_size, 10)

    elif dataset_name.lower().startswith("imdb"):
        train_texts, train_labels = read_imdb_split(f'./{data_dir}/aclImdb/train')

        custom_IMDbBERTData = custom_dataset(IMDbBERTData)
        dataset = custom_IMDbBERTData(train_texts, train_labels, transform=transform)

        dataset.data, dataset.targets, dataset.mask, dataset.true_target =\
                random_label_noise(dataset.data, dataset.targets, clean_size, noise_size, 2)

    else:
        raise NotImplementedError("Please add support for %s dataset" % dataset_name)

    return dataset

def get_test_data(data_dir, dataset_name, transform = None): 

    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root = data_dir, train=False, transform=transform, download=True)

    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100(root = data_dir, train=False, transform=transform, download=True)

    elif dataset_name.lower() == "mnist":
        dataset = MNIST(root = data_dir, train=False, transform=transform, download=True)

    elif dataset_name.lower().startswith("imdb"):
        test_texts, test_labels = read_imdb_split(f'./{data_dir}/aclImdb/test')

        dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

    else:
        raise NotImplementedError("Please add support for %s dataset" % dataset_name)

    return dataset


def get_tranform(dataset_name):
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset_name.lower() == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    elif dataset_name.lower().startswith("imdb"):
        transform = initialize_bert_transform('distilbert-base-uncased')

    else:
        raise NotImplementedError("Please add support for %s dataset" % dataset_name)

    return transform