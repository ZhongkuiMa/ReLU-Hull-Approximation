import csv

import numpy as np


def get_means_stds(dataset):
    assert dataset in ["mnist", "cifar10"]
    if dataset == "mnist":
        return [0], [1]
    else:
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]


def get_samples(dataset):
    try:
        if dataset == "cifar10":
            csvfile = open(f'../data/{dataset}_test_5000.csv', 'r')
            print("Use the first 5000 examples.")
        else:
            csvfile = open(f'../data/{dataset}_test_full.csv', 'r')
            print("Use full examples.")
    except Exception:
        csvfile = open(f'../data/{dataset}_test.csv', 'r')
        print("Only the first 100 samples are available.")

    # csvfile = open(f'../data/{dataset}_test.csv', 'r')
    # print("Only the first 100 samples are available.")
    return csv.reader(csvfile, delimiter=',')


def normalize(image, means, stds, dataset, domain, is_conv):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds != None:
                image[i] /= stds[i]
    elif dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0]) / stds[0]
    elif (dataset == 'cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0]) / stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1]) / stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2]) / stds[2]
            count = count + 1

        is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            # for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count + 1
                image[i + 1024] = tmp[count]
                count = count + 1
                image[i + 2048] = tmp[count]
                count = count + 1


def denormalize(image, means, stds, dataset):
    if dataset in ['mnist', 'fashion']:
        for i in range(len(image)):
            image[i] = image[i] * stds[0] + means[0]
    elif dataset == 'cifar10':
        count = 0
        tmp = np.zeros(3072)
        for _ in range(1024):
            tmp[count] = image[count] * stds[0] + means[0]
            count += 1
            tmp[count] = image[count] * stds[1] + means[1]
            count += 1
            tmp[count] = image[count] * stds[2] + means[2]
            count += 1

        for i in range(3072):
            image[i] = tmp[i]
