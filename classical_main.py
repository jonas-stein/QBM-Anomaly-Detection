#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from time import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.rbm import RBM
from src.utils import *

parser = argparse.ArgumentParser(
    description='Generate clustered datasets with outliers.')
parser.add_argument('-hn', '--hnodes',
                    metavar='INT',
                    help='Amount of hidden units for RBM model',
                    default=157,
                    type=int)

parser.add_argument('-e', '--epochs',
                    metavar='INT',
                    help='Epochs for training',
                    default=13,  # best fit for dataset l_o7_c5_d3_p200_v1
                    type=int)

parser.add_argument('-b', '--batchsize',
                    metavar='INT',
                    help='Batchsize for training',
                    default=10,  # best fit for dataset l_o7_c5_d3_p200_v1
                    type=int)

flags = parser.parse_args()

########## CONFIGURATION ##########
EPOCHS = flags.epochs
HIDDEN_UNITS = flags.hnodes
CD_K = 2
SEED = 77
QUANTILE = 0.95
BATCH_SIZE = flags.batchsize

CLUSTER = 5
DATASET_PATH = 'src/datasets/l_o7_c5_d3_p200_v1.npy'


start = time()

########## LOADING CUSTOM DATA ##########

data = import_dataset(DATASET_PATH)
# print(np.amin(data)) #TODO: Fix bug negative datapoints for variance 1.0 in generator

training_dataset, testing_dataset = np.split(np.array(data), [int(len(data)/2)])

training_data, training_labels = split_dataset_labels(training_dataset)


########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(training_data, HIDDEN_UNITS, CD_K, SEED, epochs=EPOCHS, trained=False, quantile=QUANTILE)

rbm.train_model(BATCH_SIZE)

#save_weights(model = rbm, weight_csv='weights', title="Weights", type_quantum=False)

########## GENERATING CUSTOM TRAIN DATA FOR LOGISTIC REGRESSION ##########

tensor_training_data = torch.from_numpy(RBM.binary_encode_data(training_data)[0])
tensor_training_labels = torch.from_numpy(np.array(training_labels))

batches_training_data = tensor_training_data.split(BATCH_SIZE)
batches_training_label = tensor_training_labels.split(BATCH_SIZE)

batches_training = list(zip(batches_training_data, batches_training_label))


########## GENERATING CUSTOM TEST DATA FOR LOGISTIC REGRESSION ##########

testing_data, testing_labels = split_dataset_labels(testing_dataset)

tensor_testing_data = torch.from_numpy(RBM.binary_encode_data(testing_data)[0])
tensor_testing_labels = torch.from_numpy(testing_labels)

batches_testing_data = tensor_testing_data.split(BATCH_SIZE)
batches_testing_label = tensor_testing_labels.split(BATCH_SIZE)

batches_testing = list(zip(batches_testing_data, batches_testing_label))


########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(training_data), HIDDEN_UNITS))
train_labels = np.zeros(len(training_data))
test_features = np.zeros((len(testing_data), HIDDEN_UNITS))
test_labels = np.zeros(len(testing_data))

for i, (batch, labels) in enumerate(batches_training):
    batch = batch.view(len(batch), rbm.num_visible)  # flatten input data

    train_features[i*BATCH_SIZE:i*BATCH_SIZE +
                   len(batch)] = rbm.sample_hidden_from_visible(batch)[1].cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(batches_testing):
    batch = batch.view(len(batch), rbm.num_visible)  # flatten input data

    test_features[i*BATCH_SIZE:i*BATCH_SIZE +
                  len(batch)] = rbm.sample_hidden_from_visible(batch)[1].cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()


########## CLASSIFICATION ##########
print('Classifying...')

sag = LogisticRegression(solver="sag", max_iter=500)
sag.fit(train_features, train_labels)
predictions = sag.predict(test_features)

print('Result: {0:.2%}'.format(
    sum(predictions == test_labels) / test_labels.shape[0]))


########## EXTRACTING OUTLIER AND CLUSTERPOINTS ##########

outliers = RBM.get_binary_outliers(
    dataset=testing_dataset, outlier_index=CLUSTER)
points = RBM.get_binary_cluster_points(
    dataset=testing_dataset, cluster_index=CLUSTER-1)


########## ENERGY COMPARISON ##########
print('Energy comparison...')

outlier_energy = []

for outlier in outliers:
    outlier = torch.from_numpy(np.reshape(outlier, (1, rbm.num_visible)))
    outlier_energy.append(rbm.free_energy(outlier).cpu().numpy().tolist())

outlier_energy = np.array(outlier_energy)

cluster_point_energy = []

for point in points:
    point = torch.from_numpy(np.reshape(point, (1, rbm.num_visible)))
    cluster_point_energy.append(rbm.free_energy(point).cpu().numpy().tolist())

cluster_point_energy = np.array(cluster_point_energy)

o = outlier_energy.reshape((outlier_energy.shape[0]))
c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

RBM.plot_energy_diff([o, c], rbm.outlier_threshold, "rbm_energies.pdf")

RBM.plot_hist(c, o, rbm.outlier_threshold, "rbm_hist.pdf")


########## OUTLIER CLASSIFICATION ##########
print('Outlier classification...')

predict_points = np.zeros(len(tensor_testing_data), dtype=int)

for index, point in enumerate(tensor_testing_data.split(1),0):
    point = point.view(1, rbm.num_visible)
    predict_points[index], _ = rbm.predict_point_as_outlier(point)

print("Predicted points test: ", predict_points)
true_points = np.where(testing_labels < CLUSTER, 0, 1)
accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(true_points,predict_points), recall_score(true_points, predict_points)
f1 = f1_score(true_points, predict_points)
tn, fp, fn, tp = confusion_matrix(true_points, predict_points, labels=[0, 1]).ravel()

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True Positive: {tp}, Num False Positive: {fp}')

end = time()
print(f'Wallclock time: {(end-start):.2f} seconds')


path = "energies/"
dir_list = os.listdir(path)

data = {"Advantage":[],"DW_2000Q":[],"Fujitsu":[],"SA":[]}
for file in dir_list:
    if file.endswith(".npy"):
        for key in data.keys():
            if key in str(file):
                data[key].append(np.load(path + file))

data["RBM"] = [c, o]

thresholds = [-45.721946607342616, -43.80789125236296, -228378.3863380747, -37.95750422218046, rbm.outlier_threshold]

keys = ["Advantage","DW_2000Q", "SA", "RBM"]
RBM.plot_energy_diff_multiple(data, keys, -0.55, -0.3, thresholds, 4, "Advantage_DWave_SA_RBM_energies_multiple.pdf")
