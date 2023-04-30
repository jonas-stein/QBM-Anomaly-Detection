#!/usr/bin/env python

# enable to import from modules in the qtl-covid-classification folder when running from the command line
import argparse
import os
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

import src.utils
from src.qbm import QBM
from src.utils import *

DATASET_PATH = 'src/datasets/l_o7_c5_d3_p200_v1.npy'
CLUSTER = 5


def main(seed=77, n_hidden_nodes=128, solver="SA", sample_count=20, anneal=1000, beta_eff=1, epochs=3, batch_size=10, learning_rate=0.005, restricted=True, save=""):
    print("Start")

    random.seed(seed)
    print("Seed is " + str(seed))

    start = time.time()

    data = import_dataset(DATASET_PATH)
    training_dataset, testing_dataset = np.split(
        np.array(data), [int(len(data) / 2)])
    training_data, training_labels = split_dataset_labels(training_dataset)

    dataset_name = os.path.splitext(DATASET_PATH.split("/")[-1])[0]
    param_string = "_se" + str(seed) + "_h" + str(n_hidden_nodes) + "_sol" + solver + "_sc" + str(sample_count) + "_b" + str(
        beta_eff) + "_e" + str(epochs) + "_l" + str(learning_rate) + "_r" + str(restricted) + "_data" + dataset_name
    # create DQBM
    dqbm = QBM(seed=seed, data=training_data, epochs=epochs, n_hidden_nodes=n_hidden_nodes, solver=solver,
               restricted=restricted, sample_count=sample_count, anneal_steps=anneal, beta_eff=beta_eff, param_string=param_string, savepoint=save)
    # train
    print('Training QBM...')
    dqbm.train_model(batch_size=batch_size, learning_rate=learning_rate)

    # test
    testing_data, testing_labels = split_dataset_labels(testing_dataset)
    encoded_data_test, bits_input_vector, num_features = QBM.binary_encode_data(
        testing_data, use_folding=True)

    ########## EXTRACT FEATURES ##########
    print('Extracting features... Skipped')
    #train_features = dqbm.get_hidden_features(dqbm.encoded_data)
    #test_features = dqbm.get_hidden_features(encoded_data_test)

    ########## CLASSIFICATION ##########
    print('Classifying hidden activations as test... Skipped')

    #sag = LogisticRegression(solver="sag", max_iter=500)
    #sag.fit(train_features, training_labels)
    #predictions = sag.predict(test_features)

    #print('Result: {0:.2%}'.format(sum(predictions == testing_labels) / testing_labels.shape[0]))
    print("Result: Skipped")

    ########## ENERGY COMPARISON ##########
    print('Energy comparison & Outlier classification...')

    outliers = QBM.get_binary_outliers(
        dataset=testing_dataset, outlier_index=CLUSTER)
    predict_points_outliers = np.zeros(len(outliers), dtype=int)
    points = QBM.get_binary_cluster_points(
        dataset=testing_dataset, cluster_index=CLUSTER - 1)
    predict_points_cluster = np.zeros(len(points), dtype=int)

    print("Outlier threshold: ", dqbm.outlier_threshold)
    print("Calculate outlier Energy")
    outlier_energy = []
    for index, outlier in enumerate(tqdm(outliers), 0):
        outlier = np.reshape(outlier, (dqbm.dim_input))
        predict_points_outliers[index], this_outlier_energy = dqbm.predict_point_as_outlier(
            outlier)
        outlier_energy.append(this_outlier_energy)
    outlier_energy = np.array(outlier_energy)

    o = outlier_energy.reshape((outlier_energy.shape[0]))
    title = "qbm_energies" + dqbm.paramstring
    src.utils.save_output(title="outlier_" + title, object=o)
    print("Calculate cluster energy")
    cluster_point_energy = []
    for index, point in enumerate(tqdm(points), 0):
        point = np.reshape(point, (dqbm.dim_input))
        predict_points_cluster[index], this_cluster_point_energy = dqbm.predict_point_as_outlier(
            point)
        cluster_point_energy.append(this_cluster_point_energy)
    cluster_point_energy = np.array(cluster_point_energy)

    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

    src.utils.save_output(title="cluster_" + title, object=c)
    QBM.plot_energy_diff([o, c], dqbm.outlier_threshold, title + ".pdf")

    QBM.plot_hist(c, o, dqbm.outlier_threshold, "qbm_hist.pdf")

    ########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification: Results...')
    predict_points = np.concatenate(
        (predict_points_cluster, predict_points_outliers))

    print("Predicted points test: ", predict_points)

    true_points = np.concatenate(
        (np.zeros_like(cluster_point_energy), np.ones_like(outlier_energy)))

    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(
        true_points, predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(
        true_points, predict_points, labels=[0, 1]).ravel()

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True Positive: {tp}, Num False Positive: {fp}')

    end = time.time()
    print(f'Wallclock time: {(end-start):.2f} seconds')

    print("Outlier threshold: ", dqbm.outlier_threshold)
    print("Average clusterpoint energy: ", np.average(cluster_point_energy))
    print("Outlier energy: ", outlier_energy)


if __name__ == '__main__':
    # solver = 'SA' for local simulator
    # 'QBSolv' for testing UQO without losing coins
    # 'DW_2000Q_6' [max hnodes==94 vnodes==21] or 'Advantage_system4.1' [max hnodes==634 vnodes==21] for D-Wave Quantum Annealer (needs coins)
    # 'FujitsuDAU' for accessing Fujitsu Digital Annealing Unit (needs coins)
    #  when using FujitsuDAU, sample_count must be a value in [16, 128]
    parser = argparse.ArgumentParser(
        description='Generate clustered datasets with outliers.')
    parser.add_argument('-hn', '--hnodes',
                        metavar='INT',
                        help='Amount of hidden units for RBM model',
                        default=82,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    parser.add_argument('-e', '--epochs',
                        metavar='INT',
                        help='Epochs for training',
                        default=7,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    parser.add_argument('-b', '--batchsize',
                        metavar='INT',
                        help='Batchsize for training',
                        default=10,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

    flags = parser.parse_args()
    main(epochs=flags.epochs, n_hidden_nodes=flags.hnodes,
         batch_size=flags.batchsize,  solver='SA', restricted=False, seed=77)
