#!/usr/bin/env python

from pathlib import Path
import sys


# Accuracy: 0.9444444444444444, Precision: 0.0, Recall: 0.0, F1-Score: 0.0,
# Num True Negative: 476, Num False Negative: 3, Num True Positive: 0, Num False Positive: 25
# Wallclock time: 129.91 seconds
# Outlier threshold:  7.204306517353976


pathlist = [path for directory in range(1, len(sys.argv)) for path in Path(sys.argv[directory]).glob('epoch_*.txt')]
file_amount = len(pathlist)

with open("metrics.csv", 'w') as csv:

    data_labels = ["approach", "cpu", "hidden_units", "true_negative", "false_negative",
                   "true_positive", "false_positive", "precision", "recall",
                   "accuracy", "f1_score", "outlier_threshold",
                   "wallclock_time_seconds"]

    for i in range(len(data_labels)):
        csv.write(data_labels[i])
        if i < len(data_labels)-1:
            csv.write(',')
    csv.write('\n')

    i = 1
    for path in pathlist:
        print(f'File {i}/{file_amount}', end='\r')
        i += 1

        file = open(path, 'r')
        lines = file.readlines()
        file.close()

        metrics_dict = dict()

        metrics_dict['approach'] = 'classic' if 'RBM' in lines[0] else 'quantum'
        metrics_dict['cpu'] = "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"
        metrics_dict['hidden_units'] = path.stem.split("_")[1]

        basic_metrics = [metric.split(": ")[1]
                         for metric in lines[-5].strip().split(", ")]
        metrics_dict['true_negative'] = basic_metrics[0]
        metrics_dict['false_negative'] = basic_metrics[1]
        metrics_dict['true_positive'] = basic_metrics[2]
        metrics_dict['false_positive'] = basic_metrics[3]

        advanced_metrics = [metric.split(
            ": ")[1] for metric in lines[-6].strip()[:-1].split(", ")]
        metrics_dict['accuracy'] = advanced_metrics[0]
        metrics_dict['precision'] = advanced_metrics[1]
        metrics_dict['recall'] = advanced_metrics[2]
        metrics_dict['f1_score'] = advanced_metrics[3]

        quant_line = lines[-3].strip().split(":  ")
        metrics_dict['outlier_threshold'] = quant_line[1]

        time_line = lines[-4].strip().split(": ")
        metrics_dict['wallclock_time_seconds'] = time_line[1].split(" ")[0]

        csv_line = ""
        for label in data_labels:
            csv_line += metrics_dict[label]+","
        csv_line = csv_line[: -1]+"\n"
        csv.write(csv_line)
    print()
