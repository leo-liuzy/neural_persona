import os
import json
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
sns.set(style="darkgrid")
MODELS_LOG = "../model_logs"

models_for_plot = [x for x in os.listdir(MODELS_LOG)]
# fig, axes = plt.subplots(nrows=len(models_for_plot), ncols=4,  figsize=(15,3 * len(models_for_plot)))

measure2idx = {
    "nll": 0,
    "nkld": 1,
    "npmi": 2,
    "loss": 3
}

idx2meansure = {v: k for k, v in measure2idx.items()}

for i in tqdm(range(len(models_for_plot[1:2]))):
    model_dir = f"{MODELS_LOG}/{models_for_plot[i]}"
    print()
    experiments = sorted(os.listdir(model_dir), key=lambda x: int(x.split("_")[0][-1]), reverse=True)
    metrics = [None for i in range(len(experiments))]
    metrics_json = [None for i in range(len(experiments))]

    all_training_nll = []
    all_training_nkld = []
    all_training_npmi = []
    all_training_loss = []

    all_validation_nll = []
    all_validation_nkld = []
    all_validation_npmi = []

    all_validation_loss = []

    min_training_nll = []
    min_training_nkld = []
    max_training_npmi = []
    min_training_loss = []

    min_validation_nll = []
    min_validation_nkld = []
    max_validation_npmi = []
    min_validation_loss = []

    for experiment in experiments:
        experiment_dir = f"{model_dir}/{experiment}"
        metrics[i] = [y for x in os.walk(experiment_dir) for y in glob(os.path.join(x[0], 'metrics_epoch_*.json'))]
        metrics[i] = sorted(metrics[i], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        metrics_json[i] = [json.load(open(metric, "r")) for metric in metrics[i]]

        training_nll = [metric["training_nll"] for metric in metrics_json[i]]
        if len(training_nll) == 0:
            continue
        training_nkld = [metric["training_nkld"] for metric in metrics_json[i]]
        training_npmi = [metric["training_npmi"] for metric in metrics_json[i]]
        training_loss = [metric["training_loss"] for metric in metrics_json[i]]

        validation_nll = [metric["validation_nll"] for metric in metrics_json[i]]
        validation_nkld = [metric["validation_nkld"] for metric in metrics_json[i]]
        validation_npmi = [metric["validation_npmi"] for metric in metrics_json[i]]
        validation_loss = [metric["validation_loss"] for metric in metrics_json[i]]

        min_training_nll.append(min(training_nll))
        min_training_nkld.append(min(training_nkld))
        max_training_npmi.append(max(training_npmi))
        min_training_loss.append(min(training_loss))

        min_validation_nll.append(min(validation_nll))
        min_validation_nkld.append(min(validation_nkld))
        max_validation_npmi.append(max(validation_npmi))
        min_validation_loss.append(min(validation_loss))

        all_training_nll.append(training_nll)
        all_training_nkld.append(training_nkld)
        all_training_npmi.append(training_npmi)
        all_training_loss.append(training_loss)

        all_validation_nll.append(validation_nll)
        all_validation_nkld.append(validation_nkld)
        all_validation_npmi.append(validation_npmi)
        all_validation_loss.append(validation_loss)

    max_epoch = max([len(elm) for elm in all_training_nll])
    num_trial = len(all_training_nll)
    num_measure = len(measure2idx)

    train_result = [
        np.array([all_training_nll[i], all_training_nkld[i],
                  all_training_npmi[i], all_training_loss[i]])
        for i in range(num_trial)
    ]

    dev_result = [
        np.array([all_validation_nll[i], all_validation_nkld[i],
                  all_validation_npmi[i], all_validation_loss[i]])
        for i in range(num_trial)
    ]

    # no. of dev result and no. of train results are the same
    assert all(len(train_result[i]) == len(train_result[i]) for i in range(num_trial))

    # create masked array for summation
    train_arr = np.ma.empty((num_measure, max_epoch, num_trial))
    dev_arr = np.ma.empty((num_measure, max_epoch, num_trial))
    train_arr.mask = True
    dev_arr.mask = True

    for i in range(num_trial):
        train_x = train_result[i]
        dev_x = dev_result[i]

        train_arr[:train_x.shape[0], :train_x.shape[1], i] = train_x
        dev_arr[:dev_x.shape[0], :dev_x.shape[1], i] = dev_x

    ave_train = train_arr.mean(axis=2)
    ave_dev = dev_arr.mean(axis=2)

    ave_train_result = {idx2meansure[i]: ave_train[i] for i in range(num_measure)}
    ave_dev_result = {idx2meansure[i]: ave_dev[i] for i in range(num_measure)}

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3 * 1))
    epochs = np.arange(max_epoch)

    for i, metric_name in enumerate(measure2idx.keys()):
        axes[i].title.set_text(metric_name.upper())
        axes[i].plot(epochs, ave_train_result[metric_name], label=f"train_{metric_name}")
        axes[i].plot(epochs, ave_dev_result[metric_name], label=f"dev_{metric_name}")
        axes[i].legend()

    fig.suptitle(models_for_plot[i])
    plt.savefig(f"{models_for_plot[i]}_AVE.png")