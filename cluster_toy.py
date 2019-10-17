from neural_persona.models import BasicL
from neural_persona.common import PROJ_DIR
from neural_persona.common.util import partition_labeling, movies_ontology, variation_of_information, purity
from allennlp.models.archival import load_archive
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('tkagg')
from tqdm import tqdm
import os
os.environ["DATA_DIR"] = "/proj/neural_persona/examples/nyt/entity_based/"
os.environ["VOCAB_SIZE"] = "3000"
os.environ["LAZY"] = "1"
os.environ["SEED"] = "6"
from environments import ENVIRONMENTS
import pickle
import json
import torch
import numpy as np
from math import log
from itertools import chain
from math import sqrt
from numpy import isnan


def infer_repr(archive_file, dataset, ontology, namespace):
    """

    :param labeled_data: [(docid, label(s), forward_output), ... ] each tuple is for one document.
    e.g. the output tensor is of size (1=num_document, num_entities, entity_dim). In our case, label is an entities' name
    :param ontology: is a lookup function f, that f(docid, label(char name)) will give the cluster index
    if not valid query(name is not labeled), return None
    :return: X, y
    """
    X = []
    y = []
    # print("Inferring")
    archive = load_archive(archive_file)
    model = archive.model
    # dev = pickle.load(open(in_file, "rb"))
    model.eval()
    print(f"K: {model.vae._decoder_topic.in_features}, P: {model.vae._decoder_topic.out_features}")
    for doc in dataset:
        true_e = doc["e"]
        entities = doc["entities"]
        if len(entities) == 0:
            continue
        idxs = ontology(true_e)
        entities_text = entities

        tensor_input = torch.from_numpy(entities_text)
        # turn input into float tensor of size (batch_size=1, num_entity, vocab_size)
        tensor_input = tensor_input.float().unsqueeze(0)
        output_dic = model.vae(tensor_input)
        tensor_output = output_dic[namespace][0].detach().numpy()
        for i in range(len(idxs)):
            if idxs[i] is not None:
                X.append(tensor_output[i])
                y.append(idxs[i])
    X = np.array(X)
    y = np.array(y)

    return X, y

argmax_ontology = lambda x: np.argmax(x, axis=1)
argmax_clustering = partition_labeling(lambda x: np.argmax(x, axis=1))
gold_clustering = partition_labeling(lambda x: x)


def describe_string(desc, percent: bool = False):
    mean = desc.mean * 100 if percent else desc.mean
    std = 0 if isnan(desc.variance) else sqrt(desc.variance)
    std = std * 100 if percent else std

    ret = f"{mean:.2f} +/- {std:.2f}"
    if percent:
        ret += " %"
    return ret


if __name__ == "__main__":
    import sys

    from glob import glob
    from scipy.stats import describe
    from pprint import pprint
    stdout_target = lambda m: open(f"eval_output_toy_{m}.txt", "w")
    model_dir_name_func = lambda m: lambda k, p: f"{PROJ_DIR}/archives/basic_ladder/toy/K{k}P{p}-{m}/"
    # train = pickle.load(open("examples/movies/entity_based_namefree/train.pk", "rb"))
    dev = pickle.load(open(f"{PROJ_DIR}/examples/toy/basic/dev.pk", "rb"))
    test = dev
    K_vals = [25, 50, 100]
    P_vals = [25, 50, 100]

    metrics_mean = {"VI": np.zeros((len(P_vals), len(K_vals))),
                    "Purity": np.zeros((len(P_vals), len(K_vals)))}
    metrics_std = {"VI": np.zeros((len(P_vals), len(K_vals))),
                   "Purity": np.zeros((len(P_vals), len(K_vals)))}
    max_size_mean = np.zeros((len(P_vals), len(K_vals)))
    max_size_std = np.zeros((len(P_vals), len(K_vals)))
    import matplotlib
    metrics = [("argmax", argmax_ontology)]

    for metric in ["e_npmi", "d_npmi", "loss"]:
        dir = model_dir_name_func(metric)
        sys.stdout = stdout_target(metric)
        for i, K in tqdm(list(enumerate(K_vals))):
            for j, P in tqdm(list(enumerate(P_vals))):
                model_dir = dir(K, P)
                pprint(model_dir)
                print(f"K: {K}, P: {P}")
                for name, ontology in tqdm(metrics):
                    VIs = []
                    purity_scores = []
                    max_sizes = []
                    for trial in tqdm(glob(f"{model_dir}/**/*.tar.gz")):
                        # print(trial)
                        type_of_model = model_dir.split("/")[-2]
                        dirname = os.path.dirname(trial)
                        num_trial = dirname.split("/")[-1]
                        fig, ax = plt.subplots()
                        figname = f"{type_of_model}-{num_trial}-pca.png"
                        X, y = infer_repr(archive_file=trial,
                                          dataset=test,
                                          ontology=ontology,
                                          namespace="e")
                        num_class = len(set(y))
                        pca = PCA(n_components=2)
                        X_r = pca.fit(X).transform(X)
                        #
                        colors = list(matplotlib.cm.colors.get_named_colors_mapping().values())[:num_class]
                        for k, c in zip(range(num_class), colors):
                            plt.scatter(X_r[y == k, 0], X_r[y == k, 1], c=c, label=str(k), s=5, alpha=.5)

                        plt.xlabel('PCA1'), plt.ylabel('PCA2'), ax.grid('on')
                        # plt.ylim([-4,4])
                        plt.title("PCA"), plt.legend(bbox_to_anchor=(1.05, 1))
                        plt.savefig(f"{dirname}/{figname}"), plt.close()

                        print(f"n_data: {len(y)}, gold_clustering: {name}")
                        algo_partitions = argmax_clustering(X)
                        max_sizes.append(max(len(lst) for lst in algo_partitions))
                        print(f"biggest partition size: {max(len(lst) for lst in algo_partitions)}")
                        gold_partitions = gold_clustering(y)
                        VI = variation_of_information(algo_partitions, gold_partitions)
                        purity_score = purity(algo_partitions, gold_partitions)
                        VIs.append(VI)
                        purity_scores.append(purity_score)
                    max_sizes_stats = describe(max_sizes)
                    VI_stats = describe(VIs)
                    purity_scores_stats = describe(purity_scores)
                    print(f"Metric {name}")
                    print(f"Variation of Information: {describe_string(VI_stats, percent=False)}")
                    print(f"Purity Score: {describe_string(purity_scores_stats, percent=True)}")
                    print(f"Max Sizes: {describe_string(max_sizes_stats, percent=False)}")
                    print()

                    max_size_mean[j, i] = max_sizes_stats.mean
                    max_size_std[j, i] = 0 if isnan(max_sizes_stats.variance) else sqrt(max_sizes_stats.variance)
                    metrics_mean["VI"][j, i] = VI_stats.mean
                    metrics_std["VI"][j, i] = 0 if isnan(VI_stats.variance) else sqrt(VI_stats.variance)
                    metrics_mean["Purity"][j, i] = purity_scores_stats.mean * 100
                    metrics_std["Purity"][j, i] = 0 if isnan(purity_scores_stats.variance) else sqrt(purity_scores_stats.variance) * 100

        pprint(f"metrics_mean: {metrics_mean}")
        pprint(f"metrics_std: {metrics_std}")
        pprint(f"max_size_mean: {max_size_mean}")
        pprint(f"max_size_std: {max_size_std}")

    visualize = False
    if visualize:
        print("Visualizing")
        tsne = TSNE(n_components=2, random_state=0)
        print("Reducing entities with tsne")
        X_2d = tsne.fit_transform(X)
        target_ids = list(range(len(char_name_lst)))
        target = char_name_lst
        plt.figure(figsize=(6, 5))
        count = 0
        for i, label in zip(target_ids, target):
            detect = y == i
            if sum(detect) > 2:
                count += 1
                plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label)
        plt.legend()
        # plt.scatter(X_2d[:, 0], X_2d[:, 1])
        plt.title(f"tsne_entity_({count})")
        plt.savefig(f"archives/basic_ladder/movies/tsne_inferred_entity_{name}_cluster")
        plt.show()

    # name_ontology("33059372", 'preston , who harbors a mysterious connection to chromeskull')

    # ontology = json.load(open("archives/all.json", "r"))
    # fb_id_table = {}
    # for idx, (k, v) in enumerate(ontology.items()):
    #     for (fb_id, count) in v:
    #         try:
    #             assert fb_id not in fb_id_table
    #             fb_id_table[fb_id] = idx
    #         except:
    #             print(fb_id)
    # points = pickle.load(open("archives/basic_ladder/inferred_entity_dev.pk", "rb"))
    # X = np.array([point[1].tolist() for point in points])
    # y = np.array([fb_id_table[point[0]] for point in points])
    #

    # points = pickle.load(open("archives/inferred_doc_dev.pk", "rb"))
    # X = np.array([point.squeeze(0).tolist() for point in points])

    # print("Reducing doc with tsne")
    # tsne = TSNE(n_components=2, random_state=0)
    # X_2d = tsne.fit_transform(X)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.title("tsne_doc")
    # plt.savefig("archives/tsne_inferred_doc")
    # plt.show()

    # pca = PCA(n_components=2, random_state=0)
    # X_2d = pca.fit_transform(X)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.title("pca")
    # plt.savefig("archives/pca_inferred_entity")
    # plt.show()