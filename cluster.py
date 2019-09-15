from neural_persona.models import BasicL
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


def infer_repr(archive_file, in_file, out_file1, out_file2):
    entities_pair_lst = []
    doc_lst = []
    archive = load_archive(archive_file)
    model = archive.model
    dev = pickle.load(open(in_file, "rb"))

    model.eval()
    print("Inferring")
    for doc in tqdm(dev):
        entities = doc["entities"]
        if len(entities) == 0:
            continue
        entities_text = np.asarray(np.stack([entity["text"].sum(0) for entity in entities]))
        entities_label = [entity["label"] for entity in entities]
        tensor_input = torch.from_numpy(entities_text)
        # turn input into float tensor of size (batch_size=1, num_entity, vocab_size)
        tensor_input = tensor_input.float().unsqueeze(0)

        output_dic = model.vae(tensor_input)
        e = output_dic["e"].squeeze(0)
        theta = output_dic["theta"]
        pairs = list(zip(entities_label, e))
        entities_pair_lst += pairs
        doc_lst.append(theta)
    pickle.dump(entities_pair_lst, open(out_file1, "wb"))
    pickle.dump(doc_lst, open(out_file2, "wb"))
    # print(len(entities_pair_lst))


if __name__ == "__main__":
    # infer_repr(archive_file="archives/basic_l_model.tar.gz",
    #            in_file=f"{os.getcwd()}/examples/nyt/entity_based/dev.pk",
    #            out_file1="archives/inferred_entity_dev.pk",
    #            out_file2="archives/inferred_doc_dev.pk")

    ontology = json.load(open("archives/all.json", "r"))
    fb_id_table = {}
    for idx, (k, v) in enumerate(ontology.items()):
        for (fb_id, count) in v:
            try:
                assert fb_id not in fb_id_table
                fb_id_table[fb_id] = idx
            except:
                print(fb_id)

    points = pickle.load(open("archives/basic_ladder/inferred_entity_dev.pk", "rb"))
    X = np.array([point[1].tolist() for point in points])
    y = np.array([fb_id_table[point[0]] for point in points])

    print("Reducing entities with tsne")
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    target = list(ontology.keys())[:2]
    target_ids = list(range(len(target)))[:1]
    plt.figure(figsize=(6, 5))
    for i, label in zip(target_ids, target):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label)
    plt.legend()
    plt.show()
    # plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.title("tsne_entity")
    plt.savefig("archives/basic_ladder/tsne_inferred_entity")
    plt.show()

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