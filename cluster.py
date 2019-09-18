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


def infer_repr(archive_file, in_file, ontology, namespace):
    """

    :param labeled_data: [(docid, label(s), forward_output), ... ] each tuple is for one document.
    e.g. the output tensor is of size (1=num_document, num_entities, entity_dim). In our case, label is an entities' name
    :param ontology: is a lookup function f, that f(docid, label(char name)) will give the cluster index
    if not valid query(name is not labeled), return None
    :return: X, y
    """
    X = []
    y = []
    archive = load_archive(archive_file)
    model = archive.model
    dev = pickle.load(open(in_file, "rb"))

    model.eval()
    print("Inferring")
    for doc in tqdm(dev):
        docid = doc['docid']
        entities = doc["entities"]
        if len(entities) == 0:
            continue
        idxs = [ontology(docid, entity["label"]) for entity in entities]
        entities_text = np.asarray(np.stack([entity["text"].sum(0) for entity in entities]))

        tensor_input = torch.from_numpy(entities_text)
        # turn input into float tensor of size (batch_size=1, num_entity, vocab_size)
        tensor_input = tensor_input.float().unsqueeze(0)
        output_dic = model.vae(tensor_input)
        tensor_output = output_dic[namespace][0].detach().numpy()
        for i in range(len(idxs)):
            if idxs[i] is not None:
                X.append(tensor_output[i])
                y.append(idxs[i])
    return np.array(X), np.array(y)


def movies_ontology(table):
    import json
    docs = json.load(open("dataset/movies/doc_id2char_id_map.json", "r"))

    def f(docid: str, name: str):
        parts = name.split(" ")
        doc = docs[docid]
        char_fb_id = None
        for part in parts:
            if part in doc:
                char_fb_id = doc[part]
                break
        if char_fb_id is None or char_fb_id not in table:
            return None
        return table[char_fb_id]
    return f


charid2nameidx, char_name_lst = json.load(open("dataset/movies/charid2nameidx.json", "r"))
charid2tropeidx, trope_name_lst = json.load(open("dataset/movies/charid2tropeidx.json", "r"))
name_ontology = movies_ontology(charid2nameidx)
tvtrope_ontology = movies_ontology(charid2tropeidx)


if __name__ == "__main__":
    X, y = infer_repr(archive_file="archives/basic_ladder/movies/model.tar.gz",
                      in_file=f"examples/movies/entity_based/train.pk",
                      ontology=name_ontology,
                      namespace="e")

    print("Reducing entities with tsne")
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    target_ids = list(range(len(char_name_lst)))
    target = char_name_lst
    plt.figure(figsize=(6, 5))
    for i, label in zip(target_ids, target):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label)
    plt.legend()
    plt.show()
    # plt.scatter(X_2d[:, 0], X_2d[:, 1])
    plt.title("tsne_entity")
    plt.savefig("archives/basic_ladder/movies/tsne_inferred_entity_name_cluster")
    plt.show()

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