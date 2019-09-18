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
    :return: X, y
    """
    X = []
    y = []
    docs = []
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

        for i in range(len(entities)):
            entity = entities[i]
            label = entity["label"]
            idx = ontology(docid, label)
            entity_text = np.asarray(entity["text"].sum(0))
            
        entities_label = [entity["label"] for entity in entities]
        tensor_input = torch.from_numpy(entities_text)
        # turn input into float tensor of size (batch_size=1, num_entity, vocab_size)
        tensor_input = tensor_input.float().unsqueeze(0)

        output_dic = model.vae(tensor_input)
        out.append((docid, entities_label, output_dic))
    pickle.dump(out, open(out_file, "wb"))


def cluster(ontology, labeled_data, namespace):
    """

    :param labeled_data: [(docid, label(s), forward_output), ... ] each tuple is for one document.
    e.g. the output tensor is of size (1=num_document, num_entities, entity_dim). In our case, label is an entities' name
    :param ontology: is a lookup function f, that f(label) will give the cluster index
    :return: X, y
    """
    X = []
    y = []
    docs = []
    for datum in labeled_data:
        docid, labels, output_dic = datum
        tensors = output_dic[namespace][0]
        assert len(labels) == tensors.shape[0]
        for i in range(len(labels)):
            label = labels[i]
            idx = ontology(docid, label)
            if idx is None:
                continue
            tensor = tensors[i].detach().numpy()
            X.append(tensor)
            y.append(idx)
            docs.append(docid)
    return np.array(X), np.array(y), docs


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


name_ontology = movies_ontology(json.load(open("dataset/movies/charid2nameidx.json", "r")))
tvtrope_ontology = movies_ontology(json.load(open("dataset/movies/charid2tropeidx.json", "r")))


if __name__ == "__main__":
    infer_repr(archive_file="archives/basic_ladder/movies/model.tar.gz",
               in_file=f"{os.getcwd()}/examples/movies/entity_based/dev.pk",
               out_file="archives/basic_ladder/movies/out.pk")

    # ontology = json.load(open("archives/all.json", "r"))
    # fb_id_table = {}
    # for idx, (k, v) in enumerate(ontology.items()):
    #     for (fb_id, count) in v:
    #         try:
    #             assert fb_id not in fb_id_table
    #             fb_id_table[fb_id] = idx
    #         except:
    #             print(fb_id)
    labeled_inferred_data = pickle.load(open("archives/basic_ladder/movies/out.pk", "rb"))
    cluster(name_ontology, labeled_inferred_data, "e")
    # points = pickle.load(open("archives/basic_ladder/inferred_entity_dev.pk", "rb"))
    # X = np.array([point[1].tolist() for point in points])
    # y = np.array([fb_id_table[point[0]] for point in points])
    #
    # print("Reducing entities with tsne")
    # tsne = TSNE(n_components=2, random_state=0)
    # X_2d = tsne.fit_transform(X)
    #
    # target = list(ontology.keys())[:2]
    # target_ids = list(range(len(target)))[:1]
    # plt.figure(figsize=(6, 5))
    # for i, label in zip(target_ids, target):
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label)
    # plt.legend()
    # plt.show()
    # # plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.title("tsne_entity")
    # plt.savefig("archives/basic_ladder/tsne_inferred_entity")
    # plt.show()

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