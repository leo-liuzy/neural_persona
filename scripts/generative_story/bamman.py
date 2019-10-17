from typing import *
import pickle
import json
from tqdm import tqdm
import os
import numpy as np
from scipy.special import softmax
# from torch.nn.functional import gumbel_softmax

np.random.seed(6)

alpha = 1
persona_per_topic = 4
K = 50
P = 100
d_dim = P * 2
num_doc = 10000
vocab_size = 3001
result_dir = f"/home/lzy/proj/neural_persona/examples/toy/basicK{K}P{P}"
vocabulary_fname = f"/home/lzy/proj/neural_persona/examples/toy/basic/vocabulary/entity_based.txt"

vocab = open(vocabulary_fname, "r").read().split("\n")[:-1]
idx2word = dict(enumerate(vocab))
num_word_func = lambda: np.random.randint(100, 200)
num_entity_func = lambda: np.random.randint(1, 20)
entity_func = lambda x:  np.random.multivariate_normal(x, np.eye(x.shape[0]))

pk_output: List[Dict[str, np.ndarray]] = []
json_output: List[Dict[str, str]] = []  # this is for making reference corpus

# global parameters
beta = np.zeros((K, vocab_size))
num_word_per_topic = vocab_size // K
# only a subset of vocabularis are corresponding to
for i in range(K):
    beta[i, i * num_word_per_topic:(i + 1) * num_word_per_topic] = 1
beta = softmax(beta, axis=-1)
# persona model is a matrix () that maps d_i -- a mixture of personas that we expect in document i --
# to a logit vector which we use to sample persona representations of characters in document i
persona_models = np.random.dirichlet(np.ones(P) * alpha, d_dim)
# b = softmax(np.ones(vocab_size))

for i in tqdm(range(num_doc)):
    # sample a document representation
    d = np.random.standard_normal(d_dim)
    theta = softmax(d)

    # this is the probability distribution from which we will sample entity representation
    p_i = theta @ persona_models

    # sample number of entities in the document
    E_i = num_entity_func()
    # sample number of words in the document
    C_i = num_word_func()
    # calculate number of word per entity in the document
    C_ij = C_i // E_i

    entity_repr_vectors = []
    entity_bow_vectors = []
    doc = []
    E = np.random.multinomial(1, p_i, size=E_i)
    for _ in range(E_i):
        # uniformly sample a center from [persona_per_topic * d_max_idx, persona_per_topic * (d_max_idx + 1))
        # persona representation
        e_j = np.random.multinomial(1, p_i, size=1)
        entity_repr_vectors.append(e_j)

        # calculate the probability over vocabulary
        p_s_j = softmax(e_j @ beta + b, axis=-1)
        bow_vector = np.random.multinomial(C_ij, p_s_j)
        sentence = []
        for idx in np.nonzero(bow_vector)[0]:
            word = idx2word[idx]
            sentence += [word] * bow_vector[idx]
        doc.append(" ".join(sentence))

        assert np.sum(bow_vector) != 0
        entity_bow_vectors.append(bow_vector)
    entity_repr_vectors = np.array(entity_repr_vectors)
    entity_bow_vectors = np.array(entity_bow_vectors)

    pk_output.append({"entities": entity_bow_vectors, "theta": theta, "e": entity_repr_vectors, "text": entity_bow_vectors.sum(0)})
    json_output.append({"text": " ".join(doc)})

# create folder if not exists
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
ziped = list(zip(pk_output, json_output))
np.random.shuffle(ziped)
pk_output, json_output = zip(*ziped)
train_pk = pk_output[:num_doc // 10 * 8]
dev_pk = pk_output[num_doc // 10 * 8:]

train_json = json_output[:num_doc // 10 * 8]
dev_json = json_output[num_doc // 10 * 8:]

pickle.dump(train_pk, open(f"{result_dir}/train.pk", "wb"))
pickle.dump(dev_pk, open(f"{result_dir}/dev.pk", "wb"))
json.dump(dict(zip(vocab, b)), open(f"{result_dir}/entity_based.bgfreq", "w"))

with open(f"{result_dir}/train.jsonl", "w") as f:
    for datum in train_json:
        json.dump(datum, f)
        f.write("\n")
with open(f"{result_dir}/dev.jsonl", "w") as f:
    for datum in dev_json:
        json.dump(datum, f)
        f.write("\n")

open(f"{result_dir}/train.txt", "w").write("\n".join([item["text"] for item in train_json]))  # for mallet
open(f"{result_dir}/dev.txt", "w").write("\n".join([item["text"] for item in dev_json]))