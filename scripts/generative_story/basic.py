from typing import *
import pickle
from tqdm import tqdm
import os
import numpy as np
from scipy.special import softmax

np.random.seed(6)

alpha = 1
d_dim = 20
s_dim = d_dim  # assumption of basic model
num_doc = 10000
vocab_size = 3001
result_dir = "/home/lzy/proj/neural_persona/examples/toy/basic"

num_word_func = lambda: np.random.randint(100, 200)
num_entity_func = lambda: np.random.randint(1, 20)
entity_func = lambda x:  np.random.multivariate_normal(x, np.eye(x.shape[0]))

pk_output: List[Dict[str, np.ndarray]] = []

# global parameters
beta = np.random.dirichlet(np.ones(vocab_size) * alpha, d_dim)
b = softmax(np.ones(vocab_size))

for i in tqdm(range(num_doc)):
    # sample a document representation
    d = np.random.standard_normal(d_dim)
    theta = softmax(d)

    # sample number of entities in the document
    E_i = num_entity_func()
    # sample number of words in the document
    C_i = num_word_func()
    # calculate number of word per entity in the document
    C_ij = C_i // E_i

    # sample s_i from N(d, 1)
    s_i = np.random.multivariate_normal(d, np.eye(d.shape[0]), size=E_i)
    e_i = softmax(s_i, axis=-1)
    p_s_i = softmax(e_i @ beta + b, axis=-1)
    k = C_ij // 2

    # reconstruct entity BoW vector
    p_x = softmax(d @ beta + b, axis=-1)
    entities_vector = np.random.multinomial(C_ij, p_x, size=E_i)
    pk_output.append({"entities": entities_vector, "theta": theta, "e_i": e_i})

# create folder if not exists
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

np.random.shuffle(pk_output)
train = pk_output[:num_doc // 10 * 8]
dev = pk_output[num_doc // 10 * 8:]

pickle.dump(train, open(f"{result_dir}/train.pk", "wb"))
pickle.dump(dev, open(f"{result_dir}/dev.pk", "wb"))
