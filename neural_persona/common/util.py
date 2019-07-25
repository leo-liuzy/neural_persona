import codecs
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from allennlp.data import Vocabulary
from scipy import sparse


def create_trainable_BatchNorm1d(num_features: int,
                                 weight_learnable: bool = False,
                                 bias_learnable: bool = True,
                                 momentum: float = 0.001,
                                 eps: float = 0.001,
                                 affine: bool = True):
    """

    :param num_features: C from an expected input of size (N,C,L) or L from input of size (N,L)
    :param weight_learnable: true of want gamma to be learnable
    :param bias_learnable: true of want beta to be learnable
    :param momentum: the value used for the running_mean and running_var computation.
                        Can be set to None for cumulative moving average (i.e. simple average)
    :param eps: a value added to the denominator for numerical stability.
    :param affine: a boolean value that when set to True, this module has learnable affine parameters.
    :return:
    """
    bn = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
    if not weight_learnable:
        bn.weight.data.copy_(torch.ones(num_features))
    bn.weight.requires_grad = weight_learnable
    # bias is initialized to be all zero
    bn.bias.requires_grad = bias_learnable
    return bn


def normal_kl(N0, N1, eps=1e-12):
    """
    (Roughly) A pragmatic translation of:
     https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    Note:
        - N0 and N1 are assumed to be of diagonal covariance matrix
        - N0 and N1 are of the same dimension
        - N0 and N1 are batch-first
    :param N0:
    :param N1:
    :return:
    """
    mu_0, log_var_0 = N0
    var_0 = log_var_0.exp()
    mu_1, log_var_1 = N1
    batch_size, k = mu_0.size()
    var_1 = log_var_1.exp()

    d = mu_1 - mu_0
    det_var_0 = var_0.prod(dim=1)
    det_var_1 = var_1.prod(dim=1)

    first_term = torch.sum(var_0 / var_1, dim=1)
    second_term = torch.sum(d ** 2 / var_1, dim=1)
    result = 0.5 * (first_term + second_term - k + torch.log(det_var_1 / det_var_0 + eps))

    return result


def compute_background_log_frequency(vocab: Vocabulary, vocab_namespace: str, precomputed_bg_file=None):
    """
    Load in the word counts from the JSON file and compute the
    background log term frequency w.r.t this vocabulary.
    """
    # precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
    log_term_frequency = torch.FloatTensor(vocab.get_vocab_size(vocab_namespace))
    if precomputed_bg_file is not None:
        with open(precomputed_bg_file, "r") as file_:
            precomputed_bg = json.load(file_)
    else:
        precomputed_bg = vocab._retained_counter.get(vocab_namespace)  # pylint: disable=protected-access
        if precomputed_bg is None:
            return log_term_frequency
    for i in range(vocab.get_vocab_size(vocab_namespace)):
        token = vocab.get_token_from_index(i, vocab_namespace)
        if token in ("@@UNKNOWN@@", "@@PADDING@@", '@@START@@', '@@END@@') or token not in precomputed_bg:
            log_term_frequency[i] = 1e-12
        elif token in precomputed_bg:
            log_term_frequency[i] = precomputed_bg[token]
    log_term_frequency = torch.log(log_term_frequency)
    return log_term_frequency


def log_standard_categorical(logits: torch.Tensor):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)

    Originally from https://github.com/wohlert/semi-supervised-pytorch.
    """
    # Uniform prior over y
    prior = torch.softmax(torch.ones_like(logits), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(logits * torch.log(prior + 1e-8), dim=1)

    return cross_entropy


def separate_labeled_unlabeled_instances(text: torch.LongTensor,
                                         classifier_text: torch.Tensor,
                                         label: torch.LongTensor,
                                         metadata: List[Dict[str, Any]]):
    """
    Given a batch of examples, separate them into labeled and unlablled instances.
    """
    labeled_instances = {}
    unlabeled_instances = {}
    is_labeled = [int(md['is_labeled']) for md in metadata]

    is_labeled = np.array(is_labeled)
    # labeled is zero everywhere an example is unlabeled and 1 otherwise.
    labeled_indices = (is_labeled != 0).nonzero()  # type: ignore
    labeled_instances["tokens"] = text[labeled_indices]
    labeled_instances["classifier_tokens"] = classifier_text[labeled_indices]
    labeled_instances["label"] = label[labeled_indices]

    unlabeled_indices = (is_labeled == 0).nonzero()  # type: ignore
    unlabeled_instances["tokens"] = text[unlabeled_indices]
    unlabeled_instances["classifier_tokens"] = classifier_text[unlabeled_indices]

    return labeled_instances, unlabeled_instances


def schedule(batch_num, anneal_type="sigmoid"):
    """
    weight annealing scheduler
    """
    if anneal_type == "linear":
        return min(1, batch_num / 2500)
    elif anneal_type == "sigmoid":
        return float(1/(1+np.exp(-0.0025*(batch_num-2500))))
    elif anneal_type == "constant":
        return 1.0
    elif anneal_type == "reverse_sigmoid":
        return float(1/(1+np.exp(0.0025*(batch_num-2500))))
    else:
        return 0.01


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data


def read_jsonlist(input_filename):
    data = []
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line, encoding='utf-8'))
    return data


def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = [x.strip() for x in input_file.readlines()]
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)


def save_sparse(sparse_matrix: sparse.spmatrix, output_filename: str):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def save_named_sparse(named_sparse_matrices: Dict[str, sparse.spmatrix], output_filename: str):
    assert all(sparse.issparse(matrix) for matrix in named_sparse_matrices.values())
    coo = {name: sparse_matrix if sparse.isspmatrix_coo(sparse_matrix) else sparse_matrix.tocoo()
           for name, sparse_matrix in named_sparse_matrices.items()}
    coo = {name: {"data": matrix.data,
                  "col": matrix.col,
                  "row": matrix.row,
                  "shape": matrix.shape}
           for name, matrix in coo.items()}
    np.savez(output_filename, **coo)


def load_named_sparse(input_filename, key):
    npy = np.load(input_filename)[key]
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()
