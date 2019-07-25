import argparse
import json
import os
from typing import List, Dict

import nltk
import numpy as np
import pandas as pd
import spacy
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from neural_persona.common.util import read_text, save_sparse, write_to_json, save_named_sparse


def load_data(data_path: str, tokenize: bool = False, tokenizer_type: str = "just_spaces",
              token_field_names: List[str] = ["text"]) -> Dict[str, List[str]]:
    if tokenizer_type == "just_spaces":
        tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en')
        tokenizer = Tokenizer(nlp.vocab)

    named_tokenized_examples = {token_field_name: [] for token_field_name in token_field_names}

    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            for token_field_name in token_field_names:
                example = json.loads(line)
                assert token_field_name in example
                if tokenize:
                    if tokenizer_type == 'just_spaces':
                        tokens = list(map(str, tokenizer.split_words(example[token_field_name])))
                    elif tokenizer_type == 'spacy':
                        tokens = list(map(str, tokenizer(example[token_field_name])))
                    text = ' '.join(tokens)
                else:
                    text = example[token_field_name]
                named_tokenized_examples[token_field_name].append(text)
    return named_tokenized_examples


def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to the train jsonl file.")
    parser.add_argument("--dev-path", type=str, required=True,
                        help="Path to the dev jsonl file.")
    parser.add_argument("--train-mentions-path", type=str, required=False,
                        help="Path to the train mentions jsonl file.")
    parser.add_argument("--dev-mentions-path", type=str, required=False,
                        help="Path to the dev mentions jsonl file.")
    parser.add_argument("--serialization-dir", "-s", type=str, required=True,
                        help="Path to store the preprocessed output.")
    parser.add_argument("--data-type", type=str, required=False, default="d",
                        help="Path to store the preprocessed output.")
    parser.add_argument("--vocab-namespace", type=str, required=False, default="vampire",
                        help="Path to store the preprocessed output.")
    parser.add_argument("--vocab-size", type=int, required=False, default=10000,
                        help="Vocabulary set size")
    parser.add_argument("--tokenize", action='store_true',
                        help="Whether to tokenize the input")
    parser.add_argument("--tokenizer-type", type=str, default="just_spaces",
                        help="Tokenizer type: just_spaces | spacy")
    # naming convention: if you want a field called "doc text", you should name it "doc_text"
    parser.add_argument("--token-field-names", type=str, nargs="*", default=["text"],
                        help="token field names separable by space like, \"doc\", \"entity_mentions\". "
                             "Naming Convention: if you want a field called \"doc text\","
                             " you should name it \"doc_text\"")
    parser.add_argument("--global-repr", action='store_true',
                        help="Whether use document level information")

    args = parser.parse_args()
    assert args.data_type in ["d", "d+e"]

    global_repr = args.vocab_namespace in ["partial-gen"]
    print("Using document information" if global_repr else "Discarding document information")
    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    
    vocabulary_dir = os.path.join(args.serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    # {token_field_name1: [ sentencen0-1, sentencen1-1, ...] ,
    #  token_field_name2: [ sentencen0-2, sentencen1-2, ...] }
    token_field_names = args.token_field_names
    if not global_repr and args.data_type == "d+e":
        token_field_names = list(filter(lambda a: a != "doc_text", token_field_names))
    named_tokenized_train_examples = load_data(args.train_path, args.tokenize, args.tokenizer_type,
                                               token_field_names)
    named_tokenized_dev_examples = load_data(args.dev_path, args.tokenize, args.tokenizer_type,
                                             token_field_names)

    print("fitting count vectorizer...")
    count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size,
                                       token_pattern=r'\b[^\d\W]{3,30}\b')
    if global_repr:
        text = list(set(named_tokenized_train_examples["doc_text"])) + list(set(named_tokenized_dev_examples["doc_text"]))
    else:
        train_mentions = json.load(open(args.train_mentions_path))
        dev_mentions = json.load(open(args.dev_mentions_path))
        text = train_mentions + dev_mentions

    # master is simply vectorized of the document corpus(no duplicate documents)
    master = count_vectorizer.fit_transform(text)

    named_vectorized_train_examples = {token_field_name:
                                           count_vectorizer.transform(named_tokenized_train_examples[token_field_name])
                                       for token_field_name in token_field_names}
    named_vectorized_dev_examples = {token_field_name:
                                           count_vectorizer.transform(named_tokenized_dev_examples[token_field_name])
                                     for token_field_name in token_field_names}
    # add @@unknown@@ token vector for both doc and entity representation
    # this decision is for code simplicity
    for token_field_name in token_field_names:
        named_vectorized_train_examples[token_field_name] = sparse.hstack(
            (np.array([0] * len(named_tokenized_train_examples[token_field_name]))[:, None],
             named_vectorized_train_examples[token_field_name])
        )
        named_vectorized_dev_examples[token_field_name] = sparse.hstack(
            (np.array([0] * len(named_tokenized_dev_examples[token_field_name]))[:, None],
             named_vectorized_dev_examples[token_field_name])
        )
    # add @@unknown@@ token vector
    master = sparse.hstack((np.array([0] * len(text))[:, None], master))

    vocab = ["@@UNKNOWN@@"] + count_vectorizer.get_feature_names()
    # generate background frequency
    print("generating background frequency...")
    # bgfreq = dict(zip(count_vectorizer.get_feature_names(), master.toarray().sum(1) / args.vocab_size))
    bgfreq = dict(zip(vocab, np.array(master.sum(0))[0] / master.sum()))

    print("saving data...")
    save_named_sparse(named_vectorized_train_examples, os.path.join(args.serialization_dir, "train.npz"))
    save_named_sparse(named_vectorized_dev_examples, os.path.join(args.serialization_dir, "dev.npz"))

    write_to_json(bgfreq, os.path.join(args.serialization_dir, f"{args.vocab_namespace}.bgfreq"))
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(),
                       os.path.join(vocabulary_dir, f"{args.vocab_namespace}.txt"))
    write_list_to_file(['*tags', '*labels', args.vocab_namespace],
                       os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))


def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')


if __name__ == '__main__':
    main()
