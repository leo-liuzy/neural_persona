import argparse
import json
import os
from typing import List, Dict, Any
import pickle
import nltk
import numpy as np
from itertools import chain
import pandas as pd
import spacy
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from neural_persona.common.util import read_text, save_sparse, write_to_json, save_named_sparse


def load_data(data_path: str, tokenize: bool = False, tokenizer_type: str = "just_spaces", entity: str = True) -> List[Dict[str, Any]]:
    if tokenizer_type == "just_spaces":
        tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en')
        tokenizer = Tokenizer(nlp.vocab)

    texts = []

    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            example = json.loads(line)
            sentences = example["text"]
            # sentences = [" ".join(sentence) for sentence in tokenized_text]
            text = []
            for sentence in sentences:
                if tokenize:
                    if tokenizer_type == 'just_spaces':
                        tokens = list(map(str, tokenizer.split_words(sentence)))
                    elif tokenizer_type == 'spacy':
                        tokens = list(map(str, tokenizer(sentence)))
                    sentence = ' '.join(tokens)
                text.append(sentence)

            if entity:
                assert "entities" in example
                entities = example["entities"]
                # new_entities = [{"label": entity["entity_label"], "text": text[entity["entity_text_ids"]]} for entity in entities]
                texts.append({"docid": example["docid"], "text": text, "entities": entities})
            else:
                texts.append({"docid": example["docid"], "text": text})

    return texts


def create_mentions(examples: List[Dict[str, Any]], unique_sentence: bool = False):
    result = []
    for example in examples:
        entities = example["entities"]
        all_mentions_idx = chain(*[entity["mentions"] for entity in entities])
        if unique_sentence:
            all_mentions_idx = set(all_mentions_idx)
        all_mentions_idx = list(all_mentions_idx)
        result.append(" ".join([example["text"][i] for i in all_mentions_idx]))
    return result


def collapse_entities(data_set):
    result = []
    for example in data_set:
        mat = example["text"]
        entities_idx = [entity["mentions"] for entity in example["entities"]]
        if len(entities_idx) == 0:
            continue
        entities = np.stack([mat[elm].sum(0) for elm in entities_idx])
        result.append(entities)
    return result


def extract_entity_from_doc_as_doc(data_set):
    result = []
    for example in data_set:
        entities_idx = list(chain(*[entity["mentions"] for entity in example["entities"]]))
        if len(entities_idx) == 0:
            continue
        mat = example["text"]
        new_example = np.asarray(mat[entities_idx].sum(0)).squeeze(0)
        result.append(new_example)
    return result


def extract_context_from_doc_as_doc(data_set):
    result = []
    for example in data_set:
        entities_idx = list(chain(*[entity["mentions"] for entity in example["entities"]]))
        mat = example["text"]
        context_idx = [i for i in range(mat.shape[0]) if i not in entities_idx]
        if len(context_idx) == 0:
            continue
        new_example = np.asarray(mat[context_idx].sum(0)).squeeze(0)
        result.append(new_example)
    return result


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
    parser.add_argument("--vocab-namespace", type=str, required=False, default="vampire",
                        help="Path to store the preprocessed output.")
    parser.add_argument("--vocab-size", type=int, required=False, default=10000,
                        help="Vocabulary set size")
    parser.add_argument("--tokenize", action='store_true',
                        help="Whether to tokenize the input")
    parser.add_argument("--tokenizer-type", type=str, default="just_spaces",
                        help="Tokenizer type: just_spaces | spacy")
    # naming convention: if you want a field called "doc text", you should name it "doc_text"
    # parser.add_argument("--token-field-names", type=str, nargs="*", default=["text"],
    #                     help="token field names separable by space like, \"doc\", \"entity_mentions\". "
    #                          "Naming Convention: if you want a field called \"doc text\","
    #                          " you should name it \"doc_text\"")
    parser.add_argument("--global-repr", action='store_true',
                        help="Whether use document level information")
    parser.add_argument("--use-mention", action='store_true',
                        help="Only use mentions")
    parser.add_argument("--entity", "-e", action='store_true',
                        help="Whether to model persona")

    args = parser.parse_args()

    global_repr = args.global_repr
    print("Using document information" if global_repr else "Discarding document information")
    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)

    ser_dir = args.serialization_dir
    if not os.path.exists(ser_dir):
        os.mkdir(ser_dir)
    vocabulary_dir = os.path.join(ser_dir, "vocabulary")
    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    # {token_field_name1: [ sentencen0-1, sentencen1-1, ...] ,
    #  token_field_name2: [ sentencen0-2, sentencen1-2, ...] }
    train_examples = load_data(args.train_path, args.tokenize, args.tokenizer_type, args.entity)
    dev_examples = load_data(args.dev_path, args.tokenize, args.tokenizer_type, args.entity)

    print("fitting count vectorizer...")
    count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size,
                                       token_pattern=r'\b[^\d\W]{3,30}\b')
    if not args.use_mention:
        text = [" ".join(example["text"]) for example in train_examples] + [" ".join(example["text"]) for example in dev_examples]
    else:
        train_mentions = create_mentions(train_examples, args.use_mention)
        dev_mentions = create_mentions(dev_examples, args.use_mention)
        text = train_mentions + dev_mentions

    # master is simply vectorized of the document corpus(no duplicate documents)
    master = count_vectorizer.fit_transform(text)
    max_entity_per_doc = max(len(example['text']) for example in train_examples + dev_examples)

    vectorized_train_examples = [{"docid": example["docid"],
                                  "text": sparse.hstack((np.array([0] * len(example["text"]))[:, None],
                                                         count_vectorizer.transform(example["text"]))).tocsc(),
                                  "entities": example["entities"],
                                  }
                                 for example in train_examples]
    vectorized_dev_examples = [{"docid": example["docid"],
                                "text": sparse.hstack((np.array([0] * len(example["text"]))[:, None],
                                                       count_vectorizer.transform(example["text"]))).tocsc(),
                                "entities": example["entities"]}
                               for example in dev_examples]
    # add @@unknown@@ token vector for both doc and entity representation
    # this decision is for code simplicity
    vectorized_train_examples = [{"docid": example["docid"],
                                  "text": np.asarray(example["text"].sum(0)).squeeze(0),
                                  "entities": [{"label": entity["name"],
                                                "text": example["text"][entity["mentions"]]}
                                               for entity in example["entities"]]}
                                 for example in vectorized_train_examples]
    vectorized_dev_examples = [{"docid": example["docid"],
                                "text": np.asarray(example["text"].sum(0)).squeeze(0),
                                "entities": [{"label": entity["name"],
                                              "text": example["text"][entity["mentions"]]}
                                             for entity in example["entities"]]}
                               for example in vectorized_dev_examples]

    # vectorized_train_examples = extract_entity_from_doc_as_doc(vectorized_train_examples)
    # vectorized_dev_examples = extract_entity_from_doc_as_doc(vectorized_dev_examples)
    # vectorized_train_examples = extract_context_from_doc_as_doc(vectorized_train_examples)
    # vectorized_dev_examples = extract_context_from_doc_as_doc(vectorized_dev_examples)

    # add @@unknown@@ token vector
    master = sparse.hstack((np.array([0] * master.shape[0])[:, None], master))

    vocab = ["@@UNKNOWN@@"] + count_vectorizer.get_feature_names()
    # generate background frequency
    print("generating background frequency...")
    # bgfreq = dict(zip(count_vectorizer.get_feature_names(), master.toarray().sum(1) / args.vocab_size))
    bgfreq = dict(zip(vocab, np.array(master.sum(0))[0] / master.sum()))

    print("saving data...")
    pickle.dump(vectorized_train_examples, open(os.path.join(ser_dir, "train.pk"), "wb"))
    pickle.dump(vectorized_dev_examples, open(os.path.join(ser_dir, "dev.pk"), "wb"))
    # np.save(os.path.join(ser_dir, "train.pk"), vectorized_train_examples)
    # np.save(os.path.join(ser_dir, "dev.pk"), vectorized_dev_examples)

    write_to_json(bgfreq, os.path.join(ser_dir, f"{args.vocab_namespace}.bgfreq"))
    
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
