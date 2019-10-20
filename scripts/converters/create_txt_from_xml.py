import json
from tqdm import tqdm
import itertools
import os
import re
import sys
import numpy as np
import csv
from typing import List
from glob import glob
import xml.etree.ElementTree as ET
from pprint import pprint
import gzip  # not strictly necessary

"""
This scripts is for creating d, d_e type of dataset
"""

data_home = "/home/lzy/proj/neural_persona/dataset/movies"
target_home = "/home/lzy/proj/frequency_probe/data"
xml_folder = f"{data_home}/corenlp_plot_summaries"
meta_folder = f"{data_home}/MovieSummaries"
character_meta = f"{meta_folder}/character.metadata.tsv"

xml_docs = glob(f"{xml_folder}/*")
docs = {}
actors = {}


def readMetadata(nameFile):
    global docs, actors
    file = open(nameFile, "r")
    print("Reading Metadata")
    for line in tqdm(file):
        cols = line.rstrip().lower().split("\t")
        if len(cols) > 12:
            id = cols[0]
            fbid = cols[10]
            actor = cols[12]
            name = cols[3]
            actors[fbid] = actor

            if id not in docs:
                docs[id] = {}

            parts = name.split(" ")
            for p in parts:
                docs[id][p] = fbid

    file.close()

# readMetadata(character_meta)


def corexmls_from_files(xml_docs):
    print("Processing Files")
    for doc_i, filename in tqdm(list(enumerate(xml_docs))):
        if doc_i % 100 == 0:
            sys.stderr.write('.')
        fp = gzip.open(filename, "r") if filename.endswith('.gz') else open(filename, "r")
        data = fp.read().decode('utf-8','replace').encode('utf-8')
        s = filename
        s = os.path.basename(s)
        s = re.sub(r'\.gz$', '', s)
        s = re.sub(r'\.txt\.xml$', '', s)
        s = re.sub(r'\.xml$', '', s)
        docid = s
        yield docid, data


def convert_sentences(doc_x):
    sents_x = doc_x.find('document').find('sentences').findall('sentence')
    sents = []
    for sent_x in sents_x:
        # sent_infos = {}

        toks_x = sent_x.findall(".//token")
        toks_j = [(t.findtext(".//word").lower(), t.findtext(".//POS"), t.findtext(".//NER")) for t in toks_x]
        sents.append(toks_j)  # join tokenized sentence and turn into lower-case
    return sents


class Entity(dict):
    def __hash__(self):
        return hash('entity::' + self['id'])


def convert_coref(doc_etree, sentences):
    coref_x = doc_etree.find('document').find('coreference')
    if coref_x is None:
        return []

    entities = []
    for entity_x in coref_x.findall('coreference'):
        mentions = []
        for mention_x in entity_x.findall('mention'):
            m = {}
            start = int(mention_x.find('start').text) - 1
            end = int(mention_x.find('end').text) - 1
            m['sentence'] = int(mention_x.find('sentence').text) - 1
            m['name'] = sentences[m['sentence']][start:end]
            # m['head'] = int(mention_x.find('head').text) - 1
            mentions.append(m)
        ent = Entity()
        ent['mentions'] = mentions
        # first_mention = min((m['sentence'], m['head']) for m in mentions)
        # ent['first_mention'] = first_mention
        # ent['id'] = '%s:%s' % first_mention
        entities.append(ent)
    return entities


if __name__ == "__main__":

    readMetadata(character_meta)
    processed_corpus = []
    num_missed_doc = 0
    num_ignored_doc = 0
    # among those included document...
    num_missed_entity = 0
    num_entity = 0
    # among those included entities...
    num_covered_mentions = 0
    num_uncovered_mentions = 0

    for docid, data in corexmls_from_files(xml_docs):
        if docid not in docs:
            num_missed_doc += 1
            continue
        doc_etree = ET.fromstring(data)
        sentences = convert_sentences(doc_etree)
        text = " ".join([" ".join([token[0] for token in sentence]) for sentence in sentences])
        processed_corpus.append(text)

    with open(f"{target_home}/plot_summaries_processed.txt", "w") as f:
        for datum in processed_corpus:
            f.write(datum)
            f.write("\n")
