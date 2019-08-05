import json
from tqdm import tqdm
import itertools
import os
import numpy as np
from typing import List
from glob import glob
"""
This scripts is for creating d, d_e type of dataset
"""

data_home = "/home/lzy/proj/neural_persona/dataset/nyt/e2m/"
new_data_home = "/home/lzy/proj/neural_persona/examples/nyt"
filenames = [y for x in os.walk(data_home) for y in glob(os.path.join(x[0], '*.json'))]
n_data = len(filenames)
np.random.shuffle(filenames)
portion = n_data // 100
train_files = filenames[:80 * portion]
dev_files = filenames[80 * portion: 87 * portion]
test_files = filenames[87 * portion:]


def write_json_from_e2m_file(files: List[str], data_set: str = "train", model_type: str = "avitm"):
    doc_cover_pairs = []
    if not os.path.exists(f"{new_data_home}/{model_type}/"):
        os.makedirs(f"{new_data_home}/{model_type}/")
    assert model_type in ["avitm", "partial-gen"]
    assert data_set in ["train", "dev", "test"]
    data = []  # {"docs": [], "entities": [], "doc_reindex_table": {i: i for i in range(len(files))}}
    mentions = []
    for filename in tqdm(files):
        # doc_id = filename.split("/")[-1].split(".")[0]
        try:
            content = json.load(open(filename, "r"))
        except:
            print(f"Read Problem from Doc. {filename}")
            continue

        doc_sentences = [" ".join(tokens) for tokens in content["content"]]
        doc_len = len(doc_sentences)
        doc_text = " ".join(doc_sentences)
        # doc_idx = len(data["docs"])
        if model_type == "avitm":
            data.append({'text': doc_text})
            continue

        entities = content["entities"]
        all_mentions_idx = list(itertools.chain(*[entity["mentions"] for entity in entities]))
        doc_cover_pairs.append((doc_len, len(set(all_mentions_idx))))
        mentions.append(" ".join([doc_sentences[i] for i in all_mentions_idx]))
        for entity in entities:
            entity_label = entity["MID"]
            entity_text = " ".join([doc_sentences[i] for i in entity["mentions"]])
            datum = {"doc_text": doc_text, "entity_label": entity_label, "entity_text": entity_text}
            data.append(datum)
    with open(f"{new_data_home}/{model_type}/{data_set}.jsonl", "w") as f:
        for datum in data:
            json.dump(datum, f)
            f.write("\n")

    if model_type == "avitm":
        return
    with open(f"{new_data_home}/{model_type}/{data_set}_mentions.jsonl", "w") as f:
        json.dump(mentions, f)
    with open(f"{new_data_home}/{model_type}/{data_set}_doc_cover_pairs.json", "w") as f:
        json.dump(doc_cover_pairs, f)


# for d_e and d type of data feed
write_json_from_e2m_file(train_files, "train", "avitm")
write_json_from_e2m_file(dev_files, "dev", "avitm")
write_json_from_e2m_file(test_files, "test", "avitm")


write_json_from_e2m_file(train_files, "train", "partial-gen")
write_json_from_e2m_file(dev_files, "dev", "partial-gen")
write_json_from_e2m_file(test_files, "test", "partial-gen")
