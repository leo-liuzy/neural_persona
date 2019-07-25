import json
from tqdm import tqdm
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


def write_json_from_e2m_file(files: List[str], data_set: str = "train", data_type: str = "d"):
    if not os.path.exists(f"{new_data_home}/{data_type}/"):
        os.makedirs(f"{new_data_home}/{data_type}/")
    assert data_type in ["d", "d+e"]
    assert data_set in ["train", "dev", "test"]
    data = []  # {"docs": [], "entities": [], "doc_reindex_table": {i: i for i in range(len(files))}}
    for filename in tqdm(files):
        # doc_id = filename.split("/")[-1].split(".")[0]
        try:
            content = json.load(open(filename, "r"))
        except:
            print(f"Read Problem from Doc. {filename}")
            continue

        doc_tokens = content["content"]
        doc_text = " ".join([" ".join(tokens) for tokens in doc_tokens])
        datum = {"doc_text": doc_text}
        # doc_idx = len(data["docs"])
        if data_type == "d":
            data.append(datum)
            continue
        # TODO: make entity points to the doc that contains it, this save lots of spaces
        entities = content["entities"]
        for entity in entities:
            entity_label = entity["MID"]
            entity_text = [doc_tokens[i] for i in entity["mentions"]]
            datum = {"doc_text": doc_text, "entity_label": entity_label, "entity_text": entity_text}
            data.append(datum)
    # for name, lst in data.items():
    with open(f"{new_data_home}/{data_type}/{data_set}.json", "w") as f:
        for datum in data:
            json.dump(datum, f)
            f.write("\n")


# for d_e and d type of data feed
# write_json_from_e2m_file(train_files, "train", "d")
# write_json_from_e2m_file(dev_files, "dev", "d")
# write_json_from_e2m_file(test_files, "test", "d")


write_json_from_e2m_file(train_files, "train", "d+e")
write_json_from_e2m_file(dev_files, "dev", "d+e")
write_json_from_e2m_file(test_files, "test", "d+e")
