"""Thanks for the guid from https://janakiev.com/blog/wikidata-mayors/"""

import requests
from tqdm import tqdm
import time
from pprint import pprint
from collections import Counter
import pickle
import json

url = 'https://query.wikidata.org/sparql'


def func(property: str, property_value: str):
    return lambda fb_id: f"""
                            SELECT ?item
                            WHERE
                            {{
                        
                                ?item wdt:P646 \"{fb_id}\".
                                ?item wdt:{property} {property_value}
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
                            }}
                          """


HUMAN = func("P31", "wd:Q5")
PLACE = func("P2046", "?islandArea")
# COUNTRY = func("P31", "Q6256")
# STATES = func("P31", "Q35657")
PARTY = func("P31", "wd:Q7278")
ORG = func("P856", "?officialWeb")

TYPES = [("HUMAN", HUMAN, []),
         ("PLACE", PLACE, []),
         ("PARTY", PARTY, []), ("ORG", ORG, [])
         , ("OTHER", None, [])]
points = json.load(open("archives/all_id_for_cluster.json", "r"))["OTHER"][:10]
# fb_ids = [point[0] for point in points]
# tmp = dict(Counter(fb_ids))
fb_ids = points   # {k: v for k, v in tmp.items() if v > 20}
bad_request = 0

for i, (fb_id, count) in tqdm(list(enumerate(fb_ids))):
    time.sleep(2)
    for j, (_, func, lst) in enumerate(TYPES):
        # time.sleep(0)
        if j == len(TYPES) - 1:
            lst.append((fb_id, count))
            break
        r = requests.get(url, params={'format': 'json', 'query': func(fb_id)})
        if r.status_code == 400:
            bad_request += 1
        # if (j % 5) + 1 == 0:
        #     # to avoid 429 error: too frequent requests
        #     time.sleep(1)
        try:
            data = r.json()["results"]["bindings"]
        except:
            continue
        if len(data) != 0:
            lst.append((fb_id, count))
            break


output_dic = {t[0]: t[-1] for t in TYPES}
# json.dump(output_dic, open("archives/all_id_for_cluster2.json", "w"))
# print(fb_ids)
print(
    f"""
    Total entities: {len(fb_ids)}
    Bad Request: {bad_request}
    Distribution: { {k: len(v) for k, v in output_dic.items()}}"""
)
