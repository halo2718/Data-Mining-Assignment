import json
import os

root = r"C:\Users\lt\Desktop\Data-Mining-Assignment\src\hw1\crawl"
subdir = os.listdir(root)

final_json = {}
for sub in subdir:
    trg = os.path.join(root, sub, "data.json")
    with open(trg, "r") as F:
        json_data = json.load(F)
        final_json.update(json_data)

print(len(final_json.keys()))
with open("./text_data.json", "w") as F:
    json.dump(final_json, F)