import json
import re

with open('./data/project-44-at-2025-05-02-19-50-8cbff96a.json', 'r') as file:
    data = json.load(file)

print("Total:", len(data))

with open('./data/cleaned_data.json', 'r') as file:
    data = json.load(file)

print("tot:", len(data))

with open('./data/results.json', 'r') as file:
    data = json.load(file)
unique_base_ids = set()

for entry in data:
    match = re.match(r"(\d+)_fish", entry["id"])
    if match:
        base_id = match.group(1)
        unique_base_ids.add(base_id)

print("segmented:", len(unique_base_ids))

print("fish in total:", len(data))


with open('./data/diff_output.json', 'r') as file:
    data = json.load(file)

print("special:", len(data))
