import json
import math

WIDTH = 4014
HEIGHT = 3016

with open("./data/badoutputs/badresults.json", "r") as f:
    results_data = json.load(f)

with open("./data/cleaned_data.json", "r") as f:
    cleaned_data = json.load(f)

cleaned_by_id = {str(entry["id"]): entry for entry in cleaned_data}

output = []

for entry in results_data:
    rid = entry["id"]
    if "_special" not in rid:
        continue

    numeric_id = rid.split("_")[0]

    if numeric_id not in cleaned_by_id:
        continue

    clean_entry = cleaned_by_id[numeric_id]

    def percent_to_pixel(p):
        return {"x": p["x"]/100 * WIDTH, "y": p["y"]/100 * HEIGHT}

    clean_snout = percent_to_pixel(clean_entry["snout"])
    clean_fork = percent_to_pixel(clean_entry["fork"])

    res_snout = entry["result"]["snout"]
    res_fork = entry["result"]["fork"]

    snout_diff = math.hypot(res_snout["x"] - clean_snout["x"], res_snout["y"] - clean_snout["y"])
    fork_diff = math.hypot(res_fork["x"] - clean_fork["x"], res_fork["y"] - clean_fork["y"])

    line_len = math.hypot(res_snout["x"] - res_fork["x"], res_snout["y"] - res_fork["y"])

    normalized_diff = (snout_diff + fork_diff) / line_len if line_len != 0 else None

    output.append({
        "id": rid,
        "snout_pixel_diff": snout_diff,
        "fork_pixel_diff": fork_diff,
        "normalized_pixel_diff": normalized_diff
    })

with open("./data/badoutputs/diff_output.json", "w") as f:
    json.dump(output, f, indent=2)