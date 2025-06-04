import json
from pathlib import Path

def load_diff_file(path):
    with open(path, 'r') as f:
        return {entry["id"]: entry for entry in json.load(f)}

def winner_label(val1, val2):
    if val1 < val2:
        return "NEW"
    elif val2 < val1:
        return "------"
    else:
        return "tie"

def compare_diffs(file1, file2):
    diffs1 = load_diff_file(file1)
    diffs2 = load_diff_file(file2)
    excluded_ids = {"200490_fish1_special"}
    common_ids = sorted((set(diffs1.keys()) & set(diffs2.keys())) - excluded_ids)
    # common_ids = sorted(set(diffs1.keys()) & set(diffs2.keys()))

    print(f"{'ID':<30} {'Snout Δ':>10} {'Winner':>12} {'Fork Δ':>10} {'Winner':>12} {'Norm Δ':>10} {'Winner':>12}")
    print("-" * 100)

    sum_1 = {"snout": 0.0, "fork": 0.0, "norm": 0.0}
    sum_2 = {"snout": 0.0, "fork": 0.0, "norm": 0.0}
    n = len(common_ids)

    for id in common_ids:
        a = diffs1[id]
        b = diffs2[id]

        snout1 = a["snout_pixel_diff"]
        snout2 = b["snout_pixel_diff"]
        fork1 = a["fork_pixel_diff"]
        fork2 = b["fork_pixel_diff"]
        norm1 = a["normalized_pixel_diff"]
        norm2 = b["normalized_pixel_diff"]

        sum_1["snout"] += snout1
        sum_1["fork"] += fork1
        sum_1["norm"] += norm1

        sum_2["snout"] += snout2
        sum_2["fork"] += fork2
        sum_2["norm"] += norm2

        print(f"{id:<30} "
              f"{abs(snout1 - snout2):10.3f} {winner_label(snout1, snout2):>12} "
              f"{abs(fork1 - fork2):10.3f} {winner_label(fork1, fork2):>12} "
              f"{abs(norm1 - norm2):10.3f} {winner_label(norm1, norm2):>12}")

    print("\n--- Averages across common IDs ---")
    print(f"{'Metric':<15} {'diff_output':>15} {'badoutputs':>15}")
    print(f"{'Snout':<15} {sum_1['snout']/n:15.3f} {sum_2['snout']/n:15.3f}")
    print(f"{'Fork':<15} {sum_1['fork']/n:15.3f} {sum_2['fork']/n:15.3f}")
    print(f"{'Normalized':<15} {sum_1['norm']/n:15.3f} {sum_2['norm']/n:15.3f}")
    print(n)

if __name__ == "__main__":
    file_a = Path("./data/diff_output.json")
    file_b = Path("./data/badoutputs/diff_output.json")
    compare_diffs(file_a, file_b)
