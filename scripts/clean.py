import json

with open("./data/project-44-at-2025-05-02-19-50-8cbff96a.json") as f:
    data = json.load(f)

filtered = []

for task in data:
    task_id = task.get("id")
    img_url = task.get("data", {}).get("img", "")
    
    for annotation in task.get("annotations", []):
        keypoints = annotation.get("result", [])

        snout_point = None
        fork_point = None

        for kp in keypoints:
            if kp.get("type") == "keypointlabels":
                labels = kp.get("value", {}).get("keypointlabels", [])
                x = kp.get("value", {}).get("x")
                y = kp.get("value", {}).get("y")

                if "Snout" in labels:
                    snout_point = {"x": x, "y": y}
                elif "Fork" in labels:
                    fork_point = {"x": x, "y": y}

        if snout_point and fork_point:
            filtered.append({
                "id": task_id,
                "img": img_url,
                "snout": snout_point,
                "fork": fork_point
            })
            break

with open("./data/cleaned_data.json", "w") as f:
    json.dump(filtered, f, indent=2)