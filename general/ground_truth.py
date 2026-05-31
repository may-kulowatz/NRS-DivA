import os
import zipfile
from tqdm import tqdm


def load_ground_truth_mind(behaviors_file):
    results = []
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            impr_id = int(cols[0])
            user_id = cols[1]
            candidates = cols[4].split()
            labels = [int(c.split("-")[1]) for c in candidates]
            article_ids = [c.split("-")[0] for c in candidates]
            results.append((impr_id, user_id, labels, article_ids))
    return results


def save_ground_truth_mind(results, output_file):
    with open(output_file, "w") as f:
        for impr_id, user_id, labels, article_ids in tqdm(results):
            clicked = [(i + 1, article_ids[i]) for i, label in enumerate(labels) if label == 1]
            positions = [str(pos) for pos, _ in clicked]
            ids = [aid for _, aid in clicked]
            f.write(f"{impr_id} {user_id} [" + ",".join(positions) + "] [" + ",".join(ids) + "]\n")

    zip_path = output_file.replace(".txt", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, arcname="prediction.txt")


if __name__ == "__main__":
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mind_dir = os.path.join(_project_dir, "data", "MIND")
    behaviors_file = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
    output_file = os.path.join(mind_dir, "prediction_ground_truth.txt")

    results = load_ground_truth_mind(behaviors_file)
    save_ground_truth_mind(results, output_file)
