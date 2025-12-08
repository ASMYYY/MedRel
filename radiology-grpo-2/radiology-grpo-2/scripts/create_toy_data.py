import json
from pathlib import Path

OUT_PATH = Path("data/toy_reports.jsonl")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    examples = [
        {
            "prompt": "FINDINGS: ",
            "reference": "The lungs are clear. No pleural effusion or pneumothorax is seen.",
        },
        {
            "prompt": "FINDINGS: ",
            "reference": "There is mild cardiomegaly with clear lung fields.",
        },
        {
            "prompt": "FINDINGS: ",
            "reference": "Left lower lobe consolidation consistent with pneumonia.",
        },
        {
            "prompt": "FINDINGS: ",
            "reference": "No acute cardiopulmonary abnormality identified.",
        },
    ]

    # duplicate a bit to have more samples
    data = examples * 50

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")

    print(f"Wrote {len(data)} toy samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
