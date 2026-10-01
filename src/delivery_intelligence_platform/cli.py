"""Predict from a JSON acceptance event using a saved model bundle."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .predict import EtaPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="JSON object or nonempty list of objects")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = [payload] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        parser.error("Input must be one JSON object or a nonempty list of objects.")
    events = pd.DataFrame(records)
    result = EtaPredictor(args.model_dir).predict(events)
    print(result.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
