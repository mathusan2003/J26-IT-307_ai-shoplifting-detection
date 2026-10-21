
import json
from pathlib import Path

from src.risk_scoring import calculate_risk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "inputs" / "sample_behaviour.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "sample_risk_results.json"


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        behaviour_data = json.load(file)

    if not isinstance(behaviour_data, list):
        raise ValueError("Input JSON must contain a list of people.")

    risk_results = [
        calculate_risk(person)
        for person in behaviour_data
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(risk_results, file, indent=2)

    print(f"Processed people: {len(risk_results)}")
    print(f"Risk results saved to: {OUTPUT_PATH}")

    for result in risk_results:
        print(result)


if __name__ == "__main__":
    main()
