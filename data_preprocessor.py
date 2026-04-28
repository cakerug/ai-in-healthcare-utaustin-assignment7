#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

from lib.data_loader import (
    PROCESSED_GT_PATH,
    PROCESSED_PATIENTS_PATH,
    PROCESSED_TRIALS_PATH,
)

# Paths relative to project root
INPUT_TRIALGPT_MATCHING_RESULTS_PATH = Path(
    "dataset/original/trialgpt/train-00000-of-00001.parquet"
)
INPUT_TRIALGPT_PARSED_TRIAL_CORPUS_PATH = Path(
    "dataset/original/trialgpt/sigir/corpus.jsonl"
)
INPUT_TRIALGPT_PARSED_PATIENT_CORPUS_PATH = Path(
    "dataset/original/trialgpt/sigir/queries.jsonl"
)


# I used different labels from TrialMatchAI because I felt they were more clear
# An area of future work is on which labels work better
# We also modified the TrialGPT prompt to include exclusion and inclusion criteria in one
# prompt because IMO that's a better baseline. The choice to split them is already a prompting decision that muddles the test.
def _map_eligibility(status):
    mapping = {
        "included": "Met",
        "not included": "Not Met",
        "excluded": "Violated",
        "not excluded": "Not Violated",
        "not enough information": "Unclear",
        "not applicable": "Irrelevant",
    }

    if status not in mapping:
        raise Exception("unknown eligibility status")

    return mapping[status]


def main():
    if not INPUT_TRIALGPT_MATCHING_RESULTS_PATH.exists():
        print(
            f"Error: Parquet file not found at {INPUT_TRIALGPT_MATCHING_RESULTS_PATH}"
        )
        return

    # --- Load Parquet and organize by trial ---
    print(f"Loading parquet dataset from: {INPUT_TRIALGPT_MATCHING_RESULTS_PATH}")
    table = pq.read_table(INPUT_TRIALGPT_MATCHING_RESULTS_PATH)
    parquet_data = table.to_pylist()

    unique_trial_ids = set()
    unique_patient_ids = set()

    # trial_criteria[trial_id][type] = list of {"text": str, "id": int}
    trial_criteria_with_ground_truths = {}

    print("Generating ground truths from parquet...")

    ground_truths = []

    for row in parquet_data:
        annotation_id = row.get("annotation_id")
        t_id = row.get("trial_id")
        p_id = row.get("patient_id")
        text = row.get("criterion_text") or "none"

        # we create our own ground_truth.json with this with the filtering logic above
        filtered_row = {
            k: v
            for k, v in row.items()
            if k
            in {
                "annotation_id",
                "patient_id",
                "trial_id",
                "criterion_type",
            }
        }
        filtered_row["criterion_text"] = text  # handles the "none" properly
        filtered_row["trialgpt_prediction"] = _map_eligibility(row["gpt4_eligibility"])
        filtered_row["expert_eligibility"] = _map_eligibility(row["expert_eligibility"])
        ground_truths.append(filtered_row)

        # we use this to pull the full patient info we need later
        if p_id:
            unique_patient_ids.add(p_id)

        # We use trial_criteria as criteria instead of the ones found in corpus because there are slight
        # differences that are just a result of parsing. So aggregating here so that we can use it when we parse in other metadata from the corpus.
        if t_id not in trial_criteria_with_ground_truths:
            trial_criteria_with_ground_truths[t_id] = {"inclusion": [], "exclusion": []}
        # We delineate criteria only by new lines, so removing them here and replacing them with spaces
        clean_text = " ".join(text.strip().split("\n"))
        c_type = filtered_row.get("criterion_type")  # 'inclusion' or 'exclusion'
        existing = [
            c
            for c in trial_criteria_with_ground_truths[t_id][c_type]
            # There are cases where it's capitalized for some reason in some and not in others
            if c["text"].lower() == clean_text.lower()
        ]
        # We use just the first one we get
        if not existing:
            trial_criteria_with_ground_truths[t_id][c_type].append(
                {"text": clean_text, "id": annotation_id}
            )

    unique_trial_ids = set(trial_criteria_with_ground_truths.keys())
    print(
        f"Found {len(unique_trial_ids)} unique trials and {len(unique_patient_ids)} unique patients in parquet."
    )

    # Process Trials (Corpus)
    if not INPUT_TRIALGPT_PARSED_TRIAL_CORPUS_PATH.exists():
        print(
            f"ERROR: Corpus file not found at {INPUT_TRIALGPT_PARSED_TRIAL_CORPUS_PATH}"
        )
        sys.exit()

    print(f"Processing trials from {INPUT_TRIALGPT_PARSED_TRIAL_CORPUS_PATH}...")
    matched_trials = {}
    # nested in metadata field
    requested_fields = [
        "brief_title",
        "diseases_list",
        "drugs_list",
        "brief_summary",
    ]  # exclusion_criteria & inclusion_criteria handled separately

    with open(INPUT_TRIALGPT_PARSED_TRIAL_CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            trial_raw = json.loads(line)
            trial_id = trial_raw.get("_id")
            if trial_id in unique_trial_ids:
                metadata = trial_raw.get("metadata", {})

                if trial_id in matched_trials:
                    print("ERROR: duplicate trial in corpus")
                    sys.exit()

                matched_trials[trial_id] = {
                    key: metadata.get(key, "") for key in requested_fields
                }

                # use ground truth criteria texts directly
                matched_trials[trial_id]["exclusion_criteria"] = [
                    c["text"]
                    for c in trial_criteria_with_ground_truths[trial_id]["exclusion"]
                ]
                matched_trials[trial_id]["inclusion_criteria"] = [
                    c["text"]
                    for c in trial_criteria_with_ground_truths[trial_id]["inclusion"]
                ]

    PROCESSED_TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_TRIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(matched_trials, f, indent=2)
    print(f"Saved {len(matched_trials)} trials to {PROCESSED_TRIALS_PATH}")

    # finally, save gathered ground truth data
    PROCESSED_GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_GT_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truths, f, indent=2)
    print(f"Saved {len(ground_truths)} ground truth records to {PROCESSED_GT_PATH}")

    # Process Patients (Queries)
    if INPUT_TRIALGPT_PARSED_PATIENT_CORPUS_PATH.exists():
        print(
            f"Processing patients from {INPUT_TRIALGPT_PARSED_PATIENT_CORPUS_PATH}..."
        )
        matched_patients = {}
        with open(
            INPUT_TRIALGPT_PARSED_PATIENT_CORPUS_PATH, "r", encoding="utf-8"
        ) as f:
            for line in f:
                patient_raw = json.loads(line)
                patient_id = patient_raw.get("_id")
                if patient_id in unique_patient_ids:
                    matched_patients[patient_id] = patient_raw
        PROCESSED_PATIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_PATIENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(matched_patients, f, indent=2)
        print(f"Saved {len(matched_patients)} patients to {PROCESSED_PATIENTS_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
