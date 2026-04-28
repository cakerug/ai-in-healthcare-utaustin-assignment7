import sys
import json
from pathlib import Path

from collections import Counter, defaultdict

from tqdm import tqdm
from tabulate import tabulate

from langchain_core.callbacks import UsageMetadataCallbackHandler

from lib.data_loader import (
    GroundTruthsData,
    PatientsData,
    TrialsData,
    INCLUSION_VALUES,
    EXCLUSION_VALUES,
)

RESULTS_DIR = Path("results")
LOG_DIR = Path("log")


def evaluate_against_ground_truth(
    gt_data: GroundTruthsData,
    tr_data: TrialsData,
    patient_id: str,
    trial_id: str,
    results: dict,
) -> tuple[list[dict], dict[str, list]]:
    comparison_rows = []

    evals = [
        (results.get("inclusion_criteria_evaluation", []), "inclusion"),
        (results.get("exclusion_criteria_evaluation", []), "exclusion"),
    ]

    matches = {"model_only": [], "matched": [], "trial_only": {}}
    trial_set = {
        "exclusion": set(
            x.lower() for x in tr_data.get(trial_id)["exclusion_criteria"]
        ),
        "inclusion": set(
            x.lower() for x in tr_data.get(trial_id)["inclusion_criteria"]
        ),
    }

    # print("trial_set", trial_set)
    for criteria_list, c_type in evals:
        for item in criteria_list:
            text = item["criterion"]
            actual = gt_data.get(patient_id, trial_id, c_type, text)
            if actual is None:
                # if text in matches["model_only"]:
                #     print(text, "appeared more than once")
                matches["model_only"].append(text)
                continue

            predicted = item.get("classification")
            expert = actual["expert_eligibility"]
            # trialgpt = actual["trialgpt_prediction"]

            # if text in matches["matched"]:
            #     print(text, "appeared more than once")

            matches["matched"].append(text)
            trial_set[c_type].discard(text.lower())

            comparison_rows.append(
                {
                    "criterion": text,
                    "criterion_type": c_type,
                    "predicted": predicted,
                    "expert_eligibility": expert,
                    # "trialgpt_prediction": trialgpt,
                    "model_match": predicted == expert,
                    # "trialgpt_match": trialgpt == expert,
                }
            )

    matches["trial_only"] = {k: list(v) for k, v in trial_set.items()}

    return comparison_rows, matches


def _classification_metrics(y_true: list, y_pred: list, labels: set) -> dict:
    sorted_labels = sorted(labels)
    label_idx = {l: i for i, l in enumerate(sorted_labels)}
    n = len(sorted_labels)

    cm: list[list[int]] = [[0] * n for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        if true in label_idx and pred in label_idx:
            cm[label_idx[true]][label_idx[pred]] += 1

    per_class = {}
    for i, label in enumerate(sorted_labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        tn = sum(cm[r][c] for r in range(n) for c in range(n)) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "specificity": round(specificity, 4),
        }

    grand_total = sum(cm[r][c] for r in range(n) for c in range(n))
    accuracy = sum(cm[i][i] for i in range(n)) / grand_total if grand_total else 0.0

    macro_precision = sum(per_class[l]["precision"] for l in sorted_labels) / n
    macro_recall = sum(per_class[l]["recall"] for l in sorted_labels) / n
    macro_f1 = sum(per_class[l]["f1"] for l in sorted_labels) / n
    macro_specificity = sum(per_class[l]["specificity"] for l in sorted_labels) / n

    confusion_matrix = {
        sorted_labels[r]: {sorted_labels[c]: cm[r][c] for c in range(n)}
        for r in range(n)
    }

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_specificity": round(macro_specificity, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "n_samples": grand_total,
    }


def _metrics_for_split(subset: list[dict], labels: set) -> dict:
    expert = [r["expert_eligibility"] for r in subset]
    predicted = [r["predicted"] for r in subset]
    return _classification_metrics(expert, predicted, labels)


def compute_metrics(rows: list[dict]) -> dict:
    inclusion_rows = [r for r in rows if r["criterion_type"] == "inclusion"]
    exclusion_rows = [r for r in rows if r["criterion_type"] == "exclusion"]

    return {
        "overall": _metrics_for_split(rows, INCLUSION_VALUES | EXCLUSION_VALUES),
        "inclusion": _metrics_for_split(inclusion_rows, INCLUSION_VALUES),
        "exclusion": _metrics_for_split(exclusion_rows, EXCLUSION_VALUES),
    }


def load_results(experiment_name: str) -> list[dict]:
    path = RESULTS_DIR / f"{experiment_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results file found at {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_metrics(metrics: dict, experiment_name: str = "") -> None:
    header = (
        f"=== Metrics: {experiment_name} ===" if experiment_name else "=== Metrics ==="
    )
    print(header)
    for split in ("overall", "inclusion", "exclusion"):
        print(f"\n  -- {split} --")
        m = metrics[split]
        print(f"\nn={m['n_samples']}")
        print(f"    accuracy:          {m['accuracy']:.4f}")
        print(f"    macro precision:   {m['macro_precision']:.4f}")
        print(f"    macro recall:      {m['macro_recall']:.4f}")
        print(f"    macro F1:          {m['macro_f1']:.4f}")
        print(f"    macro specificity: {m['macro_specificity']:.4f}")
        print("    per-class:")
        for label, stats in m["per_class"].items():
            print(
                f"      {label:<14}  "
                f"P={stats['precision']:.3f}  R={stats['recall']:.3f}  "
                f"F1={stats['f1']:.3f}  spec={stats['specificity']:.3f}  "
                f"(tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} tn={stats['tn']})"
            )
        print("    confusion matrix (rows=true, cols=pred):")
        cm = m["confusion_matrix"]
        col_labels = sorted(cm.keys())
        print("      " + "  ".join(f"{c:<14}" for c in col_labels))
        for row_label in col_labels:
            row_str = "  ".join(f"{cm[row_label][c]:<14}" for c in col_labels)
            print(f"      {row_str}  <- {row_label}")
    print()


def print_trialgpt_metrics(gt_data: GroundTruthsData) -> None:
    rows = []
    inclusion_rows = []
    exclusion_rows = []

    for key, entry in gt_data.items():
        _, _, criterion_type, _ = key
        row = {
            "criterion_type": criterion_type,
            "predicted": entry["trialgpt_prediction"],
            "expert_eligibility": entry["expert_eligibility"],
            "model_match": entry["trialgpt_prediction"] == entry["expert_eligibility"],
            # "trialgpt_prediction": entry["trialgpt_prediction"],
        }
        rows.append(row)

        if criterion_type == "inclusion":
            inclusion_rows.append(row)
        elif criterion_type == "exclusion":
            exclusion_rows.append(row)

    # Print metrics for all criteria
    print_metrics(compute_metrics(rows), experiment_name="TrialGPT results")

    # Print metrics for inclusion criteria only
    if inclusion_rows:
        print_metrics(
            compute_metrics(inclusion_rows),
            experiment_name="TrialGPT results (inclusion)",
        )

    # Print metrics for exclusion criteria only
    if exclusion_rows:
        print_metrics(
            compute_metrics(exclusion_rows),
            experiment_name="TrialGPT results (exclusion)",
        )


def apply_self_consistency(
    name: str, cot_runs: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Apply self-consistency voting across multiple CoT runs.

    Args:
        all_metrics: dict mapping experiment_name -> list of row dicts.
                     Only processes keys that look like cot__*__temp05__<i>.

    Returns:
        (consolidated_rows, tie_log, dropped_log)
        - consolidated_rows: one row per unique (patient_id, trial_id, criterion)
          with added fields: pool_size, vote_count, vote_breakdown, sc_predicted, sc_model_match
        - tie_log: rows where a 1-of-2 tie occurred, with original per-run results
        - dropped_log: rows dropped because only 1 run extracted them
    """
    # Group rows by their unique key
    grouped = defaultdict(list)
    for run_name, rows in cot_runs.items():
        for row in rows:
            key = (
                row["patient_id"],
                row["trial_id"],
                row["criterion"],
                row["criterion_type"],
            )
            grouped[key].append((run_name, row))

    # Vote
    consolidated_rows = []
    tie_log = []
    dropped_log = []

    for key, run_rows in grouped.items():
        patient_id, trial_id, criterion, criterion_type = key
        pool_size = len(run_rows)

        # Drop singletons — no consistency signal
        if pool_size < 2:
            dropped_log.append(
                {
                    "patient_id": patient_id,
                    "trial_id": trial_id,
                    "criterion": criterion,
                    "criterion_type": criterion_type,
                    "run": run_rows[0][0],
                    "predicted": run_rows[0][1]["predicted"],
                    "reason": "only_extracted_by_one_run",
                }
            )
            continue

        predictions = [r["predicted"] for _, r in run_rows]
        vote_breakdown = dict(Counter(predictions))
        top_prediction, top_count = Counter(predictions).most_common(1)[0]

        # Tie: pool_size == 2 and each run said something different
        is_tie = pool_size == 2 and len(set(predictions)) == 2

        if is_tie:
            sc_predicted = "Unclear"
            tie_log.append(
                {
                    "patient_id": patient_id,
                    "trial_id": trial_id,
                    "criterion": criterion,
                    "criterion_type": criterion_type,
                    "pool_size": pool_size,
                    "per_run_results": [
                        {"run": run_name, "predicted": r["predicted"]}
                        for run_name, r in run_rows
                    ],
                }
            )
        else:
            sc_predicted = top_prediction

        base_row = run_rows[0][1].copy()
        base_row.update(
            {
                "predicted": sc_predicted,
                "sc_predicted": sc_predicted,
                "sc_model_match": (
                    sc_predicted == base_row["expert_eligibility"]
                    if sc_predicted != "Unclear"
                    else None
                ),
                "model_match": (
                    sc_predicted == base_row["expert_eligibility"]
                    if sc_predicted != "Unclear"
                    else None
                ),
                "pool_size": pool_size,
                "vote_count": top_count,
                "vote_breakdown": vote_breakdown,
                "is_tie": is_tie,
            }
        )
        consolidated_rows.append(base_row)

    # Save to files
    base_name = name

    results_dir = Path("results")
    log_dir = Path("log")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{base_name}.jsonl"
    mismatches_path = log_dir / f"{base_name}_mismatches.jsonl"

    with open(results_path, "w") as f:
        for row in consolidated_rows:
            f.write(json.dumps(row) + "\n")

    with open(mismatches_path, "w") as f:
        for entry in tie_log + dropped_log:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(consolidated_rows)} rows -> {results_path}")
    print(
        f"Saved {len(tie_log)} tie(s) and {len(dropped_log)} dropped row(s) -> {mismatches_path}"
    )

    return consolidated_rows, tie_log, dropped_log


def summarize_ties(tie_log: list[dict], dropped_log: list[dict]) -> None:
    """Print a readable summary of all ties and dropped singletons."""
    if not tie_log and not dropped_log:
        print("No ties or dropped rows found.")
        return

    if tie_log:
        print(f"=== Tie Summary: {len(tie_log)} tie(s) found ===\n")
        for t in tie_log:
            print(
                f"  Patient: {t['patient_id']} | Trial: {t['trial_id']}\n"
                f"  Criterion ({t['criterion_type']}): {t['criterion']}\n"
                f"  Per-run results:"
            )
            for r in t["per_run_results"]:
                print(f"    - {r['run']}: {r['predicted']}")
            print()

    if dropped_log:
        print(f"=== Dropped Singletons: {len(dropped_log)} row(s) dropped ===\n")
        for d in dropped_log:
            print(
                f"  Patient: {d['patient_id']} | Trial: {d['trial_id']}\n"
                f"  Criterion ({d['criterion_type']}): {d['criterion']}\n"
                f"  Extracted by: {d['run']} with predicted={d['predicted']}\n"
            )


def run_experiment(
    chain,
    experiment_name: str,
    gt_data: GroundTruthsData,
    pt_data: PatientsData,
    tr_data: TrialsData,
    inc_exc: str = "both",  # both | inclusion | exclusion
) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{experiment_name}.jsonl"

    # LOG_DIR.mkdir(exist_ok=True)
    # mismatch_path = LOG_DIR / f"{experiment_name}_mismatches.jsonl"

    completed_pairs: set[tuple[str, str]] = set()
    all_rows: list[dict] = []

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                all_rows.append(row)
                completed_pairs.add((row["patient_id"], row["trial_id"]))

        tqdm.write(
            f"[resume] Found {len(completed_pairs)} completed pairs in {output_path}"
        )

    # completed_pairs_mismatches: set[tuple[str, str]] = set()
    # if mismatch_path.exists():
    #     with open(mismatch_path) as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             row = json.loads(line)
    #             completed_pairs_mismatches.add((row["patient_id"], row["trial_id"]))

    #     tqdm.write(
    #         f"[resume] Found {len(completed_pairs_mismatches)} missed pairs in {mismatch_path}"
    #     )

    # diff = completed_pairs_mismatches - completed_pairs
    # if len(diff) != 0 or len(completed_pairs) != len(completed_pairs):
    #     print(
    #         "WARNING: the resume is malformed because the two files are not synced. Safest thing to do is to delete the jsonl files in question in the log and results folders and re-run. Stuff should be cached by langchain anyway so shouldn't be so slow. This would result in incorrect details in the mismatches file. This is somewhat irrelevant for the multi-agent flows"
    #     )
    # sys.exit()

    pairs = list(gt_data.get_patient_trial_pairs())
    total = len(pairs)

    with open(output_path, "a") as f_out:
        # with open(output_path, "a") as f_out, open(mismatch_path, "a") as f_mismatches:
        with tqdm(
            total=total,
            desc=experiment_name,
            unit="pair",
            # dynamic_ncols=True,
            # bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {bar} {postfix}",
        ) as pbar:
            for idx, (patient_id, trial_id) in enumerate(pairs):
                if (patient_id, trial_id) in completed_pairs:
                    pbar.update(1)
                    continue

                pbar.set_postfix(patient=patient_id, trial=trial_id)

                try:
                    # callback = UsageMetadataCallbackHandler()
                    response = chain.invoke(
                        {
                            "patient_note": pt_data.get(patient_id),
                            "clinical_trial": tr_data.get_formatted_trial(
                                trial_id, inc_exc=inc_exc
                            ),
                        },
                        # config={"callbacks": [callback]},
                    )
                    # print(callback.usage_metadata)
                    # sys.exit()
                    structured = response.get("structured_data", {})
                    # print(">>>>>>>>>>>>>>>>> structured_data", structured)
                    # sys.exit()
                except Exception as e:
                    tqdm.write(f"  !! chain error ({patient_id} / {trial_id}): {e}")
                    continue

                rows, matches = evaluate_against_ground_truth(
                    gt_data, tr_data, patient_id, trial_id, structured
                )

                for row in rows:
                    row["patient_id"] = patient_id
                    row["trial_id"] = trial_id
                    f_out.write(json.dumps(row) + "\n")
                    all_rows.append(row)

                f_out.flush()

                # n_inclusion = len(tr_data.get(trial_id)["inclusion_criteria"])
                # n_exclusion = len(tr_data.get(trial_id)["exclusion_criteria"])
                # n_expected = n_inclusion + n_exclusion
                # n_returned = len(
                #     structured.get("inclusion_criteria_evaluation", [])
                # ) + len(structured.get("exclusion_criteria_evaluation", []))
                # if n_returned != n_expected:
                #     mismatch_entry = {
                #         "patient_id": patient_id,
                #         "trial_id": trial_id,
                #         "n_expected": n_expected,
                #         "n_returned": n_returned,
                #         "matches": matches,  # already has model_only / matched / trial_only
                #     }
                #     f_mismatches.write(json.dumps(mismatch_entry) + "\n")
                #     f_mismatches.flush()
                #     tqdm.write(
                #         f"  [mismatch] {patient_id}/{trial_id}: expected {n_expected}, got {n_returned}"
                #     )

                pbar.update(1)

    tqdm.write(f"[done] {len(all_rows)} criterion rows written to {output_path}")
    return all_rows


def print_overview_metrics(
    all_metrics: dict[str, list[dict]], gt_data: list[dict]
) -> None:
    """Print a side-by-side comparison of all experiments plus the TrialGPT baseline."""

    baseline_rows = [
        {
            "criterion_type": key[2],  # TODO: not great
            "expert_eligibility": row["expert_eligibility"],
            "predicted": row["trialgpt_prediction"],
        }
        for key, row in gt_data.items()
    ]
    baseline_metrics = compute_metrics(baseline_rows)

    all_entries: dict[str, dict] = {"TrialGPT (baseline)": baseline_metrics}
    for name, rows in all_metrics.items():
        all_entries[name] = compute_metrics(rows)

    # (split, metric_key, display_label)
    metric_keys = [
        ("overall", "accuracy", "Accuracy"),
        ("overall", "macro_f1", "Macro F1"),
        ("overall", "macro_precision", "Macro Precision"),
        ("overall", "macro_recall", "Macro Recall"),
        ("overall", "macro_specificity", "Macro Specificity"),
        ("inclusion", "accuracy", "Incl. Accuracy"),
        ("inclusion", "macro_f1", "Incl. Macro F1"),
        ("exclusion", "accuracy", "Excl. Accuracy"),
        ("exclusion", "macro_f1", "Excl. Macro F1"),
    ]

    col_names = list(all_entries.keys())
    counts = {"TrialGPT (baseline)": len(baseline_rows)}
    counts.update({name: len(rows) for name, rows in all_metrics.items()})

    cols_per_row = 4
    num_chunks = (len(col_names) + cols_per_row - 1) // cols_per_row

    print("\n" + "=" * 100)
    print("OVERVIEW — ALL EXPERIMENTS vs TrialGPT BASELINE")
    print("=" * 100)

    for chunk_idx in range(num_chunks):
        start_col = chunk_idx * cols_per_row
        end_col = min(start_col + cols_per_row, len(col_names))
        chunk_cols = col_names[start_col:end_col]

        rows = []
        for split, key, label in metric_keys:
            row = [label] + [
                f"{all_entries[c][split].get(key, float('nan')):.4f}"
                for c in chunk_cols
            ]
            rows.append(row)

        # Add counts row
        count_row = ["N (predictions)"] + [f"{counts[c]}" for c in chunk_cols]
        rows.append(count_row)

        headers = ["Metric"] + chunk_cols
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        if chunk_idx < num_chunks - 1:
            print()

    print()
