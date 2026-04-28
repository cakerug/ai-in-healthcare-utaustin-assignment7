import json
from pathlib import Path
from typing import Literal, get_args

PROCESSED_GT_PATH = Path("dataset/preprocessed/sigir/ground_truths.json")
PROCESSED_TRIALS_PATH = Path("dataset/preprocessed/sigir/trials.json")
PROCESSED_PATIENTS_PATH = Path("dataset/preprocessed/sigir/patients.json")

InclusionEligibility = Literal["Met", "Not Met", "Unclear", "Irrelevant"]
ExclusionEligibility = Literal["Violated", "Not Violated", "Unclear", "Irrelevant"]

INCLUSION_VALUES = set(get_args(InclusionEligibility))
EXCLUSION_VALUES = set(get_args(ExclusionEligibility))

EligibilityValue = InclusionEligibility | ExclusionEligibility
CriterionType = Literal["inclusion", "exclusion"]
LookupKey = tuple[
    str, str, str, str
]  # (patient_id, trial_id, criterion_type, criterion_text)


class GroundTruthsData:
    def __init__(self, path: str | Path = PROCESSED_GT_PATH):
        with open(path) as f:
            raw = json.load(f)

        self._lookup: dict[LookupKey, dict[str, EligibilityValue]] = {}

        for entry in raw:
            key: LookupKey = (
                entry["patient_id"],
                entry["trial_id"],
                entry["criterion_type"],
                entry["criterion_text"].lower(),
            )
            self._lookup[key] = {
                "expert_eligibility": entry["expert_eligibility"],
                "trialgpt_prediction": entry["trialgpt_prediction"],
            }

    def get(self, patient_id, trial_id, criterion_type, criterion_text):
        entry = self._lookup.get(
            (patient_id, trial_id, criterion_type, criterion_text.lower())
        )
        if entry is None:
            return None

        expert = entry["expert_eligibility"]
        trialgpt = entry["trialgpt_prediction"]
        valid = INCLUSION_VALUES if criterion_type == "inclusion" else EXCLUSION_VALUES

        # This should've been handled in data processing if we do hit this assertion error
        assert expert in valid, (
            f"Invalid expert_eligibility {expert!r} for {criterion_type!r}"
        )
        assert trialgpt in valid, (
            f"Invalid trialgpt_prediction {trialgpt!r} for {criterion_type!r}"
        )

        return entry

    # def expert_eligibility(
    #     self,
    #     patient_id: str,
    #     trial_id: str,
    #     criterion_type: CriterionType,
    #     criterion_text: str,
    # ) -> EligibilityValue | None:
    #     entry = self.get(patient_id, trial_id, criterion_type, criterion_text)
    #     return entry["expert_eligibility"] if entry else None

    # def trialgpt_prediction(
    #     self,
    #     patient_id: str,
    #     trial_id: str,
    #     criterion_type: CriterionType,
    #     criterion_text: str,
    # ) -> EligibilityValue | None:
    #     entry = self.get(patient_id, trial_id, criterion_type, criterion_text)
    #     return entry["trialgpt_prediction"] if entry else None

    def get_patient_trial_pairs(self) -> list[tuple[str, str]]:
        return sorted(set((p, t) for p, t, _, _ in self._lookup))

    def __len__(self) -> int:
        return len(self._lookup)

    def __contains__(self, key: LookupKey) -> bool:
        return key in self._lookup

    def items(self):
        return self._lookup.items()

    def values(self):
        return self._lookup.values()


class PatientsData:
    def __init__(self, path: str | Path = PROCESSED_PATIENTS_PATH):
        with open(path) as f:
            raw = json.load(f)

        self._lookup: dict[str, str] = {
            patient_id: data["text"] for patient_id, data in raw.items()
        }

    def get(self, patient_id: str) -> str | None:
        return self._lookup.get(patient_id)

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self):
        return len(self._lookup)


class TrialsData:
    def __init__(self, path: str | Path = PROCESSED_TRIALS_PATH):
        with open(path) as f:
            self._lookup = json.load(f)

    def get(self, trial_id: str) -> dict | None:
        return self._lookup.get(trial_id)

    def _format_criteria_list(self, lst) -> str:
        return "\n".join(lst)

    def get_formatted_trial(
        self, trial_id: str, inc_exc: Literal["inclusion", "exclusion", "both"] = "both"
    ) -> str:

        trial_info = self._lookup[trial_id]

        trial = f"Title: {trial_info['brief_title']}\n"
        trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
        trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
        trial += f"Summary: {trial_info['brief_summary']}\n\n"

        if inc_exc in ["inclusion", "both"]:
            trial += "Inclusion Criteria:\n%s\n" % self._format_criteria_list(
                trial_info.get("inclusion_criteria")
            )

        if inc_exc == "both":
            trial += "\n"

        if inc_exc in ["exclusion", "both"]:
            trial += "Exclusion Criteria:\n%s\n" % self._format_criteria_list(
                trial_info.get("exclusion_criteria")
            )

        return trial

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self):
        return len(self._lookup)
