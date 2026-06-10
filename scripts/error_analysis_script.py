#!/usr/bin/env python3
"""Generate a standalone Vakra evaluation error-analysis.

The script aligns benchmark inputs, ground-truth outputs, model predictions,
and evaluator results by domain and UUID. For each sample, it extracts final
turn scores and answers, compares predicted tool calls with ground-truth tool
calls, and records missing-data or evaluator-consistency errors.

It writes two JSON files:
  * <output-file>: flattened analysis records suitable for inspection.
  * WITH_<output-file>: the same records plus the original source payloads.

The script uses only Python's standard library and can be run directly from any
working directory.

Example:
    ./error_analysis_script.py \
        --input-dir /path/to/capability/input \
        --gt-output-dir /path/to/capability/output \
        --prediction-dir /path/to/predictions \
        --results-file /path/to/results.json \
        --output-file /path/to/error_analysis.json \
        --pretty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Vakra error-analysis dataset by aligning benchmark inputs, "
            "ground-truth outputs, predictions, and evaluator results by UUID."
        ),
        epilog=(
            "Writes OUTPUT_FILE and a WITH_OUTPUT_FILE companion containing the "
            "original source payloads. No Vakra installation or third-party "
            "Python packages are required."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input <DOMAIN>.json files.",
    )
    parser.add_argument(
        "--gt-output-dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth output <DOMAIN>.json files.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing predicted output <DOMAIN>.json files.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Results JSON file with entries under domains.<DOMAIN>.dialogues[*].uuid.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Where to write the aligned JSON output.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional subset of domain names to align. Defaults to all domains in --input-dir.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_domains(input_dir: Path) -> list[str]:
    return sorted(path.stem for path in input_dir.glob("*.json"))


def validate_path(path: Path, path_label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path_label} does not exist: {path}")


def index_records(records: Any, source_name: str, domain: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{source_name} for domain '{domain}' must be a list.")

    indexed: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{source_name} for domain '{domain}' has a non-object record at index {idx}."
            )
        uuid = record.get("uuid")
        if not uuid:
            raise ValueError(
                f"{source_name} for domain '{domain}' is missing 'uuid' at index {idx}."
            )
        if uuid in indexed:
            raise ValueError(
                f"{source_name} for domain '{domain}' has duplicate uuid '{uuid}'."
            )
        indexed[uuid] = record
    return indexed


def extract_question_type(
    input_record: dict[str, Any],
    input_dir: Path,
    gt_tool_call: Any | None = None,
) -> str:
    if "capability_1_bi_apis" in str(input_dir):
        gt_calls = gt_tool_call if isinstance(gt_tool_call, list) else []
        slot_tool_names = {"filter_data", "retrieve_data", "aggregate_data"}
        if any(
            isinstance(tool_call, dict) and tool_call.get("name") in slot_tool_names
            for tool_call in gt_calls
        ):
            return "(SLOT)"
        return "(SEL)"

    raw_type = input_record.get("type")
    if not raw_type:
        return "(API)"

    type_text = str(raw_type)
    last_open = type_text.rfind("(")
    last_close = type_text.rfind(")")
    if last_open != -1 and last_close > last_open:
        return type_text[last_open : last_close + 1]
    return type_text


def get_final_output_turn(record: dict[str, Any], source_name: str, domain: str, uuid: str) -> dict[str, Any]:
    output = record.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError(
            f"{source_name} for domain '{domain}' and uuid '{uuid}' must contain a non-empty 'output' list."
        )

    final_turn = output[-1]
    if not isinstance(final_turn, dict):
        raise ValueError(
            f"{source_name} for domain '{domain}' and uuid '{uuid}' has a non-object final output turn."
        )
    return final_turn


def get_nested_value(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def coerce_value_for_comparison(gt_value: Any, pred_value: Any) -> Any:
    if isinstance(gt_value, str):
        pred_text = str(pred_value)
        try:
            gt_int = int(gt_value)
        except (TypeError, ValueError):
            gt_int = None
        else:
            try:
                pred_number = float(pred_text)
            except (TypeError, ValueError):
                pred_number = None
            if pred_number is not None and pred_number.is_integer() and int(pred_number) == gt_int:
                return str(gt_int)

        try:
            gt_float = float(gt_value)
        except (TypeError, ValueError):
            gt_float = None
        else:
            try:
                pred_float = float(pred_text)
            except (TypeError, ValueError):
                pred_float = None
            if pred_float is not None and pred_float == gt_float:
                return gt_value

        return pred_text

    if isinstance(gt_value, int) and not isinstance(gt_value, bool):
        try:
            return int(float(pred_value))
        except (TypeError, ValueError):
            return pred_value

    if isinstance(gt_value, float):
        try:
            return float(pred_value)
        except (TypeError, ValueError):
            return pred_value

    return pred_value


def compare_single_tool_call(gt_call: Any, pred_calls: list[Any]) -> dict[str, float]:
    default_scores = {
        "score": 0.0,
        "tool_name": 0.0,
        "arg_name": 0.0,
        "arg_value": 0.0,
    }

    if not isinstance(gt_call, dict):
        return default_scores

    gt_name = gt_call.get("name")
    gt_arguments = gt_call.get("arguments")

    if (gt_name == "initialize_active_data") or (gt_name == "get_data"):
        return {
            "score": 1.0,
            "tool_name": 1.0,
            "arg_name": 1.0,
            "arg_value": 1.0,
        }

    same_name_calls = [
        pred_call
        for pred_call in pred_calls
        if isinstance(pred_call, dict) and pred_call.get("name") == gt_name
    ]
    if not same_name_calls:
        return default_scores

    tool_name_score = 1.0

    if not isinstance(gt_arguments, dict):
        arg_name_score = 1.0 if any(pred_call.get("arguments") == gt_arguments for pred_call in same_name_calls) else 0.0
        arg_value_score = arg_name_score
        final_score = arg_value_score
        return {
            "score": final_score,
            "tool_name": tool_name_score,
            "arg_name": arg_name_score,
            "arg_value": arg_value_score,
        }

    same_arg_name_calls = []
    for pred_call in same_name_calls:
        pred_arguments = pred_call.get("arguments")
        if isinstance(pred_arguments, dict):
            gt_arg_keys = {key for key in gt_arguments.keys() if key not in {"data", "data_label"}}
            pred_arg_keys = {key for key in pred_arguments.keys() if key not in {"data", "data_label"}}
            if gt_arg_keys == pred_arg_keys:
                same_arg_name_calls.append(pred_call)

    arg_name_score = 1.0 if same_arg_name_calls else 0.0
    if not same_arg_name_calls:
        return {
            "score": 0.0,
            "tool_name": tool_name_score,
            "arg_name": arg_name_score,
            "arg_value": 0.0,
        }

    comparable_keys = [key for key in gt_arguments if (key != "data") and (key != "data_label")]
    arg_value_score = 1.0 if any(
        all(
            coerce_value_for_comparison(gt_arguments[key], pred_call["arguments"][key])
            == gt_arguments[key]
            for key in comparable_keys
        )
        for pred_call in same_arg_name_calls
    ) else 0.0

    return {
        "score": arg_value_score,
        "tool_name": tool_name_score,
        "arg_name": arg_name_score,
        "arg_value": arg_value_score,
    }


def compare_tool_calls(gt_tool_call: Any, pred_tool_call: Any) -> list[dict[str, float]]:
    gt_calls = gt_tool_call if isinstance(gt_tool_call, list) else []
    pred_calls = pred_tool_call if isinstance(pred_tool_call, list) else []

    return [compare_single_tool_call(gt_call, pred_calls) for gt_call in gt_calls]


def build_with_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"WITH_{output_path.name}")


def align_domain(
    domain: str,
    input_dir: Path,
    gt_output_dir: Path,
    prediction_dir: Path,
    results_by_domain: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = input_dir / f"{domain}.json"
    gt_output_path = gt_output_dir / f"{domain}.json"
    prediction_path = prediction_dir / f"{domain}.json"

    validate_path(input_path, "Input file")
    input_records = load_json(input_path)
    domain_path_errors: list[str] = []

    if gt_output_path.exists():
        gt_output_records = load_json(gt_output_path)
    else:
        gt_output_records = []
        domain_path_errors.append(f"Ground-truth output file does not exist: {domain}")

    if prediction_path.exists():
        prediction_records = load_json(prediction_path)
    else:
        prediction_records = []
        domain_path_errors.append(f"Prediction file does not exist: {domain}")
        print(f"Prediction file does not exist for domain: {domain}")

    result_domain_payload = results_by_domain.get(domain)
    domain_result_error: str | None = None
    if result_domain_payload is None:
        result_records: dict[str, dict[str, Any]] = {}
        domain_result_error = f"Results file is missing domain '{domain}'."
    else:
        result_dialogues = result_domain_payload.get("dialogues")
        result_records = index_records(result_dialogues, "Results dialogues", domain)

    input_by_uuid = index_records(input_records, "Input records", domain)
    gt_output_by_uuid = index_records(gt_output_records, "Ground-truth output records", domain)
    prediction_by_uuid = index_records(prediction_records, "Prediction records", domain)

    aligned_samples: list[dict[str, Any]] = []
    missing_in_gt_output: list[str] = []
    missing_in_predictions: list[str] = []
    missing_in_results: list[str] = []

    for uuid, input_record in input_by_uuid.items():
        gt_output_record = gt_output_by_uuid.get(uuid)
        prediction_record = prediction_by_uuid.get(uuid)
        result_record = result_records.get(uuid)

        if gt_output_record is None:
            missing_in_gt_output.append(uuid)
        if prediction_record is None:
            missing_in_predictions.append(uuid)
        if result_record is None:
            missing_in_results.append(uuid)

        aligned_samples.append(
            {
                "uuid": uuid,
                "domain": domain,
                "input": input_record,
                "ground_truth_output": gt_output_record,
                "prediction": prediction_record,
                "result": result_record,
            }
        )

    extra_gt_output = sorted(set(gt_output_by_uuid) - set(input_by_uuid))
    extra_predictions = sorted(set(prediction_by_uuid) - set(input_by_uuid))
    extra_results = sorted(set(result_records) - set(input_by_uuid))

    flattened_samples: list[dict[str, Any]] = []
    for sample in aligned_samples:
        input_record = sample["input"]
        gt_output_record = sample["ground_truth_output"]
        prediction_record = sample["prediction"]
        uuid = input_record["uuid"]

        if gt_output_record is None:
            raise ValueError(
                f"Cannot derive final-turn fields for domain '{domain}' and uuid '{uuid}' because "
                "ground-truth output is missing."
            )

        gt_final_turn = get_final_output_turn(
            gt_output_record,
            "Ground-truth output record",
            domain,
            uuid,
        )
        gt_query = gt_final_turn.get("query")
        missing_prediction_error: str | None = None
        if prediction_record is None:
            pred_final_turn: dict[str, Any] = {}
            missing_prediction_error = (
                f"Cannot derive final-turn fields for domain '{domain}' and uuid '{uuid}' because "
                "prediction is missing."
            )
        else:
            pred_final_turn = get_final_output_turn(
                prediction_record,
                "Prediction record",
                domain,
                uuid,
            )

            gt_turn_id = gt_final_turn.get("turn_id")
            pred_turn_id = pred_final_turn.get("turn_id")
            if gt_turn_id != pred_turn_id:
                raise AssertionError(
                    f"Turn id mismatch for domain '{domain}' and uuid '{uuid}': "
                    f"ground truth={gt_turn_id!r}, prediction={pred_turn_id!r}"
                )

            pred_query = pred_final_turn.get("query")
            if gt_query != pred_query:
                raise AssertionError(
                    f"Query mismatch for domain '{domain}' and uuid '{uuid}': "
                    f"ground truth={gt_query!r}, prediction={pred_query!r}"
                )

        gt_sequence = gt_final_turn.get("sequence") or {}
        pred_sequence = pred_final_turn.get("sequence") or {}
        gt_tool_call = gt_sequence.get("tool_call")
        pred_tool_call = pred_sequence.get("tool_call") or {}
        gt_tool_call_len = len(gt_tool_call) if isinstance(gt_tool_call, list) else 0
        pred_tool_call_len = len(pred_tool_call) if isinstance(pred_tool_call, list) else 0

        result_record = sample["result"]
        result_details = result_record.get("details") if isinstance(result_record, dict) else None
        per_turn = result_details.get("per_turn") if isinstance(result_details, dict) else None
        final_per_turn = per_turn[-1] if isinstance(per_turn, list) and per_turn else {}
        raw_final_metadata = final_per_turn.get("metadata") if isinstance(final_per_turn, dict) else {}
        final_metadata_missing = not isinstance(raw_final_metadata, dict)
        final_metadata = raw_final_metadata if isinstance(raw_final_metadata, dict) else {}

        dialogue_score = (
            get_nested_value(result_record, ["details", "dialogue_score"])
            if isinstance(result_record, dict)
            else None
        )
        exactmatch_score = final_metadata.get("exactmatch_score")
        if exactmatch_score is None and isinstance(result_record, dict):
            exactmatch_score = get_nested_value(result_record, ["details", "exactmatch_score"])
        answer_score = final_metadata.get("answer_score")
        if answer_score is None and isinstance(result_record, dict):
            answer_score = get_nested_value(result_record, ["details", "answer_score"])
        groundedness_score = final_metadata.get("groundedness_score")
        if groundedness_score is None and isinstance(result_record, dict):
            groundedness_score = get_nested_value(result_record, ["details", "groundedness_score"])
        extra_steps = final_metadata.get("extra_steps")
        if extra_steps is None and isinstance(result_record, dict):
            extra_steps = get_nested_value(result_record, ["details", "extra_steps"])

        expected_extra_steps = max(0,(pred_tool_call_len - gt_tool_call_len))
        log_errors: list[str] = []
        log_errors.extend(domain_path_errors)
        if missing_prediction_error is not None:
            log_errors.append(missing_prediction_error)
        missing_evaluation_result = (
            (prediction_record is not None and final_metadata_missing)
            or domain_result_error is not None
            or result_record is None
        )
        if missing_evaluation_result:
            log_errors.append(
                f"For uuid '{uuid}' and domain '{domain}', evaluation results are missing for this prediction output."
            )
        elif extra_steps != expected_extra_steps:
            log_errors.append(
                f"extra_steps mismatch for domain '{domain}' and uuid '{uuid}': "
                f"result={extra_steps!r}, expected={expected_extra_steps!r}"
            )

        compare_tool_call = compare_tool_calls(gt_tool_call, pred_tool_call)

        flattened_samples.append(
            {
                "uuid": uuid,
                "domain": domain,
                "query": gt_query,
                "num_turns": input_record.get("num_turns", 1),
                "type": input_record.get("type", "(API)"),
                "question_type": extract_question_type(input_record, input_dir, gt_tool_call),
                "additional_instructions": input_record.get("additional_instructions"),
                "dialogue_score": dialogue_score,
                "exactmatch_score": exactmatch_score,
                "answer_score": answer_score,
                "groundedness_score": groundedness_score,
                "extra_steps": extra_steps,
                "compare_tool_call": compare_tool_call,
                "gt_tool_call": gt_tool_call,
                "pred_tool_call": pred_tool_call,
                "gt_answer": gt_final_turn.get("answer"),
                "pred_answer": pred_final_turn.get("answer"),
                "log_errors": log_errors,
                "input": input_record,
                "ground_truth_output": gt_output_record,
                "prediction": prediction_record,
                "result": sample["result"],
            }
        )

    domain_summary = {
        "domain": domain,
        "counts": {
            "input": len(input_by_uuid),
            "ground_truth_output": len(gt_output_by_uuid),
            "prediction": len(prediction_by_uuid),
            "results": len(result_records),
            "aligned_samples": len(flattened_samples),
        },
        "missing": {
            "ground_truth_output": missing_in_gt_output,
            "prediction": missing_in_predictions,
            "results": missing_in_results,
        },
        "extra": {
            "ground_truth_output": extra_gt_output,
            "prediction": extra_predictions,
            "results": extra_results,
        },
    }
    return flattened_samples, domain_summary


def main() -> None:
    args = parse_args()

    for path_label, path in (
        ("Input directory", args.input_dir),
        ("Ground-truth output directory", args.gt_output_dir),
        ("Prediction directory", args.prediction_dir),
        ("Results file", args.results_file),
    ):
        validate_path(path, path_label)

    available_domains = list_domains(args.input_dir)
    domains = sorted(set(args.domains)) if args.domains else available_domains
    unknown_domains = sorted(set(domains) - set(available_domains))
    if unknown_domains:
        raise ValueError(
            f"Requested domains not present in input directory: {', '.join(unknown_domains)}"
        )

    results_payload = load_json(args.results_file)
    if not isinstance(results_payload, dict):
        raise ValueError("Results file must contain a JSON object.")
    results_by_domain = results_payload.get("domains")
    if not isinstance(results_by_domain, dict):
        raise ValueError("Results file must contain a 'domains' object.")

    aligned_samples: list[dict[str, Any]] = []
    summary = {
        "domains_requested": domains,
        "total_domains": len(domains),
        "total_samples": 0,
        "domains_with_missing_records": [],
        "domains_with_extra_records": [],
    }

    for domain in domains:
        domain_samples, domain_summary = align_domain(
            domain=domain,
            input_dir=args.input_dir,
            gt_output_dir=args.gt_output_dir,
            prediction_dir=args.prediction_dir,
            results_by_domain=results_by_domain,
        )
        aligned_samples.extend(domain_samples)
        summary["total_samples"] += domain_summary["counts"]["aligned_samples"]

        if any(domain_summary["missing"].values()):
            summary["domains_with_missing_records"].append(domain)
        if any(domain_summary["extra"].values()):
            summary["domains_with_extra_records"].append(domain)

    slimmed_samples = [
        {
            key: value
            for key, value in sample.items()
            if key not in {"input", "ground_truth_output", "prediction", "result"}
        }
        for sample in aligned_samples
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            slimmed_samples,
            handle,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        handle.write("\n")

    with_output_path = build_with_output_path(args.output_file)
    with with_output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            aligned_samples,
            handle,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
