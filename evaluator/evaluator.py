from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from tqdm import tqdm

from judge import CorrectnessJudge, GroundednessJudge, ExactMatchJudge
from benchmark.mcp_client import (
    MCPConnectionConfig,
    load_mcp_config,
    create_client_and_connect,
)
from mcp_tools import (
    execute_tools_batch,
    extract_toolcalls_for_mcp,
    inject_mcp_responses,
)
from scorer import (
    DialogueScorer,
    DialogueScorerConfig,
    TurnScorer,
    TurnScorerConfig,
)
from constant import PRED_OUTPUT_KEY, PRED_OUTPUT_SEQUENCE_KEY
from constant import N_TOOL_CALLS_PER_TURN
from utils import read_domain_file, pair_dialogues_by_uuid

CAPABILITY_MCP_TOOL_MAP={
    "capability_bi_apis": 1,
    "capability_dashboard_apis": 2,
    "capability_multihop_reasoning": 3,
    "capability_multiturn": 4,
}

# -----------------------------
# Capability policy / registry
# -----------------------------


@dataclass(frozen=True)
class CapabilityPolicy:

    dialogue_aggregate: str = "mean"        # "mean" | "sum" | "min"

    # --- MCP execution ---
    execute_mcp_tools: bool = True

    # --- Judges ---
    correctness_judge: Any = CorrectnessJudge(config={})
    groundedness_judge: Any = GroundednessJudge(config={})
    exactmatch_judge: Any = ExactMatchJudge(config={})



def build_default_capability_registry() -> Dict[str, CapabilityPolicy]:
    return {
        "capability_bi_apis": CapabilityPolicy(
            execute_mcp_tools=True,
            correctness_judge=CorrectnessJudge(config={}),
            groundedness_judge=GroundednessJudge(config={}),
            exactmatch_judge=ExactMatchJudge(config={})
        ),

        "capability_dashboard_apis": CapabilityPolicy(
            execute_mcp_tools=True,
            correctness_judge=CorrectnessJudge(config={}),
            groundedness_judge=GroundednessJudge(config={}),
            exactmatch_judge=ExactMatchJudge(config={})            
        ),

        "capability_multihop_reasoning": CapabilityPolicy(
            execute_mcp_tools=True,
            correctness_judge=CorrectnessJudge(config={}),
            groundedness_judge=GroundednessJudge(config={}),
            exactmatch_judge=ExactMatchJudge(config={})            
        ),
        "capability_multiturn": CapabilityPolicy(
            execute_mcp_tools=True,
            correctness_judge=CorrectnessJudge(config={}),
            groundedness_judge=GroundednessJudge(config={}),
            exactmatch_judge=ExactMatchJudge(config={})            
        ),        
    }


# -----------------------------
# capability_bi_apis helpers
# -----------------------------


def _prepend_get_data_to_batch(
    batch_tools: List[List[List[Dict[str, Any]]]],
    uuids: List[str],
    skip_initialize_active_data: bool = False,
) -> List[List[List[Dict[str, Any]]]]:
    """Prepend get_data(tool_universe_id=uuid) to first turn of each dialogue.

    Optionally removes initialize_active_data from first turn (for ground truth).
    """
    result = []
    for dialogue_tools, uuid in zip(batch_tools, uuids):
        if not dialogue_tools:
            result.append(dialogue_tools)
            continue
        first_turn = list(dialogue_tools[0])
        if skip_initialize_active_data and first_turn and (
            first_turn[0].get("name") == "initialize_active_data"
            or first_turn[0].get("name") == "get_data"
        ):
            first_turn = first_turn[1:]
        first_turn = [{"name": "get_data", "arguments": {"tool_universe_id": uuid}}] + first_turn
        result.append([first_turn] + list(dialogue_tools[1:]))
    return result


def _update_dialogue_toolcall_for_get_data(
    dialogue: Dict[str, Any],
    uuid: str,
    skip_initialize_active_data: bool = False,
) -> None:
    """Update dialogue's sequence.tool_call in-place to match modified execution sequence.

    Must be called before inject_mcp_responses to keep tool_call and tool_response aligned.
    """
    turns = dialogue.get(PRED_OUTPUT_KEY, [])
    if not turns:
        return
    seq = turns[0].get(PRED_OUTPUT_SEQUENCE_KEY) or {}
    if not isinstance(seq, dict):
        return
    tool_calls = list(seq.get("tool_call", []))
    if skip_initialize_active_data and tool_calls and tool_calls[0].get("name") == "initialize_active_data":
        tool_calls = tool_calls[1:]
    tool_calls = [{"name": "get_data", "arguments": {"tool_universe_id": uuid}}] + tool_calls
    seq["tool_call"] = tool_calls
    turns[0][PRED_OUTPUT_SEQUENCE_KEY] = seq


# -----------------------------
# Evaluator core
# -----------------------------


async def evaluate_domain(
    domain: str,
    gt_path: Path,
    pred_path: Path,
    policy: CapabilityPolicy,
    mcp_config: Optional[MCPConnectionConfig],
    capability_name: str,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Evaluate a single domain (async version).

    Returns:
        Tuple of (domain_out dict, dialogue_scores list)
    """
    # Read data files
    gt_list = read_domain_file(gt_path)
    pred_list = read_domain_file(pred_path) if pred_path.exists() else []

    # Pair dialogues
    paired, missing_pred, extra_pred = pair_dialogues_by_uuid(gt_list, pred_list)    

    if len(pred_list) == 0:
        domain_out: Dict[str, Any] = {
            "domain": domain,
            "n_groundtruth": len(gt_list),
            "n_prediction": len(pred_list),
            "n_paired": len(paired),
            "missing_prediction_uuids": missing_pred,
            "extra_prediction_uuids": extra_pred,
            "dialogues": [],
            "summary":{
                    "num_samples": len(gt_list),
                    "num_correct": 0.0,
                    "mean_dialogue_score": 0.0,
                    "min_dialogue_score":  0.0,
                    "max_dialogue_score": 0.0,
                        }
            }
        return domain_out, []

    # Build scorers per domain
    turn_cfg = TurnScorerConfig(
        capability=capability_name,
        domain=domain,
    )
    turn_scorer = TurnScorer(
        cfg=turn_cfg,
        correctness_judge=policy.correctness_judge,
        groundedness_judge=policy.groundedness_judge,
        exactmatch_judge=policy.exactmatch_judge
    )
    dialogue_scorer = DialogueScorer(
        turn_scorer=turn_scorer,
        cfg=DialogueScorerConfig(
            aggregate=policy.dialogue_aggregate,
        ),
    )

    domain_out: Dict[str, Any] = {
        "domain": domain,
        "n_groundtruth": len(gt_list),
        "n_prediction": len(pred_list),
        "n_paired": len(paired),
        "missing_prediction_uuids": missing_pred,
        "extra_prediction_uuids": extra_pred,
        "dialogues": [],
    }

    dialogue_scores: List[float] = []

    # MCP execution and scoring
    if policy.execute_mcp_tools and mcp_config and paired:
        async with create_client_and_connect(mcp_config, domain) as session:
            # Get schema from tools
            tools_result = await session.list_tools()
            schema_map = {tool.name: tool.inputSchema for tool in tools_result.tools}

            # Batch execute tools
            batch_tools_pred = [extract_toolcalls_for_mcp(pr, limit_last_n=N_TOOL_CALLS_PER_TURN) for _, pr in paired]
            batch_tools_gt = [extract_toolcalls_for_mcp(gt) for gt, _ in paired]

            # capability_bi_apis: replace initialize_active_data with get_data (GT),
            # and prepend get_data (pred), then sync dialogue tool_call sequences
            if capability_name == "capability_bi_apis":
                uuids = [str(gt.get("uuid")) for gt, _ in paired]
                batch_tools_gt = _prepend_get_data_to_batch(batch_tools_gt, uuids, skip_initialize_active_data=True)
                batch_tools_pred = _prepend_get_data_to_batch(batch_tools_pred, uuids, skip_initialize_active_data=False)
                for (gt_raw, pr_raw), uuid in zip(paired, uuids):
                    _update_dialogue_toolcall_for_get_data(gt_raw, uuid, skip_initialize_active_data=True)
                    _update_dialogue_toolcall_for_get_data(pr_raw, uuid, skip_initialize_active_data=False)

            mcp_batch_responses_pred = await execute_tools_batch(session, batch_tools_pred, schema_map)
            mcp_batch_responses_gt = await execute_tools_batch(
                session,
                batch_tools_gt,
                schema_map,
                is_gt=capability_name == "capability_bi_apis",
            )

            # Score each paired dialogue
            for idx, (gt_raw, pr_raw) in enumerate(
                tqdm(paired, desc=f"[{capability_name}][{domain}]", leave=False)
            ):
                uuid = str(gt_raw.get("uuid"))

                # Inject fresh responses so groundedness judge uses tool outputs
                inject_mcp_responses(pr_raw, mcp_batch_responses_pred[idx], type="pred", capability_name=capability_name)
                if capability_name != "capability_bi_apis" and mcp_batch_responses_gt is not None:
                    inject_mcp_responses(gt_raw, mcp_batch_responses_gt[idx], type="gt",capability_name=capability_name)

                # Score and store details
                dialogue_score, dialogue_details = dialogue_scorer.score(
                    gt_dialogue=gt_raw,
                    pred_dialogue=pr_raw,
                    pred_key=PRED_OUTPUT_KEY
                )
                dialogue_scores.append(float(dialogue_score))

                domain_out["dialogues"].append({
                    "uuid": uuid,
                    "score": float(dialogue_score),
                    "metadata": {
                        "capability": capability_name,
                        "domain": domain,
                        "policy": {
                            "dialogue_aggregate": policy.dialogue_aggregate,
                            "execute_mcp_tools": policy.execute_mcp_tools,
                        },
                    },
                    "details": dialogue_details,
                })
    else:
        # No MCP tools - just score based on predictions as-is
        for idx, (gt_raw, pr_raw) in enumerate(
            tqdm(paired, desc=f"[{capability_name}][{domain}]", leave=False)
        ):
            uuid = str(gt_raw.get("uuid"))

            dialogue_score, dialogue_details = dialogue_scorer.score(
                gt_dialogue=gt_raw,
                pred_dialogue=pr_raw,
                pred_key=PRED_OUTPUT_KEY
            )
            dialogue_scores.append(float(dialogue_score))

            domain_out["dialogues"].append({
                "uuid": uuid,
                "score": float(dialogue_score),
                "metadata": {
                    "capability": capability_name,
                    "domain": domain,
                    "policy": {
                        "dialogue_aggregate": policy.dialogue_aggregate,
                        "execute_mcp_tools": policy.execute_mcp_tools,
                    },
                },
                "details": dialogue_details,
            })

    # Domain summary
    domain_scores = [d["score"] for d in domain_out["dialogues"]]
    domain_out["summary"] = {
        "num_samples": len(gt_list),
        "num_correct": sum(domain_scores),
        "mean_dialogue_score": (sum(domain_scores) / len(domain_scores)) if domain_scores else 0.0,
        "min_dialogue_score": min(domain_scores) if domain_scores else 0.0,
        "max_dialogue_score": max(domain_scores) if domain_scores else 0.0,
    }

    return domain_out, dialogue_scores

def _load_existing_results(out_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load existing results file if it exists.

    Returns:
        Existing results dict, or None if file doesn't exist or is invalid.
    """
    if not out_path.exists():
        return None

    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "domains" in data:
            print(f"Found existing results with {len(data['domains'])} completed domain(s)")
            return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load existing results from {out_path}: {e}")

    return None


def _write_intermediate_results(
    results: Dict[str, Any],
    all_dialogue_scores: List[float],
    out_path: Path,
) -> None:
    """
    Write intermediate results to disk.

    Updates the summary with current statistics and writes to the output file.
    This allows recovery of partial results if evaluation is interrupted.
    """
    # Calculate totals from completed domains
    total_paired = sum(
        d["n_paired"] for d in results["domains"].values()
    )
    total_missing = sum(
        len(d["missing_prediction_uuids"])
        for d in results["domains"].values()
    )
    total_extra = sum(
        len(d["extra_prediction_uuids"])
        for d in results["domains"].values()
    )

    total_samples = sum(
        d["summary"]["num_samples"] for d in results["domains"].values()
    )

    # Update summary with current progress
    num_correct=(sum(all_dialogue_scores) if all_dialogue_scores else 0.0)
    results["summary"] = {
        "n_domains": len(results["domains"]),
        "n_paired_dialogues": total_paired,
        "n_missing_predictions": total_missing,
        "n_extra_predictions": total_extra,
        "n_samples": total_samples,
        "n_correct": num_correct,
        "mean_dialogue_score": (num_correct/total_samples
            if all_dialogue_scores
            else 0.0
        ),
        "min_dialogue_score": (
            min(all_dialogue_scores) if all_dialogue_scores else 0.0
        ),
        "max_dialogue_score": (
            max(all_dialogue_scores) if all_dialogue_scores else 0.0
        ),
    }

    # Write to disk
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return results


def evaluate_capability(
    capability_name: str,
    gt_dir: Path,
    pred_dir: Path,
    out_path: Path,
    registry: Dict[str, CapabilityPolicy],
    mcp_config: Optional[MCPConnectionConfig] = None,
    selected_domains: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if capability_name not in registry:
        raise ValueError(
            f"Capability '{capability_name}' not found in registry. Add it in build_default_capability_registry()."
        )

    policy = registry[capability_name]

    # Discover domain files from groundtruth folder
    all_gt_files = sorted([p for p in gt_dir.glob("*.json") if p.is_file()])

    if selected_domains:
        gt_files = [p for p in all_gt_files if p.stem in selected_domains]

        missing = selected_domains - {p.stem for p in all_gt_files}
        if missing:
            raise ValueError(
                f"Requested domains not found in groundtruth_dir: {sorted(missing)}"
            )
    else:
        gt_files = all_gt_files

    if not gt_files:
        raise ValueError("No matching domain files found for evaluation.")

    # Load existing results if available (for resume capability)
    existing_results = _load_existing_results(out_path)
    if existing_results:
        results = existing_results
        # Reconstruct all_dialogue_scores from existing results
        all_dialogue_scores: List[float] = []
        for domain_data in results["domains"].values():
            all_dialogue_scores.extend([d["score"] for d in domain_data["dialogues"]])
    else:
        results: Dict[str, Any] = {
            "capability_name": capability_name,
            "groundtruth_dir": str(gt_dir),
            "prediction_dir": str(pred_dir),
            "domains": {},
            "summary": {},
        }
        all_dialogue_scores: List[float] = []

    # Get list of already-completed domains
    completed_domains = set(results["domains"].keys())

    for gt_path in gt_files:
        domain = gt_path.stem

        # Skip already-completed domains
        if domain in completed_domains:
            print(f"Skipping already-completed domain: {domain}")
            continue

        pred_path = pred_dir / gt_path.name

        print(f"\nEvaluating domain: {domain}")
        # Run async evaluation for this domain
        domain_out, domain_scores = asyncio.run(
            evaluate_domain(
                domain=domain,
                gt_path=gt_path,
                pred_path=pred_path,
                policy=policy,
                mcp_config=mcp_config,
                capability_name=capability_name,
            )
        )

        all_dialogue_scores.extend(domain_scores)
        results["domains"][domain] = domain_out

        # Write intermediate results after each domain
        # This ensures partial results are saved if evaluation is interrupted
        _write_intermediate_results(results, all_dialogue_scores, out_path)

    # Final write (summary already updated by last intermediate write)
    # This is technically redundant but ensures the final state is written
    results=_write_intermediate_results(results, all_dialogue_scores, out_path)

    print("=========================================================================")
    print("==================================[RESULTS]==============================")
    print("=========================================================================")

    num_samples = results["summary"].get("n_samples", 0)
    correct_sum = results["summary"].get("n_correct", 0.0)
    accuracy = results["summary"].get("mean_dialogue_score", 0.0)

    print("Number of samples evaluated:", num_samples)
    print("Number of correct dialogues:", correct_sum)
    print("Accuracy:", accuracy)
    print("=========================================================================")

    return results


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability_name", required=True, help="Capability name (must exist in registry)")
    ap.add_argument("--gt_root", required=True, help="Path to capability_name/groundtruth/")
    ap.add_argument("--pred_root", required=True, help="Path to capability_name/prediction/")
    ap.add_argument("--output", default=None, help="Output results.json path (default: <capability_root>/results.json)")
    ap.add_argument(
        "--mcp-config",
        default="benchmark/mcp_connection_config.yaml",
        help="Path to MCP connection config YAML file (default: benchmark/mcp_connection_config.yaml)"
    )
    ap.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional list of domain names to evaluate (without .json extension). "
             "If omitted, all domains are evaluated."
    )
    args = ap.parse_args()

    capability_name = args.capability_name
    gt_dir = Path(args.gt_root)
    pred_dir = Path(args.pred_root)
    selected_domains = set(args.domains) if args.domains else None

    if not gt_dir.exists():
        raise SystemExit(f"groundtruth_dir does not exist: {gt_dir}")
    if not pred_dir.exists():
        raise SystemExit(f"prediction_dir does not exist: {pred_dir}")

    # Default output location: sibling of groundtruth/prediction under capability root
    if args.output:
        out_path = Path(args.output)
        # If output is a directory, append results.json
        if out_path.is_dir():
            out_path = out_path / "results.json"
    else:
        # assume structure: capability_name/groundtruth and capability_name/prediction
        capability_root = gt_dir.parent
        out_path = capability_root / "results.json"

    # Load MCP configs from YAML
    mcp_configs_by_capability_id = load_mcp_config(args.mcp_config)

    # Extract capability_id from capability_name (assumes format like "capability1", "capability2", etc.)
    try:
        capability_id = int(CAPABILITY_MCP_TOOL_MAP[capability_name])
        mcp_config = mcp_configs_by_capability_id.get(capability_id)
    except (ValueError, AttributeError):
        mcp_config = None

    registry = build_default_capability_registry()
    evaluate_capability(
        capability_name=capability_name,
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        out_path=out_path,
        registry=registry,
        mcp_config=mcp_config,
        selected_domains=selected_domains,
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
