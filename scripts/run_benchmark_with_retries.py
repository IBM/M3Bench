#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


CAPABILITY_NAME_MAP = {
    1: "capability_bi_apis",
    2: "capability_dashboard_apis",
    3: "capability_multihop_reasoning",
    4: "capability_multiturn",
}

DEFAULT_RETRY_ERROR_SUBSTRINGS = (
    "400 bad request",
    "litellm.badrequesterror",
    "connection closed",
)
DEFAULT_POLICY_JUDGE_PATH = Path(
    "/root/vakra-internal-read-only/evaluator/policy_judge.py"
)


@dataclass
class DomainScanResult:
    domain: str
    retryable_errors: int
    total_errors: int


@dataclass
class CommandRunResult:
    return_code: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VAKRA benchmark_runner, retry failed domains after the main run "
            "finishes, then run evaluator."
        )
    )
    parser.add_argument("--capability-id", type=int, required=True, choices=sorted(CAPABILITY_NAME_MAP))
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--vakra-root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Path to the VAKRA repo root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for benchmark and evaluator subprocesses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Prediction output directory passed to benchmark_runner.py.",
    )
    parser.add_argument(
        "--benchmark-log",
        type=Path,
        default=None,
        help="Log file for benchmark and retry runs.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Path to evaluator JSON output.",
    )
    parser.add_argument(
        "--eval-log",
        type=Path,
        default=None,
        help="Log file for evaluator.",
    )
    parser.add_argument(
        "--policy-judge-path",
        type=Path,
        default=DEFAULT_POLICY_JUDGE_PATH,
        help=(
            "Path passed to evaluator.py via --policy-judge-path "
            f"(default: {DEFAULT_POLICY_JUDGE_PATH})."
        ),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=None,
        help="Ground-truth root for evaluator. Defaults from capability id.",
    )
    parser.add_argument(
        "--capability-name",
        default=None,
        help="Evaluator capability name. Defaults from capability id.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help="Optional domain(s) to include in the main run. Can be repeated.",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=2,
        help="Maximum retry attempts per failed domain after the main run finishes.",
    )
    parser.add_argument(
        "--retry-error-substring",
        action="append",
        default=None,
        help="Retryable error substring. Can be repeated. Defaults to known transient errors.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Polling interval while waiting for benchmark/evaluator subprocesses.",
    )
    parser.add_argument(
        "--benchmark-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to benchmark_runner.py. Can be repeated.",
    )
    parser.add_argument(
        "--evaluator-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to evaluator.py. Can be repeated.",
    )
    parser.add_argument(
        "--run_only_evaluator",
        action="store_true",
        help="Skip the benchmark run and retries, and only run the evaluator. "
        "Useful for testing the evaluator or if predictions are already generated.",
    )
    return parser.parse_args()

def default_output_dir(vakra_root: Path, capability_id: int) -> Path:
    timestamp = datetime.now().strftime("%b_%d_%I_%M%p").lower()
    return vakra_root / "output" / f"capability_{capability_id}_{timestamp}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_logged_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    poll_seconds: float,
    label: str,
) -> CommandRunResult:
    ensure_parent(log_path)
    print(f"[{label}] launching: {quote_cmd(cmd)}")
    print(f"[{label}] log: {log_path}")
    started_at = time.perf_counter()

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n[{datetime.now().isoformat()}] START {label}\n")
        log_file.write(f"COMMAND: {quote_cmd(cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            return_code = process.poll()
            if return_code is not None:
                elapsed_seconds = time.perf_counter() - started_at
                log_file.write(f"[{datetime.now().isoformat()}] END {label} rc={return_code}\n")
                log_file.flush()
                print(f"[{label}] finished with exit code {return_code} in {elapsed_seconds:.2f}s")
                return CommandRunResult(
                    return_code=return_code,
                    elapsed_seconds=elapsed_seconds,
                )
            print(f"[{label}] still running (pid={process.pid})...")
            time.sleep(poll_seconds)


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_domain_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return data


def scan_retryable_domains(
    pred_root: Path,
    retry_substrings: Iterable[str],
    eligible_domains: set[str] | None = None,
) -> list[DomainScanResult]:
    normalized = tuple(part.lower() for part in retry_substrings)
    results: list[DomainScanResult] = []

    for json_path in sorted(pred_root.glob("*.json")):
        domain = json_path.stem
        if domain.endswith("_tools"):
            continue
        if eligible_domains is not None and domain not in eligible_domains:
            continue

        try:
            rows = load_domain_json(json_path)
        except Exception as exc:
            print(f"[scan] skipping unreadable file {json_path}: {exc}")
            continue

        retryable_errors = 0
        total_errors = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "error":
                continue
            total_errors += 1
            error_text = str(row.get("error", "")).lower()
            if any(piece in error_text for piece in normalized):
                retryable_errors += 1

        if retryable_errors > 0:
            results.append(
                DomainScanResult(
                    domain=domain,
                    retryable_errors=retryable_errors,
                    total_errors=total_errors,
                )
            )

    return results


def remove_previous_domain_outputs(pred_root: Path, domain: str) -> None:
    for candidate in (pred_root / f"{domain}.json", pred_root / f"{domain}_tools.json"):
        if candidate.exists():
            candidate.unlink()


def benchmark_command(
    *,
    python_exe: str,
    vakra_root: Path,
    capability_id: int,
    provider: str,
    model: str,
    temperature: float,
    output_dir: Path,
    domains: Sequence[str] | None,
    extra_args: Sequence[str],
) -> list[str]:
    cmd = [
        python_exe,
        "benchmark_runner.py",
        "--capability_id",
        str(capability_id),
        "--provider",
        provider,
        "--model",
        model,
        "--output",
        str(output_dir),
        "--temperature",
        str(temperature),
    ]
    for domain in domains or []:
        cmd.extend(["--domain", domain])
    cmd.extend(extra_args)
    return cmd


def evaluator_command(
    *,
    python_exe: str,
    capability_name: str,
    gt_root: Path,
    pred_root: Path,
    output_path: Path,
    policy_judge_path: Path,
    extra_args: Sequence[str],
) -> list[str]:
    cmd = [
        python_exe,
        "evaluator/evaluator.py",
        "--capability_name",
        capability_name,
        "--gt_root",
        str(gt_root),
        "--pred_root",
        str(pred_root),
        "--output",
        str(output_path),
        "--policy-judge-path",
        str(policy_judge_path),
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args = parse_args()
    workflow_started_at = time.perf_counter()
    vakra_root = args.vakra_root.resolve()
    capability_name = args.capability_name or CAPABILITY_NAME_MAP[args.capability_id]
    gt_root = (
        args.gt_root.resolve()
        if args.gt_root is not None
        else vakra_root / "data" / "test" / capability_name / "output"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(vakra_root, args.capability_id)
    )

    safe_model = args.model.replace("/", "_").replace(":", "_")
    benchmark_log = (
        args.benchmark_log.resolve()
        if args.benchmark_log is not None
        else vakra_root / "output" / "logs" / f"cap{args.capability_id}_{safe_model}_benchmark.log"
    )
    eval_output = (
        args.eval_output.resolve()
        if args.eval_output is not None
        else vakra_root / "output" / "evaluation" / f"cap{args.capability_id}_{safe_model}.json"
    )
    eval_log = (
        args.eval_log.resolve()
        if args.eval_log is not None
        else vakra_root / "output" / "evaluation" / f"cap{args.capability_id}_{safe_model}.log"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(benchmark_log)
    ensure_parent(eval_output)
    ensure_parent(eval_log)

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    repo_pythonpath = str(vakra_root)
    env["PYTHONPATH"] = (
        repo_pythonpath if not current_pythonpath else f"{repo_pythonpath}{os.pathsep}{current_pythonpath}"
    )

    print(f"[config] vakra_root={vakra_root}")
    print(f"[config] output_dir={output_dir}")
    print(f"[config] gt_root={gt_root}")
    print(f"[config] capability_name={capability_name}")
    if not args.run_only_evaluator:
        main_benchmark_cmd = benchmark_command(
            python_exe=args.python,
            vakra_root=vakra_root,
            capability_id=args.capability_id,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            output_dir=output_dir,
            domains=args.domain,
            extra_args=args.benchmark_extra_arg,
        )
        benchmark_result = run_logged_command(
            main_benchmark_cmd,
            cwd=vakra_root,
            env=env,
            log_path=benchmark_log,
            poll_seconds=args.poll_seconds,
            label="benchmark-main",
        )
        if benchmark_result.return_code != 0:
            print("[benchmark-main] non-zero exit code; continuing to inspect generated outputs before deciding retries/evaluation.")

        requested_domains = set(args.domain) if args.domain else None
        retry_substrings = args.retry_error_substring or list(DEFAULT_RETRY_ERROR_SUBSTRINGS)
        retry_attempts: dict[str, int] = {}
        retry_summaries: list[tuple[str, int, float, int]] = []

        for round_index in range(1, args.retry_count + 1):
            retryable_domains = scan_retryable_domains(
                output_dir,
                retry_substrings,
                eligible_domains=requested_domains,
            )
            remaining = [item for item in retryable_domains if retry_attempts.get(item.domain, 0) < args.retry_count]
            if not remaining:
                print("[retry] no retryable domains found.")
                break

            print(f"[retry] round {round_index}: domains to retry: {[item.domain for item in remaining]}")
            for item in remaining:
                retry_attempts[item.domain] = retry_attempts.get(item.domain, 0) + 1
                print(
                    f"[retry] domain={item.domain} "
                    f"retryable_errors={item.retryable_errors} total_errors={item.total_errors} "
                    f"attempt={retry_attempts[item.domain]}/{args.retry_count}"
                )
                remove_previous_domain_outputs(output_dir, item.domain)
                retry_cmd = benchmark_command(
                    python_exe=args.python,
                    vakra_root=vakra_root,
                    capability_id=args.capability_id,
                    provider=args.provider,
                    model=args.model,
                    temperature=args.temperature,
                    output_dir=output_dir,
                    domains=[item.domain],
                    extra_args=args.benchmark_extra_arg,
                )
                retry_result = run_logged_command(
                    retry_cmd,
                    cwd=vakra_root,
                    env=env,
                    log_path=benchmark_log,
                    poll_seconds=args.poll_seconds,
                    label=f"benchmark-retry-{item.domain}-attempt-{retry_attempts[item.domain]}",
                )
                retry_summaries.append(
                    (
                        item.domain,
                        retry_attempts[item.domain],
                        retry_result.elapsed_seconds,
                        retry_result.return_code,
                    )
                )
                if retry_result.return_code != 0:
                    print(f"[retry] retry command for {item.domain} exited with {retry_result.return_code}")

        final_retryable = scan_retryable_domains(
            output_dir,
            retry_substrings,
            eligible_domains=requested_domains,
        )
        if final_retryable:
            print(
                "[retry] retryable errors still present after retries for domains: "
                f"{[item.domain for item in final_retryable]}"
            )

    if not gt_root.exists():
        print(f"[error] ground-truth path does not exist: {gt_root}")
        return 1
    if not output_dir.exists():
        print(f"[error] prediction output path does not exist: {output_dir}")
        return 1

    eval_cmd = evaluator_command(
        python_exe=args.python,
        capability_name=capability_name,
        gt_root=gt_root,
        pred_root=output_dir,
        output_path=eval_output,
        policy_judge_path=args.policy_judge_path,
        extra_args=args.evaluator_extra_arg,
    )
    eval_result = run_logged_command(
        eval_cmd,
        cwd=vakra_root,
        env=env,
        log_path=eval_log,
        poll_seconds=args.poll_seconds,
        label="evaluator",
    )

    total_elapsed_seconds = time.perf_counter() - workflow_started_at
    if not args.run_only_evaluator:
        print(f"[done] benchmark main time: {format_duration(benchmark_result.elapsed_seconds)}")
        if retry_summaries:
            for domain, attempt, elapsed_seconds, return_code in retry_summaries:
                print(
                    f"[done] retry {domain} attempt {attempt}: "
                    f"{format_duration(elapsed_seconds)} (rc={return_code})"
                )
        else:
            print("[done] retries: none")
    print(f"[done] evaluator time: {format_duration(eval_result.elapsed_seconds)}")
    print(f"[done] total workflow time: {format_duration(total_elapsed_seconds)}")
    print(f"[done] predictions: {output_dir}")
    print(f"[done] evaluation json: {eval_output}")
    print(f"[done] benchmark log: {benchmark_log}")
    print(f"[done] evaluator log: {eval_log}")

    return 0 if eval_result.return_code == 0 else eval_result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
