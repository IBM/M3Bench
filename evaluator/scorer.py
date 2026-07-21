from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from utils import ToolCall
from judge import CorrectnessJudge, GroundednessJudge, LLMJudge, JudgeValidationError
from utils import JudgeInput
from constant import N_TOOL_CALLS_PER_TURN, PRED_OUTPUT_KEY, PRED_OUTPUT_TURN_ID_KEY, PRED_OUTPUT_QUERY_KEY, PRED_OUTPUT_ANSWER_KEY, PRED_OUTPUT_SEQUENCE_KEY, GT_OUTPUT_KEY, GT_OUTPUT_TURN_ID_KEY, GT_OUTPUT_QUERY_KEY, GT_OUTPUT_ANSWER_KEY, GT_OUTPUT_SEQUENCE_KEY


def _load_policy_judge() -> Optional[Any]:
    if find_spec("policy_judge") is None:
        return None

    policy_module = import_module("policy_judge")
    judge_cls = getattr(policy_module, "PolicyAdherenceJudge", None)
    if judge_cls is None:
        judge_cls = getattr(policy_module, "PolicyAdheranceJudge")
    return judge_cls()

# -----------------------------
# Output Scorer
# -----------------------------

@dataclass(frozen=True)
class TurnScorerConfig:
    """
    Config for tool trajectory and final answer scoring.

    Scoring approach:
    - predicted response grounded in tool_response
    - ground truth answer and predicted answer are evaluated using an LLM-as-Judge

    """
    capability: str
    domain: str

class TurnScorer:
    """
    Compares GT vs predicted tool trajectory per turn based on final answer. Produces a binary score to evaluate a turn.
    """
    def __init__(self, cfg: TurnScorerConfig | None = None,
                         correctness_judge: Optional[LLMJudge] = None,
                         groundedness_judge: Optional[LLMJudge] = None,
                         exactmatch_judge: Optional[LLMJudge] = None):
        self.cfg = cfg or TurnScorerConfig()
        self.correctness_judge=correctness_judge
        self.groundedness_judge=groundedness_judge
        self.exactmatch_judge=exactmatch_judge
        self.policy_judge=_load_policy_judge()

    def compare(self, query: str, additional_instructions:str
                , gt_answer: str, pred_answer: str
                , gt: Sequence[ToolCall], pred: Sequence[ToolCall]
                , gt_responses: Sequence[Any], pred_responses: Sequence[Any]) -> Tuple[float, Dict[str, Any]]:
        # Run tools in pred and ascertain if predicted answer is grounded on tool call answers.
        # Use Judge to ascertain pred answer is answering the query.
        input = JudgeInput(
            capability=self.cfg.capability,
            domain=self.cfg.domain,
            query=query,
            additional_instructions=additional_instructions,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            gt_tool_calls=gt,
            pred_tool_calls=pred,
            gt_tool_responses=gt_responses,
            pred_tool_responses=pred_responses,
        )
        
        # Scoring Turns
        extra_steps=len(pred)-len(gt)

        # Check for policy adherence when an internal policy_judge.py file is present.
        if self.policy_judge is not None and "multiturn" in self.cfg.capability:
            policy = self.policy_judge.judge(inp=input)
            policy_score, policy_explanation = float(policy.score), policy.explanation
            if policy_score==0.0:
                score = policy_score
                details = {
                    "gt_steps": len(gt),
                    "pred_steps": len(pred),
                    "extra_steps": max(0, extra_steps),
                    "policy_adherance_score": policy_score,
                    "exactmatch_score": None,
                    "answer_score": None,
                    "groundedness_score": None,
                    "score_explanation": {"policy": policy_explanation, "answer": None, "exactmatch": None, "groundedness":None},
                }
                return score, details

        unanswerable_no_tool_case = (
            len(input.pred_tool_calls) == 0
            and "i can not answer" in input.gt_answer.lower()
            and len(input.gt_tool_calls) == 0
        )
        if unanswerable_no_tool_case:
            exactmatch_score = 1.0
            exactmatch_explanation = "Special case applied - GT indicates unanswerable and no tool calls made, granting full exact match credit."
        else:
            exactmatch=self.exactmatch_judge.judge(inp=input)
            exactmatch_score, exactmatch_explanation=float(exactmatch.score), exactmatch.explanation
        if exactmatch_score==0.0:
            correctness=self.correctness_judge.judge(inp=input)
            answer_score, answer_explanation=float(correctness.score), correctness.explanation
            if answer_score==0.0:
                score = answer_score
                details = {
                    "gt_steps": len(gt),
                    "pred_steps": len(pred),
                    "extra_steps": max(0, extra_steps),
                    "exactmatch_score": exactmatch_score,
                    "answer_score": answer_score,
                    "groundedness_score": None,
                    "score_explanation": {"answer": answer_explanation, "exactmatch": exactmatch_explanation, "groundedness":None},
                }
                return score, details
            elif answer_score==1.0:
                if (
                    "multiturn" in self.cfg.capability
                    and "i can not answer" in input.gt_answer.lower()
                    and (len(input.pred_tool_calls) == 0 or input.pred_answer in ["", " "])
                ):
                    score = 1.0
                    details = {
                        "gt_steps": len(gt),
                        "pred_steps": len(pred),
                        "extra_steps": max(0, extra_steps),
                        "exactmatch_score": exactmatch_score,
                        "answer_score": answer_score,
                        "groundedness_score": 1.0,
                        "score_explanation": {
                            "answer": answer_explanation,
                            "exactmatch": exactmatch_explanation,
                            "groundedness": "Special case applied - GT indicates unanswerable and no tool calls made, granting full groundedness credit.",
                        },
                    }
                    return score, details

                groundedness = self.groundedness_judge.judge(inp=input)
                groundedness_score, groundedness_explanation = float(groundedness.score), groundedness.explanation
                score = groundedness_score
                details = {
                    "gt_steps": len(gt),
                    "pred_steps": len(pred),
                    "extra_steps": max(0, extra_steps),
                    "exactmatch_score": exactmatch_score,
                    "answer_score": answer_score,
                    "groundedness_score": groundedness_score,
                    "score_explanation": {"answer": answer_explanation, "exactmatch": exactmatch_explanation, "groundedness":groundedness_explanation},
                }
                return score, details
            else:
                raise JudgeValidationError(
                    f"Unexpected answer_score from correctness_judge: {answer_score!r}. "
                    f"Expected 0.0 or 1.0. Explanation: {answer_explanation!r}"
                )
        elif exactmatch_score==1.0:
            if (
                "multiturn" in self.cfg.capability
                and "i can not answer" in input.gt_answer.lower()
                and (len(input.pred_tool_calls) == 0 or input.pred_answer in ["", " "])
            ):
                score = 1.0
                details = {
                    "gt_steps": len(gt),
                    "pred_steps": len(pred),
                    "extra_steps": max(0, extra_steps),
                    "exactmatch_score": exactmatch_score,
                    "answer_score": None,
                    "groundedness_score": 1.0,
                    "score_explanation": {
                        "answer": None,
                        "exactmatch": exactmatch_explanation,
                        "groundedness": "Special case applied - GT indicates unanswerable and no tool calls made, granting full groundedness credit.",
                    },
                }
                return score, details

            groundedness = self.groundedness_judge.judge(inp=input)
            groundedness_score, groundedness_explanation = float(groundedness.score), groundedness.explanation
            score = groundedness_score
            details = {
                "gt_steps": len(gt),
                "pred_steps": len(pred),
                "extra_steps": max(0, extra_steps),
                "exactmatch_score": exactmatch_score,
                "answer_score": None,
                "groundedness_score": groundedness_score,
                "score_explanation": {"answer": None, "exactmatch": exactmatch_explanation, "groundedness":groundedness_explanation},
            }
            return score, details
        else:
            raise JudgeValidationError(
                f"Unexpected exactmatch_score: {exactmatch_score!r}. Expected 0.0 or 1.0. "
                f"exactmatch_explanation={exactmatch_explanation!r}"
            )

@dataclass(frozen=True)
class DialogueScorerConfig:
    """
    How to aggregate turn scores into a dialogue score.
    - "mean": average of per-turn scores
    - "sum": sum of per-turn scores
    - "min": minimum per-turn score (strict)
    """
    aggregate: str = "mean"  # "mean" | "sum" | "min"


class DialogueScorer:
    """
    Scores a whole dialogue by scoring each turn using TurnScorer and aggregating.

    Expected dialogue format (ground truth example):
    dialogue = {
      "output": [
        {
          "turn_id": 0,
          "query": "...",
          "answer": [[100]],
          "sequence":
            {
              "tool_call": [{"name": "...", "arguments": {...}}],
              "tool_response": [[100]]
            }
        },
        ...
      ]
    }

    Expected predicted format (one reasonable option):
    pred_dialogue = {
      "output": [
        {
          "turn_id": 0,
          "query": "...",          
          "predicted_answer": "100",
          "sequence": 
            {
              "tool_call": [{"name": "...", "arguments": {...}}],
              "tool_response": [[100]]
            }
        },
        ...
      ]
    }

    Notes:
    - Tool responses passed to TurnScorer.compare() are taken from the *predicted* tool_response if present,
      otherwise from ground-truth tool_response (fallback).
    """

    def __init__(
        self,
        turn_scorer: TurnScorer,
        cfg: Optional[DialogueScorerConfig] = None,
    ):
        self.turn_scorer = turn_scorer
        self.cfg = cfg or DialogueScorerConfig()

    def score(
        self,
        gt_dialogue: Dict[str, Any],
        pred_dialogue: Dict[str, Any],
        gt_key: str = "output",
        pred_key: str = "output",
    ) -> Tuple[float, Dict[str, Any]]:
        gt_turns: List[Dict[str, Any]] = list(gt_dialogue.get(gt_key, []))
        pred_turns: List[Dict[str, Any]] = list(pred_dialogue.get(pred_key, []))
        additional_instructions = gt_dialogue.get("additional_instructions","")
        
        assert len(pred_turns) == 1, f"Predicted Turns {len(pred_turns)} should have been 1."

        pred_by_id = {t.get(PRED_OUTPUT_TURN_ID_KEY): t for t in pred_turns if "turn_id" in t}

        per_turn: List[Dict[str, Any]] = []
        turn_scores: List[float] = []

        for gt_turn in [gt_turns[-1]]: # Only the last turn evaluated against prediction
            turn_id = gt_turn.get(GT_OUTPUT_TURN_ID_KEY)
            query = str(gt_turn.get(GT_OUTPUT_QUERY_KEY))
            gt_answer = self._stringify_answer(gt_turn.get(GT_OUTPUT_ANSWER_KEY))
            gt_calls = gt_turn.get(GT_OUTPUT_SEQUENCE_KEY, {}).get("tool_call",[])
            gt_responses = self._extract_tool_responses(gt_turn.get(GT_OUTPUT_SEQUENCE_KEY, {}).get("tool_response",[]))

            pred_turn = pred_by_id.get(turn_id, None)
            pred_answer = self._stringify_pred_answer(pred_turn.get(PRED_OUTPUT_ANSWER_KEY, ""))
            pred_sequence = pred_turn.get(PRED_OUTPUT_SEQUENCE_KEY, {}) or {}
            pred_calls_all = pred_sequence.get("tool_call", []) or []
            pred_calls = pred_calls_all[-N_TOOL_CALLS_PER_TURN:] if isinstance(pred_calls_all, list) else []
            pred_responses = self._extract_tool_responses(pred_sequence.get("tool_response", []))[-N_TOOL_CALLS_PER_TURN:]

            score, details = self.turn_scorer.compare(
                query=query,
                gt_answer=gt_answer,
                pred_answer=pred_answer,
                additional_instructions=additional_instructions,
                gt=gt_calls,
                pred=pred_calls,
                gt_responses=gt_responses,
                pred_responses=pred_responses,
            )

            per_turn.append(
                {
                    "turn_id": turn_id,
                    "query": query,
                    "pred_answer": pred_answer,
                    "score": score,
                    "metadata":details,
                }
            )
            turn_scores.append(float(score))

        dialogue_score = self._aggregate(turn_scores)

        details = {
            "dialogue_score": dialogue_score,
            "num_turns": len(gt_turns),
            "per_turn": per_turn,
            "aggregate": self.cfg.aggregate,
        }
        return dialogue_score, details

    def _aggregate(self, scores: Sequence[float]) -> float:
        if not scores:
            return 0.0

        mode = self.cfg.aggregate
        if mode == "sum":
            return float(sum(scores))
        if mode == "min":
            return float(min(scores))
        # default: mean
        return float(sum(scores) / len(scores))

    def _stringify_answer(self, answer_obj: Any) -> str:
        """
        Converts nested lists like [[1], [3], [15]] or flat lists
        into a stable comma-separated string.
        """
        if answer_obj is None:
            return ""

        try:
            if isinstance(answer_obj, list):
                flattened = []

                for item in answer_obj:
                    if isinstance(item, list):
                        # Take first element if sublist not empty
                        if item:
                            flattened.append(str(item[0]))
                    else:
                        flattened.append(str(item))

                return ", ".join(flattened)

        except Exception:
            pass

        return str(answer_obj)

    def _stringify_pred_answer(self, pred_answer_obj: Any) -> str:
        """
        Predicted answer might be a string already or follow the same nested structure as GT.
        """
        if pred_answer_obj is None:
            return ""
        if isinstance(pred_answer_obj, str):
            return pred_answer_obj
        return self._stringify_answer(pred_answer_obj)

    def _extract_tool_responses(self, sequence: Any) -> List[Any]:
        """
        Extract responses aligned with the predicted tool execution.
        We keep it simple: flatten tool_response to a 1D list of payloads.
        """
        responses: List[Any] = []
        if not isinstance(sequence, list):
            return responses

        for step in sequence:
            if isinstance(step, str):
                continue
        return sequence

    def _flatten_2d(self, obj: Any) -> List[List[Any]]:
        """
        Normalize possibly nested [ [ ... ] ] shape into List[List[Any]].
        If input is already flat list of dicts, treat it as one group.
        """
        if obj is None:
            return []
        if isinstance(obj, list):
            if not obj:
                return []
            # already list-of-lists
            if all(isinstance(x, list) for x in obj):
                return obj  # type: ignore[return-value]
            # flat list -> wrap
            return [obj]  # type: ignore[list-item]
        # non-list -> wrap twice
        return [[obj]]
    

if __name__ == "__main__":
    # ---- Example ground-truth dialogue ----
    gt_dialogue = {
        GT_OUTPUT_KEY: [
            {
                GT_OUTPUT_TURN_ID_KEY: 0,
                GT_OUTPUT_QUERY_KEY: "How many students have never been absent from school?",
                GT_OUTPUT_ANSWER_KEY: [[100]],
                GT_OUTPUT_SEQUENCE_KEY: 
                    {
                        "tool_call": [
                                {
                                    "name": "get_count_names_by_month_v1_student_loan_count_names_by_month_get",
                                    "arguments": {"month": "0"},
                                }
                        ],
                        "tool_response": [[100]],
                    }
            }
        ]
    }

    # ---- Example predicted dialogue ----
    pred_dialogue = {
        PRED_OUTPUT_KEY: [
            {
                PRED_OUTPUT_TURN_ID_KEY: 0,
                PRED_OUTPUT_QUERY_KEY: "How many students have never been absent from school?",
                PRED_OUTPUT_ANSWER_KEY: "100",
                PRED_OUTPUT_SEQUENCE_KEY: 
                    {
                        "tool_call": [
                            [
                                {
                                    "name": "get_count_names_by_month_v1_student_loan_count_names_by_month_get",
                                    "arguments": {"month": "0"},
                                }
                            ]
                        ],
                        "tool_response": [[500]],
                    }
            }
        ]
    }

    turn_cfg = TurnScorerConfig(
        capability="capability2",
        domain="student_loan",
        answer_weight=0.5,
        trajectory_weight=0.5,
        extra_step_penalty=0.1,
        )

    turn_scorer = TurnScorer(cfg=turn_cfg,
                             correctness_judge=CorrectnessJudge(),
                             groundedness_judge=GroundednessJudge(),
                             exactmatch_judge=ExactMatchJudge(config={}))

    dialogue_cfg = DialogueScorerConfig(
        aggregate="mean",          # mean | sum | min
    )

    dialogue_scorer = DialogueScorer(
        turn_scorer=turn_scorer,
        cfg=dialogue_cfg,
    )

    dialogue_score, dialogue_details = dialogue_scorer.score(
        gt_dialogue=gt_dialogue,
        pred_dialogue=pred_dialogue,
    )
    print("Dialogue details:", dialogue_details)
    print("Dialogue score:", dialogue_score)
