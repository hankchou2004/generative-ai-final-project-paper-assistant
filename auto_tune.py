"""
auto_tune.py - Auto tuning helper for the inference/evaluation page.

This module keeps parameter-search logic out of eval_page.py.  It reuses the
existing evaluation pipeline:

    EvalSample -> run_inference -> compute_scores -> _agg

The main entry point is run_auto_tune().
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Literal


Objective = Literal["balanced", "accuracy", "faithfulness", "speed"]
Method = Literal["fusion", "flare"]


@dataclass(frozen=True)
class TuneConfig:
    method: Method
    uncertain_threshold: float | None = None
    flare_top_k: int | None = None
    answer_length: str = "medium"
    temperature: float = 0.0


def build_default_configs(
    thresholds: Iterable[float],
    methods: Iterable[Method] = ("fusion", "flare"),
    flare_top_ks: Iterable[int] = (2,),
    answer_lengths: Iterable[str] = ("medium",),
    temperatures: Iterable[float] = (0.0,),
) -> list[TuneConfig]:
    """Create tuning configs from the selected method and parameter ranges."""
    selected_methods = set(methods)
    configs: list[TuneConfig] = []

    if "fusion" in selected_methods:
        for answer_length, temperature in itertools.product(answer_lengths, temperatures):
            configs.append(
                TuneConfig(
                    method="fusion",
                    uncertain_threshold=None,
                    flare_top_k=None,
                    answer_length=answer_length,
                    temperature=float(temperature),
                )
            )

    if "flare" in selected_methods:
        for threshold, top_k, answer_length, temperature in itertools.product(
            thresholds,
            flare_top_ks,
            answer_lengths,
            temperatures,
        ):
            configs.append(
                TuneConfig(
                    method="flare",
                    uncertain_threshold=float(threshold),
                    flare_top_k=int(top_k),
                    answer_length=answer_length,
                    temperature=float(temperature),
                )
            )
    return configs


def score_result(row: dict, objective: Objective, max_latency: float) -> float:
    """
    Convert metrics into one ranking score.

    Higher is better.  Latency is normalized by the slowest config in the same
    tuning run, so a faster config gets a smaller penalty.
    """
    bertscore = float(row.get("bertscore_f1", 0.0))
    rouge = float(row.get("rougeL_f", 0.0))
    faithfulness = float(row.get("faithfulness", 0.0))
    latency = float(row.get("latency_s", 0.0))
    latency_penalty = latency / max_latency if max_latency > 0 else 0.0

    if objective == "accuracy":
        score = 0.55 * bertscore + 0.45 * rouge - 0.05 * latency_penalty
    elif objective == "faithfulness":
        score = 0.65 * faithfulness + 0.20 * bertscore + 0.15 * rouge - 0.05 * latency_penalty
    elif objective == "speed":
        score = 0.35 * bertscore + 0.25 * rouge + 0.20 * faithfulness - 0.45 * latency_penalty
    else:
        score = 0.35 * bertscore + 0.30 * rouge + 0.25 * faithfulness - 0.10 * latency_penalty

    return round(score, 4)


def reset_sample_outputs(samples: list) -> list:
    """Clear prediction and score fields on copied EvalSample objects."""
    for sample in samples:
        sample.pred_fusion = ""
        sample.pred_flare = ""
        sample.scores_fusion = {}
        sample.scores_flare = {}
        sample.latency_fusion = 0.0
        sample.latency_flare = 0.0
    return samples


def run_auto_tune(
    samples: list,
    provider: str,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7, 0.85),
    methods: Iterable[Method] = ("fusion", "flare"),
    flare_top_ks: Iterable[int] = (2,),
    answer_lengths: Iterable[str] = ("medium",),
    temperatures: Iterable[float] = (0.0,),
    objective: Objective = "balanced",
    delay: float = 0.5,
    delay_unit: str = "second",
    use_bertscore: bool = True,
    max_samples: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    """
    Run several inference configurations and return a ranked leaderboard.

    Args:
        samples: EvalSample list from eval_page.py.
        provider: LLM provider name, e.g. "google", "groq", "openai", "ollama".
        thresholds: FLARE confidence thresholds to test.
        methods: Which method families to run: fusion, flare, or both.
        flare_top_ks: Number of retrieved sentences for each uncertain FLARE sentence.
        answer_lengths: short, medium, or long answer prompt variants.
        temperatures: LLM sampling temperatures to test.
        objective: Ranking objective: balanced, accuracy, faithfulness, or speed.
        delay: Delay value passed to run_inference().
        delay_unit: second, token, or sentence.
        use_bertscore: Whether compute_scores() should calculate BERTScore.
        max_samples: Optional cap for quick tuning runs.
        progress_callback: Optional callback called before/after each config.

    Returns:
        A list of dictionaries sorted by auto_score descending.
    """
    if not samples:
        return []

    from eval_page import _agg, compute_scores, run_inference

    source_samples = list(samples[:max_samples]) if max_samples else list(samples)
    configs = build_default_configs(
        thresholds,
        methods=methods,
        flare_top_ks=flare_top_ks,
        answer_lengths=answer_lengths,
        temperatures=temperatures,
    )
    rows: list[dict] = []

    for index, config in enumerate(configs, start=1):
        if progress_callback:
            progress_callback({
                "event": "start",
                "index": index,
                "total": len(configs),
                "method": config.method,
                "threshold": config.uncertain_threshold,
                "flare_top_k": config.flare_top_k,
                "answer_length": config.answer_length,
                "temperature": config.temperature,
            })

        working = reset_sample_outputs(copy.deepcopy(source_samples))
        threshold = config.uncertain_threshold if config.uncertain_threshold is not None else 0.5
        top_k = config.flare_top_k if config.flare_top_k is not None else 2

        working = run_inference(
            working,
            config.method,
            provider,
            delay,
            delay_unit=delay_unit,
            uncertain_threshold=threshold,
            flare_top_k=top_k,
            answer_length=config.answer_length,
            temperature=config.temperature,
        )
        working = compute_scores(working, use_bertscore)
        agg = _agg(working, config.method)

        row = {
            "rank": 0,
            "method": config.method,
            "uncertain_threshold": config.uncertain_threshold,
            "flare_top_k": config.flare_top_k,
            "answer_length": config.answer_length,
            "temperature": config.temperature,
            "sample_count": len(working),
            "objective": objective,
            "bertscore_f1": agg.get("bertscore_f1", 0.0),
            "rougeL_f": agg.get("rougeL_f", 0.0),
            "faithfulness": agg.get("faithfulness", 0.0),
            "compression_ratio": agg.get("compression_ratio", 0.0),
            "latency_s": agg.get("latency_s", 0.0),
            "auto_score": 0.0,
        }
        rows.append(row)

        if progress_callback:
            progress_callback({
                "event": "done",
                "index": index,
                "total": len(configs),
                "method": config.method,
                "threshold": config.uncertain_threshold,
                "flare_top_k": config.flare_top_k,
                "answer_length": config.answer_length,
                "temperature": config.temperature,
                "row": row,
            })

    max_latency = max((float(row.get("latency_s", 0.0)) for row in rows), default=0.0)
    for row in rows:
        row["auto_score"] = score_result(row, objective, max_latency)

    rows.sort(key=lambda item: item["auto_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def best_config(leaderboard: list[dict]) -> dict | None:
    """Return the highest-ranked config from a run_auto_tune() leaderboard."""
    if not leaderboard:
        return None
    return min(leaderboard, key=lambda row: row.get("rank", 999999))
