import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt import SYSTEM_PROMPT, build_prompt


# ============================================================
# PRICING
# ============================================================
# USD per 1,000,000 text tokens.
#
# Verified against OpenAI model documentation when this file
# was created. Pricing can change, so update this table before
# relying on it for financial projections.
#
# input  = non-cached input tokens
# cached = cached input tokens
# output = output tokens
# ============================================================

MODEL_PRICING_USD_PER_1M = {
    "gpt-4o-mini": {
        "input": 0.15,
        "cached": 0.075,
        "output": 0.60,
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached": 0.10,
        "output": 1.60,
    },
    "gpt-5-nano": {
        "input": 0.05,
        "cached": 0.005,
        "output": 0.40,
    },
    "gpt-5-mini": {
        "input": 0.25,
        "cached": 0.025,
        "output": 2.00,
    },
    "gpt-5": {
        "input": 1.25,
        "cached": 0.125,
        "output": 10.00,
    },
}


DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]


BASE_DIR = Path(__file__).resolve().parent
CONCEPT_FILE = BASE_DIR / "concept.json"
ANSWERS_FILE = BASE_DIR / "answers.json"
RESULTS_DIR = BASE_DIR / "benchmark_results"


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def percentile(values, percentile_value):
    """
    Simple percentile calculation without numpy.
    Uses linear interpolation between neighboring values.
    """
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def get_cached_input_tokens(usage) -> int:
    """
    Responses API exposes cached tokens inside input_tokens_details.
    This helper is defensive because SDK object shapes can vary by version.
    """
    details = getattr(usage, "input_tokens_details", None)

    if details is None:
        return 0

    cached = getattr(details, "cached_tokens", None)

    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens", 0)

    return int(cached or 0)


def get_reasoning_tokens(usage) -> int:
    """
    Reasoning tokens are normally included in output_tokens for billing.
    We record them separately for diagnostics only.
    """
    details = getattr(usage, "output_tokens_details", None)

    if details is None:
        return 0

    reasoning = getattr(details, "reasoning_tokens", None)

    if reasoning is None and isinstance(details, dict):
        reasoning = details.get("reasoning_tokens", 0)

    return int(reasoning or 0)


def calculate_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
):
    pricing = MODEL_PRICING_USD_PER_1M.get(model)

    if pricing is None:
        return None

    # input_tokens includes cached input tokens, so subtract cached
    # tokens before applying the normal input rate.
    uncached_input_tokens = max(
        input_tokens - cached_input_tokens,
        0,
    )

    input_cost = (
        uncached_input_tokens / 1_000_000
    ) * pricing["input"]

    cached_input_cost = (
        cached_input_tokens / 1_000_000
    ) * pricing["cached"]

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing["output"]

    total_cost = input_cost + cached_input_cost + output_cost

    return {
        "uncached_input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


# ============================================================
# OUTPUT VALIDATION / REPAIR
# ============================================================

def validate_evidence(evidence: dict, concept: dict):
    expected_ids = [
        c["id"]
        for c in concept["evaluation_concepts"]
    ]

    evaluations = evidence.get("evaluations", [])

    existing = {
        item.get("evaluation_concept_id"): item
        for item in evaluations
        if item.get("evaluation_concept_id")
    }

    repaired = []
    missing_concepts = []

    for concept_id in expected_ids:
        if concept_id in existing:
            repaired.append(existing[concept_id])
        else:
            missing_concepts.append(concept_id)
            repaired.append({
                "evaluation_concept_id": concept_id,
                "depth": 0,
                "confidence": 0.0,
                "evidence": None,
                "incorrect_claims": [],
                "output_error": "concept_missing_from_model_output",
            })

    evidence["evaluations"] = repaired
    evidence["missing_concepts"] = missing_concepts
    evidence["contract_valid"] = len(missing_concepts) == 0

    return evidence


# ============================================================
# ONE MODEL CALL
# ============================================================

def evaluate_answer(client, model, concept, answer_text):
    prompt = build_prompt(
        concept,
        answer_text,
    )

    start = time.perf_counter()

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )

    latency_seconds = time.perf_counter() - start

    raw_output = response.output_text.strip()

    try:
        evidence = json.loads(raw_output)
        strict_json = True

    except json.JSONDecodeError:
        # Keep the raw output for debugging and report the failure.
        return {
            "success": False,
            "latency_seconds": latency_seconds,
            "strict_json": False,
            "raw_output": raw_output,
            "error": "invalid_json",
            "usage": None,
            "cost": None,
            "evidence": None,
        }

    usage = response.usage

    input_tokens = int(
        getattr(usage, "input_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", 0) or 0
    )
    total_tokens = int(
        getattr(usage, "total_tokens", 0) or
        (input_tokens + output_tokens)
    )

    cached_input_tokens = get_cached_input_tokens(usage)
    reasoning_tokens = get_reasoning_tokens(usage)

    cost = calculate_cost(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )

    evidence = validate_evidence(
        evidence,
        concept,
    )

    return {
        "success": True,
        "latency_seconds": latency_seconds,
        "strict_json": strict_json,
        "raw_output": None,
        "error": None,
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": max(
                input_tokens - cached_input_tokens,
                0,
            ),
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        },
        "cost": cost,
        "evidence": evidence,
    }


# ============================================================
# RUN ONE MODEL OVER THE FULL BENCHMARK
# ============================================================

def benchmark_model(
    client,
    model,
    concept,
    candidate_data,
):
    answers = candidate_data["answers"]

    print("\n" + "=" * 72)
    print(f"MODEL: {model}")
    print("=" * 72)

    results = []

    benchmark_start = time.perf_counter()

    for index, answer in enumerate(answers, start=1):
        answer_id = answer["answer_id"]

        print(
            f"[{index:02d}/{len(answers):02d}] "
            f"{answer_id} ... ",
            end="",
            flush=True,
        )

        result = evaluate_answer(
            client=client,
            model=model,
            concept=concept,
            answer_text=answer["text"],
        )

        if result["success"]:
            cost_value = (
                result["cost"]["total_cost_usd"]
                if result["cost"]
                else None
            )

            cost_text = (
                f"${cost_value:.6f}"
                if cost_value is not None
                else "unknown"
            )

            print(
                f"{result['latency_seconds']:.2f}s | "
                f"{result['usage']['input_tokens']} in | "
                f"{result['usage']['output_tokens']} out | "
                f"{cost_text}"
            )
        else:
            print(
                f"FAILED after "
                f"{result['latency_seconds']:.2f}s "
                f"({result['error']})"
            )

        stored_result = {
            "candidate_id": candidate_data.get("candidate_id"),
            "answer_id": answer_id,
            "candidate_answer": answer["text"],
            "model": model,
            "latency_seconds": round(
                result["latency_seconds"],
                4,
            ),
            "strict_json": result["strict_json"],
            "usage": result["usage"],
            "estimated_cost_usd": (
                result["cost"]["total_cost_usd"]
                if result["cost"]
                else None
            ),
            "cost_breakdown": result["cost"],
            "error": result["error"],
            "raw_output_on_error": result["raw_output"],
            "evidence": result["evidence"],
        }

        results.append(stored_result)

    benchmark_wall_seconds = (
        time.perf_counter() - benchmark_start
    )

    return results, benchmark_wall_seconds


# ============================================================
# SUMMARIZE ONE MODEL
# ============================================================

def summarize_model(
    model,
    results,
    benchmark_wall_seconds,
):
    successful = [
        r for r in results
        if r["error"] is None
    ]

    latencies = [
        r["latency_seconds"]
        for r in successful
    ]

    total_input_tokens = sum(
        r["usage"]["input_tokens"]
        for r in successful
        if r["usage"]
    )

    total_cached_input_tokens = sum(
        r["usage"]["cached_input_tokens"]
        for r in successful
        if r["usage"]
    )

    total_output_tokens = sum(
        r["usage"]["output_tokens"]
        for r in successful
        if r["usage"]
    )

    total_reasoning_tokens = sum(
        r["usage"]["reasoning_tokens"]
        for r in successful
        if r["usage"]
    )

    costs = [
        r["estimated_cost_usd"]
        for r in successful
        if r["estimated_cost_usd"] is not None
    ]

    total_cost = (
        sum(costs)
        if costs
        else None
    )

    missing_concept_count = sum(
        len(
            r["evidence"].get(
                "missing_concepts",
                [],
            )
        )
        for r in successful
        if r["evidence"]
    )

    contract_successes = sum(
        1
        for r in successful
        if r["evidence"] and
        r["evidence"].get(
            "contract_valid",
            False,
        )
    )

    return {
        "model": model,
        "answers_attempted": len(results),
        "answers_successful": len(successful),
        "strict_json_rate": (
            len(successful) / len(results)
            if results
            else 0
        ),
        "schema_contract_success_rate": (
            contract_successes / len(successful)
            if successful
            else 0
        ),
        "missing_concept_count": missing_concept_count,

        "total_benchmark_wall_seconds": round(
            benchmark_wall_seconds,
            4,
        ),
        "average_latency_seconds": round(
            statistics.mean(latencies),
            4,
        ) if latencies else 0,
        "median_latency_seconds": round(
            statistics.median(latencies),
            4,
        ) if latencies else 0,
        "p95_latency_seconds": round(
            percentile(latencies, 95),
            4,
        ) if latencies else 0,
        "min_latency_seconds": round(
            min(latencies),
            4,
        ) if latencies else 0,
        "max_latency_seconds": round(
            max(latencies),
            4,
        ) if latencies else 0,

        "total_input_tokens": total_input_tokens,
        "total_cached_input_tokens": total_cached_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,

        "estimated_total_cost_usd": (
            round(total_cost, 8)
            if total_cost is not None
            else None
        ),
        "estimated_average_cost_per_answer_usd": (
            round(
                total_cost / len(successful),
                8,
            )
            if total_cost is not None and successful
            else None
        ),

        "pricing_used_usd_per_1m": (
            MODEL_PRICING_USD_PER_1M.get(model)
        ),
    }


# ============================================================
# SAVE CSV SUMMARY
# ============================================================

def save_summary_csv(path: Path, summaries):
    fields = [
        "model",
        "answers_attempted",
        "answers_successful",
        "strict_json_rate",
        "schema_contract_success_rate",
        "missing_concept_count",
        "total_benchmark_wall_seconds",
        "average_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "min_latency_seconds",
        "max_latency_seconds",
        "total_input_tokens",
        "total_cached_input_tokens",
        "total_output_tokens",
        "total_reasoning_tokens",
        "estimated_total_cost_usd",
        "estimated_average_cost_per_answer_usd",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow({
                field: summary.get(field)
                for field in fields
            })


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase-1 evidence benchmark across "
            "multiple OpenAI models and measure latency, "
            "token usage, output-contract reliability, "
            "and estimated API cost."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model IDs to test. Example: "
            "--models gpt-4o-mini gpt-4.1-mini gpt-5"
        ),
    )

    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    concept = load_json(CONCEPT_FILE)
    candidate_data = load_json(ANSWERS_FILE)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = OpenAI(
        api_key=api_key,
    )

    summaries = []

    for model in args.models:
        results, wall_seconds = benchmark_model(
            client=client,
            model=model,
            concept=concept,
            candidate_data=candidate_data,
        )

        summary = summarize_model(
            model=model,
            results=results,
            benchmark_wall_seconds=wall_seconds,
        )

        summaries.append(summary)

        evidence_file = (
            RESULTS_DIR /
            f"evidence_{safe_model_name(model)}.json"
        )

        save_json(
            evidence_file,
            {
                "concept_id": concept["concept_id"],
                "model": model,
                "results": results,
                "performance_summary": summary,
            },
        )

        print("\nMODEL SUMMARY")
        print("-" * 72)
        print(
            f"Total benchmark time: "
            f"{summary['total_benchmark_wall_seconds']:.2f}s"
        )
        print(
            f"Average latency: "
            f"{summary['average_latency_seconds']:.2f}s"
        )
        print(
            f"Median latency: "
            f"{summary['median_latency_seconds']:.2f}s"
        )
        print(
            f"P95 latency: "
            f"{summary['p95_latency_seconds']:.2f}s"
        )
        print(
            f"Input tokens: "
            f"{summary['total_input_tokens']}"
        )
        print(
            f"Cached input tokens: "
            f"{summary['total_cached_input_tokens']}"
        )
        print(
            f"Output tokens: "
            f"{summary['total_output_tokens']}"
        )
        print(
            f"Reasoning tokens: "
            f"{summary['total_reasoning_tokens']}"
        )

        total_cost = summary[
            "estimated_total_cost_usd"
        ]

        if total_cost is not None:
            print(
                f"Estimated total cost: "
                f"${total_cost:.6f}"
            )
            print(
                f"Average cost / answer: "
                f"${summary['estimated_average_cost_per_answer_usd']:.6f}"
            )
        else:
            print(
                "Estimated cost: unavailable "
                "(model missing from pricing table)"
            )

    summary_json = (
        RESULTS_DIR /
        "model_performance_summary.json"
    )

    summary_csv = (
        RESULTS_DIR /
        "model_performance_summary.csv"
    )

    save_json(
        summary_json,
        {
            "pricing_note": (
                "Estimated text-token cost based on the "
                "pricing table embedded in benchmark_models.py. "
                "Update pricing before financial projections."
            ),
            "models": summaries,
        },
    )

    save_summary_csv(
        summary_csv,
        summaries,
    )

    print("\n" + "=" * 72)
    print("ALL MODELS COMPLETE")
    print("=" * 72)
    print(f"JSON summary: {summary_json}")
    print(f"CSV summary:  {summary_csv}")


if __name__ == "__main__":
    main()
