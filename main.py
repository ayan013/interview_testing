import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt import SYSTEM_PROMPT, build_prompt


# ============================================================
# CONFIGURATION
# ============================================================

# Change for each benchmark run, or set OPENAI_MODEL in .env.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

BASE_DIR = Path(__file__).resolve().parent
CONCEPT_FILE = BASE_DIR / "concept.json"
CANDIDATE_FILE = BASE_DIR / "answers.json"

safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL)
OUTPUT_FILE = BASE_DIR / f"evidence_{safe_model_name}.json"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

client = OpenAI(api_key=api_key)


def load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_model_json(raw_output: str):
    """
    Parse the model response while preserving whether it obeyed
    the strict JSON-only output contract.

    Returns:
        (parsed_dict, parse_metadata)

    parse_metadata:
        strict_json: True only if json.loads(raw_output) worked directly.
        recovered_json: True if harmless wrapper text/fences were removed.
        recovery_method: None, "markdown_fence", or "json_object_extraction".
    """
    if raw_output is None:
        raise ValueError("Model returned None.")

    text = raw_output.strip()

    if not text:
        raise ValueError("Model returned empty output.")

    # --------------------------------------------------------
    # Attempt 1: strict JSON exactly as requested
    # --------------------------------------------------------
    try:
        parsed = json.loads(text)
        return parsed, {
            "strict_json": True,
            "recovered_json": False,
            "recovery_method": None,
        }
    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 2: remove Markdown JSON/code fences only
    # --------------------------------------------------------
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    if cleaned != text:
        try:
            parsed = json.loads(cleaned)
            return parsed, {
                "strict_json": False,
                "recovered_json": True,
                "recovery_method": "markdown_fence",
            }
        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # Attempt 3: extract the outer JSON object if the model
    # added explanatory text before/after it.
    #
    # We intentionally DO NOT repair malformed JSON syntax.
    # Missing commas/quotes/braces remain model failures.
    # --------------------------------------------------------
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_candidate = cleaned[start:end + 1]

        try:
            parsed = json.loads(json_candidate)
            return parsed, {
                "strict_json": False,
                "recovered_json": True,
                "recovery_method": "json_object_extraction",
            }
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Model returned JSON-like output, but the JSON itself is malformed: "
                f"{error}\n\nRAW OUTPUT:\n{raw_output}"
            ) from error

    raise RuntimeError(
        "Model output did not contain a recoverable JSON object.\n\n"
        f"RAW OUTPUT:\n{raw_output}"
    )


def extract_evidence(concept: dict, candidate_answer: str):
    user_prompt = build_prompt(concept, candidate_answer)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    raw_output = response.output_text

    try:
        evidence, parse_metadata = parse_model_json(raw_output)
    except Exception as error:
        print("\n==============================================")
        print("UNRECOVERABLE MODEL OUTPUT")
        print("==============================================")
        print(raw_output)
        print("==============================================\n")
        raise RuntimeError(f"Unable to parse model output: {error}") from error

    # Keep parsing/compliance information separate from semantic evaluations.
    evidence["_metadata"] = {
        "output_format": parse_metadata
    }

    return evidence


def validate_evidence(evidence: dict, concept: dict):
    """
    Ensure every expected concept appears exactly once.

    Missing concepts are inserted as depth=0 and marked as an output error.

    Expected concept order follows concept.json so benchmark outputs remain
    deterministic and easy to diff across models.
    """
    expected_ids = [
        item["id"] for item in concept["evaluation_concepts"]
    ]

    evaluations = evidence.get("evaluations", [])

    if not isinstance(evaluations, list):
        evaluations = []

    existing = {}

    for item in evaluations:
        if not isinstance(item, dict):
            continue

        concept_id = item.get("evaluation_concept_id")

        # Keep only expected IDs. If the model duplicates an ID, preserve
        # the first occurrence rather than silently changing its semantics.
        if concept_id in expected_ids and concept_id not in existing:
            existing[concept_id] = item

    repaired = []

    for concept_id in expected_ids:
        if concept_id in existing:
            repaired.append(existing[concept_id])
        else:
            repaired.append({
                "evaluation_concept_id": concept_id,
                "depth": 0,
                "confidence": 0.0,
                "evidence": None,
                "incorrect_claims": [],
                "output_error": "concept_missing_from_model_output",
            })

    evidence["evaluations"] = repaired
    return evidence


def main():
    concept = load_json(CONCEPT_FILE)
    candidate_data = load_json(CANDIDATE_FILE)
    answers = candidate_data["answers"]

    print("==============================================")
    print("Phase 1 - Interview Evidence Extraction")
    print(f"Model: {MODEL}")
    print(f"Loaded {len(answers)} candidate answers.")
    print("==============================================")

    results = []

    strict_json_count = 0
    recovered_json_count = 0

    for answer in answers:
        print(f"\nAnswer ID: {answer['answer_id']}")
        print("Candidate:", answer["text"])

        # IMPORTANT:
        # Only the concept definition and answer text are passed to the model.
        # No category, gold label, expected depth, or benchmark metadata is sent.
        evidence = extract_evidence(concept, answer["text"])
        evidence = validate_evidence(evidence, concept)

        output_format = evidence.get("_metadata", {}).get("output_format", {})

        if output_format.get("strict_json") is True:
            strict_json_count += 1

        if output_format.get("recovered_json") is True:
            recovered_json_count += 1

        results.append({
            "candidate_id": candidate_data["candidate_id"],
            "answer_id": answer["answer_id"],
            "candidate_answer": answer["text"],
            "model": MODEL,
            "evidence": evidence,
        })

    total_answers = len(answers)

    output = {
        "concept_id": concept["concept_id"],
        "model": MODEL,
        "run_metrics": {
            "total_answers": total_answers,
            "strict_json_count": strict_json_count,
            "strict_json_rate": (
                strict_json_count / total_answers if total_answers else 0.0
            ),
            "recovered_json_count": recovered_json_count,
            "recovered_json_rate": (
                recovered_json_count / total_answers if total_answers else 0.0
            ),
        },
        "results": results,
    }

    save_json(OUTPUT_FILE, output)

    print("\n==============================================")
    print("Run output-format metrics")
    print("==============================================")
    print(f"Strict JSON:    {strict_json_count}/{total_answers}")
    print(f"Recovered JSON: {recovered_json_count}/{total_answers}")
    print(f"\nCompleted. Evidence saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
