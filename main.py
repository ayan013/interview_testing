import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt import SYSTEM_PROMPT, build_prompt


# ============================================================
# CONFIGURATION
# ============================================================

# Change this value when testing different models.
#
# GPT-4o mini:
#MODEL = "gpt-4o-mini"

# GPT-5:
MODEL = "gpt-5"

# If you have a different GPT-5 model available in your account,
# put its exact model ID here.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

CONCEPT_FILE = BASE_DIR / "concept.json"
CANDIDATE_FILE = BASE_DIR / "candidate_answer.json"
OUTPUT_FILE = BASE_DIR / f"evidence{MODEL}.json"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(api_key=api_key)


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# LLM CALL
# ============================================================

def extract_evidence(concept: dict, candidate_answer: str):
    """
    Sends the concept and candidate answer to the LLM
    and returns structured evidence.
    """

    user_prompt = build_prompt(
        concept,
        candidate_answer
    )

    response = client.responses.create(
        model=MODEL,

        instructions=SYSTEM_PROMPT,

        input=user_prompt
    )

    raw_output = response.output_text

    print("\n================ RAW LLM OUTPUT ================\n")
    print(raw_output)

    print("\n=================================================\n")

    try:
        evidence = json.loads(raw_output)
    except json.JSONDecodeError as error:
        print("LLM did not return valid JSON.")
        print("JSON error:", error)

        raise

    return evidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print("Phase 1 - Interview Evidence Extraction")
    print("==============================================")
    print(f"Model: {MODEL}")

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    concept = load_json(
        CONCEPT_FILE
    )

    candidate_data = load_json(
        CANDIDATE_FILE
    )

    answers = candidate_data["answers"]

    print(
        f"Loaded {len(answers)} candidate answers."
    )

    # --------------------------------------------------------
    # Test every answer
    # --------------------------------------------------------

    results = []

    for answer in answers:

        print("\n----------------------------------------------")
        print(
            f"Answer ID: {answer['answer_id']}"
        )
        print(
            f"Expected label: {answer['label']}"
        )

        print("\nCandidate:")
        print(answer["text"])

        # ----------------------------------------------------
        # LLM evidence extraction
        # ----------------------------------------------------

        evidence = extract_evidence(
            concept,
            answer["text"]
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "candidate_id": candidate_data["candidate_id"],
            "answer_id": answer["answer_id"],
            "expected_label": answer["label"],
            "candidate_answer": answer["text"],
            "model": MODEL,
            "evidence": evidence
        }

        results.append(result)

    # --------------------------------------------------------
    # Save all evidence
    # --------------------------------------------------------

    output = {
        "concept_id": concept["concept_id"],
        "model": MODEL,
        "results": results
    }

    save_json(
        OUTPUT_FILE,
        output
    )

    print("\n==============================================")
    print("Completed.")
    print(f"Evidence saved to: {OUTPUT_FILE}")
    print("==============================================")


if __name__ == "__main__":
    main()