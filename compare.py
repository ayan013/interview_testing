import json


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


gold = load_json("gold_standard.json")
gpt4 = load_json("evidence_gpt4.json")
gpt5 = load_json("evidencegpt-5.json")


def get_predictions(data):
    predictions = {}

    for result in data["results"]:
        answer_id = result["answer_id"]

        evaluations = result["evidence"]["evaluations"]

        predictions[answer_id] = {
            item["evaluation_concept_id"]: item["depth"]
            for item in evaluations
        }

    return predictions


gpt4_predictions = get_predictions(gpt4)
gpt5_predictions = get_predictions(gpt5)


def evaluate_model(predictions, gold_data):

    total = 0
    correct = 0
    absolute_error = 0

    false_positives = 0
    actual_negative = 0

    for answer in gold_data["answers"]:

        answer_id = answer["answer_id"]
        expected = answer["expected"]
        predicted = predictions[answer_id]

        for concept_id, expected_depth in expected.items():

            predicted_depth = predicted.get(
                concept_id,
                0
            )

            total += 1

            # Exact depth accuracy
            if predicted_depth == expected_depth:
                correct += 1

            # Mean Absolute Error
            absolute_error += abs(
                predicted_depth - expected_depth
            )

            # False positive:
            # Model says demonstrated (>0)
            # but gold says not demonstrated (0)
            if expected_depth == 0:
                actual_negative += 1

                if predicted_depth > 0:
                    false_positives += 1

    return {
        "depth_accuracy": correct / total,
        "depth_mae": absolute_error / total,
        "false_positive_rate": (
            false_positives / actual_negative
            if actual_negative
            else 0
        )
    }


gpt4_metrics = evaluate_model(
    gpt4_predictions,
    gold
)

gpt5_metrics = evaluate_model(
    gpt5_predictions,
    gold
)


print("\nGPT-4o-mini")
print("----------------")
for key, value in gpt4_metrics.items():
    print(f"{key}: {value:.3f}")


print("\nGPT-5")
print("----------------")
for key, value in gpt5_metrics.items():
    print(f"{key}: {value:.3f}")