import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def get_predictions(data):
    predictions = {}
    incorrect = {}
    format_stats = {
        "total_results": 0,
        "strict_json": 0,
        "recovered_json": 0,
        "unknown_format": 0,
    }

    for result in data["results"]:
        answer_id = result["answer_id"]
        evidence = result["evidence"]
        evaluations = evidence["evaluations"]

        predictions[answer_id] = {
            item["evaluation_concept_id"]: item["depth"]
            for item in evaluations
        }

        incorrect[answer_id] = {
            item["evaluation_concept_id"]
            for item in evaluations
            if item.get("incorrect_claims")
        }

        format_stats["total_results"] += 1

        output_format = (
            evidence
            .get("_metadata", {})
            .get("output_format")
        )

        if output_format is None:
            format_stats["unknown_format"] += 1
        elif output_format.get("strict_json") is True:
            format_stats["strict_json"] += 1
        elif output_format.get("recovered_json") is True:
            format_stats["recovered_json"] += 1
        else:
            format_stats["unknown_format"] += 1

    return predictions, incorrect, format_stats


def calculate_metrics(answer_ids, predictions, incorrect_predictions, gold_by_id):
    total = correct = within_one = 0
    absolute_error = 0

    tp = fp = tn = fn = 0
    misconception_tp = misconception_fp = misconception_fn = 0

    for answer_id in answer_ids:
        expected = gold_by_id[answer_id]["expected"]
        predicted = predictions.get(answer_id, {})

        gold_incorrect = set(
            gold_by_id[answer_id].get("expected_incorrect_concepts", [])
        )
        pred_incorrect = incorrect_predictions.get(answer_id, set())

        misconception_tp += len(gold_incorrect & pred_incorrect)
        misconception_fp += len(pred_incorrect - gold_incorrect)
        misconception_fn += len(gold_incorrect - pred_incorrect)

        for concept_id, expected_depth in expected.items():
            predicted_depth = predicted.get(concept_id, 0)

            total += 1

            if predicted_depth == expected_depth:
                correct += 1

            if abs(predicted_depth - expected_depth) <= 1:
                within_one += 1

            absolute_error += abs(predicted_depth - expected_depth)

            expected_positive = expected_depth > 0
            predicted_positive = predicted_depth > 0

            if expected_positive and predicted_positive:
                tp += 1
            elif not expected_positive and predicted_positive:
                fp += 1
            elif not expected_positive and not predicted_positive:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0

    m_precision = (
        misconception_tp / (misconception_tp + misconception_fp)
        if (misconception_tp + misconception_fp)
        else 0
    )
    m_recall = (
        misconception_tp / (misconception_tp + misconception_fn)
        if (misconception_tp + misconception_fn)
        else 0
    )
    m_f1 = (
        2 * m_precision * m_recall / (m_precision + m_recall)
        if (m_precision + m_recall)
        else 0
    )

    return {
        "depth_accuracy": correct / total if total else 0,
        "depth_within_1_accuracy": within_one / total if total else 0,
        "depth_mae": absolute_error / total if total else 0,
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "misconception_precision": m_precision,
        "misconception_recall": m_recall,
        "misconception_f1": m_f1,
        "n_answers": len(answer_ids),
    }


def calculate_format_metrics(format_stats):
    total = format_stats["total_results"]

    if total == 0:
        return {
            "strict_json_rate": 0.0,
            "recovered_json_rate": 0.0,
            "unknown_format_rate": 0.0,
        }

    return {
        "strict_json_rate": format_stats["strict_json"] / total,
        "recovered_json_rate": format_stats["recovered_json"] / total,
        "unknown_format_rate": format_stats["unknown_format"] / total,
    }


def print_metric_block(metrics):
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="gold_standard.json")
    parser.add_argument("--metadata", default="benchmark_metadata.json")
    parser.add_argument("prediction_files", nargs="+")
    args = parser.parse_args()

    gold = load_json(args.gold)
    metadata = load_json(args.metadata)

    gold_by_id = {
        item["answer_id"]: item
        for item in gold["answers"]
    }

    category_by_id = {
        item["answer_id"]: item["category"]
        for item in metadata["cases"]
    }

    all_answer_ids = list(gold_by_id)

    for prediction_file in args.prediction_files:
        data = load_json(prediction_file)

        predictions, incorrect_predictions, format_stats = get_predictions(data)

        print("\n" + "=" * 72)
        print(f"MODEL: {data.get('model', Path(prediction_file).stem)}")
        print("=" * 72)

        overall = calculate_metrics(
            all_answer_ids,
            predictions,
            incorrect_predictions,
            gold_by_id,
        )

        print("\nSEMANTIC / DEPTH METRICS")
        print("-" * 72)
        print_metric_block(overall)

        print("\nOUTPUT FORMAT METRICS")
        print("-" * 72)
        format_metrics = calculate_format_metrics(format_stats)
        print_metric_block(format_metrics)

        if format_stats["unknown_format"] > 0:
            print(
                "\nNote: unknown_format_rate includes evidence files generated "
                "before output-format metadata was added."
            )

        grouped = defaultdict(list)

        for answer_id in all_answer_ids:
            category = category_by_id.get(answer_id, "uncategorized")
            grouped[category].append(answer_id)

        print("\nBY BENCHMARK CATEGORY")
        print("-" * 72)

        for category in sorted(grouped):
            metrics = calculate_metrics(
                grouped[category],
                predictions,
                incorrect_predictions,
                gold_by_id,
            )

            print(
                f"{category:32} "
                f"depth_acc={metrics['depth_accuracy']:.3f}  "
                f"precision={metrics['evidence_precision']:.3f}  "
                f"recall={metrics['evidence_recall']:.3f}  "
                f"FPR={metrics['false_positive_rate']:.3f}  "
                f"FNR={metrics['false_negative_rate']:.3f}  "
                f"misconception_f1={metrics['misconception_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
