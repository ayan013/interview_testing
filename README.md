# Linux Phase-1 Benchmark

Files:
- concept.json: evaluator-visible concept specification
- answers.json: evaluator-visible answer IDs/text only
- gold_standard.json: hidden ground truth
- benchmark_metadata.json: hidden adversarial category/test objective
- prompt.py: evaluator prompt
- main.py: benchmark runner
- compare.py: metrics calculator

Run:
1. Put OPENAI_API_KEY in .env
2. Set model, e.g.:
   OPENAI_MODEL=gpt-4o-mini python main.py
   OPENAI_MODEL=gpt-5 python main.py
3. Compare:
   python compare.py evidence_gpt-4o-mini.json evidence_gpt-5.json

Never send gold_standard.json or benchmark_metadata.json to the evaluator model.
