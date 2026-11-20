# Index

analysis.md - run 17 findings and the cross-language probe, plain English.
INDEX.md - this file, one line per file.
data/HumanEval.jsonl.gz - original HumanEval archive from the OpenAI repository.
data/humaneval-py.jsonl - 164 HumanEval Python problems with canonical tests.
data/humaneval-js.jsonl - 161 MultiPL-E JavaScript HumanEval rows with tests.
data/humaneval-js.parquet - source parquet for the JavaScript rows.
data/humaneval-java.jsonl - 158 MultiPL-E Java HumanEval rows with tests.
data/humaneval-java.parquet - source parquet for the Java rows.
data/humaneval-cpp.jsonl - 161 MultiPL-E C++ HumanEval rows with tests.
data/humaneval-cpp.parquet - source parquet for the C++ rows.
data/manifest.json - the 20 sampled problems in all four languages.
results/run17-stats.json - recomputed run 17 configuration, curve, variance and retirement.
results/base-python.json - per-question base outcomes for Python.
results/base-js.json - per-question base outcomes for JavaScript.
results/base-java.json - per-question base outcomes for Java.
results/base-cpp.json - per-question base outcomes for C++.
results/adapter-python.json - per-question adapter outcomes for Python.
results/adapter-js.json - per-question adapter outcomes for JavaScript.
results/adapter-java.json - per-question adapter outcomes for Java.
results/adapter-cpp.json - per-question adapter outcomes for C++.
results/benchmark-summary.json - pass counts and paired per-question matrix.
scripts/download_data.py - fetches HumanEval and MultiPL-E files, converts parquet.
scripts/select_problems.py - samples the 20 shared problems with seed 2026.
scripts/analyse_run17.py - streams the run 17 files and writes run17-stats.json.
scripts/generate.py - loads Qwen3 and the adapter, writes greedy completions.
scripts/grade.py - compiles and runs answers against shipped tests.
scripts/summarize_benchmark.py - combines grading files into the summary.
