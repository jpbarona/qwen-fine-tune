# MBPP data

`mbpp_full.jsonl` is an offline copy of the MBPP `full` configuration from
`google-research-datasets/mbpp` on Hugging Face.

- Source: https://huggingface.co/datasets/google-research-datasets/mbpp
- Config: `full`
- Splits merged: train (374), validation (90), test (500)
- Total: 964 rows
- Excluded: the 10 few-shot examples in the `prompt` split
- Downloaded: 2026-11-12

Each line has `task_id`, `text`, `test_list`, `test_setup_code`, and
`source_split`. The reference `code` column is not kept because nothing needs
it. The loader reads `data/` from the repository, so pytest needs no network.

The split is defined in `src/rlvr/data.py`. Seed 1234 gives 714 train and 250
test. Task ids are sorted before the shuffle, so file order cannot change the
result.

## Difficulty labels

`difficulty_labels.jsonl` holds one row per training question:

    {"prompt_id": "mbpp-464", "band": "medium"}

The rows are in the same order as the 714 row training split for seed 1234.
Bands are `very easy`, `medium`, `hard`, and `very hard`. They were assigned by
reading the problem statements in batches of 20, judging the number of concepts,
edge cases, and steps a correct solution needs. The reference `code` column was
not available, so text is the only signal.

`load_difficulty_labels` in `src/rlvr/data.py` checks the file against a split.
It fails on a missing id, an unknown id, a duplicate id, an unknown band, and a
row out of split order.

## Synthetic easy set

`synthetic_easy.jsonl` holds 50 very easy questions for the first curriculum
stage. Each row has `task_id`, `text`, `code`, `test_list`, and
`test_setup_code`, the MBPP row shape plus the reference solution.

- Task ids run from 10001 to 10050, above the MBPP range of 1 to 974.
- Prompt ids have the form `synthetic-10001`.
- Every reference solution is at most 19 lines and uses the standard library
  only.
- Every question has at least two tests.

`load_synthetic_questions` in `src/rlvr/synthetic.py` enforces the id range, the
line limit, and the two test minimum. `tests/test_difficulty.py` runs every
reference solution against its own tests with plain Python, so the suite needs
no Docker. The same 50 solutions were also checked once in `DockerTestRunner`
against `python:3.12-slim`.

