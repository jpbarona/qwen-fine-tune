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
