"""Recompute run 17 statistics from the three files in the run folder.

Streams rollouts.jsonl one line at a time. Writes results/run17-stats.json.
Only reads the run folder, never writes to it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/home/opencode/Work/rl-grpo-data/2026-11-16-run17-step313")
OUT = Path("/home/opencode/Work/qwen-finetune-analysis/results/run17-stats.json")

STAGES = ["very easy", "medium", "hard", "very hard"]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[index]


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text())
    eval_payload = json.loads((DATA / "eval.json").read_text())

    metric_rows = list(read_jsonl(DATA / "metrics.jsonl"))

    prompts = [row["prompts"] for row in metric_rows]
    sequences = [row["sequences"] for row in metric_rows]

    by_stage_metrics: dict[str, dict] = {}
    for stage in STAGES:
        rows = [row for row in metric_rows if row["stage"] == stage]
        if not rows:
            continue
        by_stage_metrics[stage] = {
            "steps": len(rows),
            "mean_reward": sum(r["mean_reward"] for r in rows) / len(rows),
            "mean_pass_rate": sum(r["pass_rate"] for r in rows) / len(rows),
            "mean_completion_tokens": sum(r["mean_completion_tokens"] for r in rows)
            / len(rows),
            "mean_truncation_rate": sum(r["truncation_rate"] for r in rows)
            / len(rows),
        }

    # One pass over the 350 MB rollout file.
    rewards_by_group: dict[tuple[int, str], list[float]] = defaultdict(list)
    bodies_by_group: dict[tuple[int, str], list[str]] = defaultdict(list)
    tests_by_group: dict[tuple[int, str], list[int]] = defaultdict(list)
    stage_by_prompt: dict[str, str] = {}
    stage_of_group: dict[tuple[int, str], str] = {}
    error_by_stage: dict[str, Counter] = defaultdict(Counter)
    first_step: dict[str, int] = {}
    last_step: dict[str, int] = {}
    prompt_rewards: dict[str, list[float]] = defaultdict(list)
    total_rows = 0

    for row in read_jsonl(DATA / "rollouts.jsonl"):
        total_rows += 1
        step = int(row["step"])
        prompt_id = row["prompt_id"]
        stage = row["stage"]
        key = (step, prompt_id)
        rewards_by_group[key].append(float(row["reward"]))
        bodies_by_group[key].append((row.get("code") or "").strip())
        tests_by_group[key].append(int(row.get("tests_passed") or 0))
        stage_of_group[key] = stage
        error_by_stage[stage][row["error_class"]] += 1
        prompt_rewards[prompt_id].append(float(row["reward"]))
        if prompt_id not in first_step or step < first_step[prompt_id]:
            first_step[prompt_id] = step
            stage_by_prompt[prompt_id] = stage
        last_step[prompt_id] = max(last_step.get(prompt_id, step), step)

    group_count = len(rewards_by_group)
    zero_variance = 0
    all_zero_reward = 0
    all_one_reward = 0
    all_failed_tests = 0
    duplicate_groups = 0
    distinct_bodies = 0
    sizes = Counter(len(v) for v in rewards_by_group.values())
    for key, rewards in rewards_by_group.items():
        if min(rewards) == max(rewards):
            zero_variance += 1
            if min(rewards) == 0.0:
                all_zero_reward += 1
            elif min(rewards) == 1.0:
                all_one_reward += 1
        bodies = bodies_by_group[key]
        distinct = len(set(bodies))
        distinct_bodies += distinct
        if distinct < len(bodies):
            duplicate_groups += 1

    # The pipeline removes a prompt when every rollout in one group passes.
    all_failed_tests = sum(
        1
        for key in tests_by_group
        if len(tests_by_group[key]) == 8 and all(v == 0 for v in tests_by_group[key])
    )
    mastered = [
        (key[1], stage_of_group[key])
        for key, rewards in rewards_by_group.items()
        if len(rewards) == 8 and all(r == 1.0 for r in rewards)
    ]
    mastered_by_stage = Counter(stage for _, stage in mastered)

    retirement = {}
    for stage in STAGES:
        entered = [pid for pid, s in stage_by_prompt.items() if s == stage]
        gone = [pid for pid, s in mastered if s == stage]

        def mean_pass(ids: list[str]) -> float | None:
            if not ids:
                return None
            per_prompt = [sum(prompt_rewards[pid]) / len(prompt_rewards[pid]) for pid in ids]
            return sum(per_prompt) / len(per_prompt)

        stay = [pid for pid in entered if pid not in set(gone)]
        retirement[stage] = {
            "entered": len(entered),
            "retired": len(gone),
            "retired_mean_pass_rate": mean_pass(gone),
            "remaining_mean_pass_rate": mean_pass(stay),
        }

    evaluations = [
        {
            "step": row["step"],
            "kind": row["kind"],
            "pass_at_1": row["pass_at_1"],
            "count": row["count"],
        }
        for row in eval_payload["evaluations"]
    ]
    checkpoint_curve = [e for e in evaluations if e["kind"] == "checkpoint"]

    payload = {
        "source": str(DATA),
        "model": meta["config"]["model"],
        "model_revision": meta["model_revision"],
        "git_commit": meta["git_commit"],
        "versions": meta["versions"],
        "training": meta["config"]["training"],
        "run": meta["config"]["run"],
        "curriculum": meta["curriculum"],
        "trainer": meta["trainer"],
        "metrics": {
            "steps": len(metric_rows),
            "first_step": metric_rows[0]["step"],
            "last_step": metric_rows[-1]["step"],
            "prompts_per_step": {
                "mean": sum(prompts) / len(prompts),
                "median": quantile(prompts, 0.5),
                "min": min(prompts),
                "max": max(prompts),
                "steps_at_or_below_3": sum(1 for v in prompts if v <= 3),
            },
            "sequences_per_step": {
                "mean": sum(sequences) / len(sequences),
                "median": quantile(sequences, 0.5),
                "max": max(sequences),
            },
            "by_stage": by_stage_metrics,
            "final_row": metric_rows[-1],
        },
        "rollouts": {
            "rows": total_rows,
            "groups": group_count,
            "mean_group_size": total_rows / group_count,
            "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
            "zero_variance_groups": zero_variance,
            "zero_variance_share": zero_variance / group_count,
            "all_zero_reward_groups": all_zero_reward,
            "all_one_reward_groups": all_one_reward,
            "all_tests_failed_groups": all_failed_tests,
            "mastered_groups_by_stage": dict(mastered_by_stage),
            "duplicate_code_groups": duplicate_groups,
            "duplicate_code_share": duplicate_groups / group_count,
            "mean_distinct_bodies": distinct_bodies / group_count,
            "error_classes_by_stage": {
                stage: dict(counter) for stage, counter in error_by_stage.items()
            },
            "unique_prompts": len(prompt_rewards),
            "prompts_with_one_group": sum(
                1 for pid in prompt_rewards if len(prompt_rewards[pid]) == 8
            ),
        },
        "retirement": retirement,
        "eval_curve": {
            "evaluations": evaluations,
            "checkpoint_min": min(e["pass_at_1"] for e in checkpoint_curve),
            "checkpoint_max": max(e["pass_at_1"] for e in checkpoint_curve),
            "first_checkpoint": checkpoint_curve[0],
            "last_checkpoint": checkpoint_curve[-1],
            "final": eval_payload["pass_at_1"],
            "subset": eval_payload["subset"],
        },
    }

    # Count prompts that appeared in exactly one step.
    groups_per_prompt = Counter()
    for (step, prompt_id) in rewards_by_group:
        groups_per_prompt[prompt_id] += 1
    payload["rollouts"]["prompts_in_one_step"] = sum(
        1 for pid, n in groups_per_prompt.items() if n == 1
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    summary = {
        "rollouts": payload["rollouts"],
        "retirement": payload["retirement"],
        "eval_curve": {k: v for k, v in payload["eval_curve"].items() if k != "evaluations"},
        "metrics": {k: v for k, v in payload["metrics"].items() if k != "final_row"},
    }
    print(json.dumps(summary, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
