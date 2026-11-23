# MLflow saved view for the study page

Date: 23 November 2026.
Status: implemented on `feat/mlflow-saved-view`.

## Problem Statement

The MLflow run page puts every model metric in one default section. The list is
alphabetical. The axes auto-scale to the data. The three numbers that matter,
mean reward, pass rate and advantage share, sit in the middle of 29 model
charts. A pass rate around 0.2 fills the chart, so the reader cannot tell that
it is low.

The run page layout lives in the browser, keyed by the run id. A second browser
starts blank, and every new run starts blank.

## Solution

A saved view on the experiment Chart view. MLflow stores a saved view as an
experiment tag. The tag holds the chart sections, the chart order and each
chart's y range. The page loads the tag from the URL. Any browser shows the
same layout with no setup.

The view has eight sections. Core holds mean reward, pass rate and advantage
share, in that order. Quality, optim, speed, data and gpu hold the model
metrics. Band and stage keep their own sections. The three core charts have
fixed ranges. A filter keeps parent runs only.

A CLI command writes the tag from code, so the layout is reproducible and
survives a store rebuild.

## User Stories

1. As a researcher, I want the three headline charts first, so I can read
   progress without scanning 29 charts.
2. As a researcher, I want the three in a fixed order, so the page looks the
   same on every run.
3. As a researcher, I want fixed ranges on the core charts, so a low pass rate
   looks low.
4. As a researcher, I want every model metric in a named section, so I can find
   a series by topic.
5. As a researcher, I want the GPU child run excluded, so its samples do not
   draw noise on the model charts.
6. As a researcher on another machine, I want the same layout with no setup, so
   I do not rebuild the charts.
7. As a researcher, I want the charts to refresh while the run trains, so I can
   watch progress.
8. As a researcher, I want the run page kept, so I can still read tables,
   artifacts and the GPU child.
9. As a maintainer, I want a command that writes the view, so I can recreate it
   after a store rebuild.
10. As a maintainer, I want a second write to replace the same tag, so reruns
    do not pile up views.
11. As a maintainer, I want the metric names unchanged, so old runs and the run
    folder stay compatible.
12. As a maintainer, I want the layout tested without a browser, so a change to
    the sections or ranges fails a test.
13. As a maintainer, I want a clear error when the tag exceeds the server
    limit, so a larger layout fails loudly.
14. As a new reader, I want one link that opens the view, so I do not need to
    learn MLflow internals.

## Implementation Decisions

- One new module holds the feature. It has four public functions: build the
  view state, encode the view state, write the tag, and build the route.
- The view state is a plain dict, so a test can compare it without MLflow.
- The tag key is stable: `mlflow.sharedViewState.rlvr-study`. A second write
  replaces the same tag.
- The tag value is the envelope MLflow reads:
  `{"name": str, "createdAt": int, "updatedAt": int, "state": str}`.
- `state` is `deflate;` plus base64 of a zlib stream of the JSON view state.
  This matches `textCompressDeflate` in the MLflow UI.
- The server caps an experiment tag value at 20000 characters. The writer
  checks the envelope length and raises before the call.
- The view state carries `compareRunCharts`, `compareRunSections`,
  `isAccordionReordered`, `autoRefreshEnabled`, `globalLineChartConfig`,
  `hideEmptyCharts` and `searchFilter`.
- Sections use the MLflow shape:
  `{"name", "uuid", "display": true, "isReordered": true, "columns", "cardHeight"}`.
- Charts use the MLflow line card shape with `type: "LINE"`, `uuid`,
  `metricSectionId`, `metricKey`, `selectedMetricKeys`, `displayName`,
  `xAxisKey: "step"`, `yAxisKey: "metric"` and `range`.
- Section order is core, quality, optim, speed, data, gpu, band, stage. Card
  order inside core is mean reward, pass rate, advantage share.
- Section membership is exact:
  - core: mean_reward, pass_rate, advantage_share.
  - quality: pass_at_1/checkpoint, pass_at_1/final, mean_entropy,
    truncation_rate, group_spread, group_sd.
  - optim: grad_norm, update_norm.
  - speed: step_seconds, generation_seconds, scoring_seconds,
    scoring_wait_seconds, update_seconds, completion_tokens_per_second,
    sequences_per_second.
  - data: prompts, train_prompts, parked_prompts, sequences, cache_hits,
    mean_completion_tokens.
  - gpu: the five memory series.
  - band: one pass rate card per band.
  - stage: one solved card and one total card per band.
- The heartbeat metric stays on the run page and is not in the view.
- The parent run also logs the heartbeat and the MLflow system metrics. The
  Chart view adds a chart for every metric key missing from `compareRunCharts`,
  so those keys would appear as extra sections. The view includes a deleted
  card for each of them. A deleted card blocks the auto-add and does not
  render. The hidden keys are `rlvr/heartbeat_seconds` and these system keys:
  `system/cpu_utilization_percentage`, `system/disk_available_megabytes`,
  `system/disk_usage_megabytes`, `system/disk_usage_percentage`,
  `system/network_receive_megabytes`, `system/network_transmit_megabytes`,
  `system/system_memory_usage_megabytes`,
  `system/system_memory_usage_percentage`,
  `system/gpu_0_memory_usage_percentage`,
  `system/gpu_0_memory_usage_megabytes`,
  `system/gpu_0_utilization_percentage`,
  `system/gpu_0_power_usage_watts`,
  `system/gpu_0_power_usage_percentage`.
- The visible page has eight sections and 41 charts. The deleted cards are not
  counted.
- Fixed ranges only on core. Mean reward is -1 to 1. Pass rate and advantage
  share are 0 to 1. Every other chart keeps the auto range.
- The search filter is ``tags.`rlvr/run_dir` != ''``. The parent run carries
  the tag. The GPU child does not.
- The route is
  `#/experiments/<id>/runs?viewStateShareKey=<id>&compareRunsMode=CHART`.
  The view mode belongs in the URL, not the tag.
- The CLI command is `rlvr saved-view`. It opens the runs root, ensures the
  `rlvr` experiment, writes the tag and prints the route.
- Metric names in the store do not change. The run folder does not change.
  The training path does not change. The run page does not change.

## Testing Decisions

A good test checks behaviour a reader can see from outside the module. It does
not assert private helpers or the order of dictionary keys.

Four tests:

1. The builder test checks the eight sections in order, the exact card
   membership of each section, the three core cards in order, the three core
   ranges, the run filter, and the deleted cards for the heartbeat and the
   system metrics.
2. The encoder test checks the envelope fields, the `deflate;` prefix, a zlib
   round trip, and the error above 20000 characters.
3. The writer test checks a real temporary SQLite store. It reads the tag with
   `MlflowClient`, checks the value, and checks that a second write replaces the
   tag rather than adding one.
4. The integration test runs `main(["saved-view", ...])` against a temporary
   store. It reads the tag, decodes the state and checks the printed route.

Prior art is `tests/test_mlflow_import.py`, which uses a temporary runs root and
a real local MLflow store, and `tests/test_cli.py`, which calls `main`.

## Out of Scope

- The run page Model metrics tab.
- Browser local storage and per-browser layout.
- A custom dashboard or any new web UI.
- Metric renames in the store.
- Chart types other than line.
- Editing the saved view through the UI.
- Charts for the GPU child run.
- The dashboard speed problem.

## Further Notes

MLflow is pinned to 3.18.0. The saved view is a 3.18 feature. A future MLflow
upgrade may change the tag shape, so the tests pin the current shape.

The UI opens a shared view read-only and shows a banner. Auto-refresh stays on
because the flag is not shareable and defaults to true.

The origin varies. On the Mac the tunnel is `http://127.0.0.1:8083`. The
command prints the hash route, and the README shows the origin.
