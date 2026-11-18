# Dashboard

The React page served by `rlvr dashboard`. Python reads the run folders and
owns `/data.json`. This app only draws the payload.

## Build

```
cd dashboard
npm ci
npm test
npm run build
```

The build writes one self-contained file to `src/rlvr/web/index.html`: the JS
and CSS are inlined and the `__RLVR_INITIAL_DATA__` marker is left for Python
to fill on each request. Commit that file. The server reads no other assets and
makes no network requests.

## Serve

```
rlvr dashboard --runs-root runs --port 8081
```

## Layout

- Header: product mark, last payload time, live/pause toggle.
- Per run: arm and run name, status, stage, last write, started, elapsed.
- Step progress against the target from `meta.json`.
- Two KPI strips: reward, pass rate, pass@1, entropy, length, truncation, cache
  hits, prompts, sequences, stage completion, and the GPU memory split in GiB.
- Charts: reward, pass rate, pass@1, entropy, completion tokens, truncation,
  batch size, stage completion, GPU memory, reward distribution, and tests
  passed distribution. SVG only, no chart library.
- Tables: pass rate per band, and every evaluation with its band.

`metrics_age_seconds` in `/data.json` drives the stale hint. A run marked
running with no metrics write for five minutes shows the age in amber, and a
write older than fifteen minutes is reported as stopped by the server.

## Tooling notes

No CDN, no web fonts, no chart library. Tailwind v4 is compiled at build time,
React is bundled, numbers use tabular figures. Add a dependency only if the SVG
charts cannot express the panel.

## Source and refresh behavior

`src/App.tsx` owns the page shell. `src/components/` contains run panels,
questions, and the system strip. `src/useDashboard.ts` owns polling and
`src/validateDashboard.ts` validates the nested collections used by the page.
Both the initial embedded payload and refresh responses pass through that
validator. Older runs can omit timing metrics and optional question and drop
collections. Invalid responses keep the last received data visible and show a
retry control.

Requests run one at a time with a 15-second timeout. Live refresh waits five
seconds after each request completes, and pauses while the page is hidden.
Pausing or leaving the page cancels its request. `npm test` uses Node 22's
built-in test runner and TypeScript stripping to exercise malformed responses,
sparse GPU readings, historical metrics, and waiting runs.
