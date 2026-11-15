# Dashboard

The React page served by `rlvr dashboard`. Python reads the run folders and
owns `/data.json`. This app only draws the payload.

## Build

```
cd dashboard
npm install
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
