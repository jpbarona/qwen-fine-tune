// Inline the Vite bundle into one HTML file and copy it into the package.
// Python fills the `__RLVR_INITIAL_DATA__` marker on each request.
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const dist = resolve(here, '../dist')
const target = resolve(here, '../../src/rlvr/web/index.html')

const escapeScript = (text) => text.replace(/<\/script/gi, '<\\/script')

let html = await readFile(resolve(dist, 'index.html'), 'utf8')

const css = [...html.matchAll(/<link[^>]+rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g)]
for (const [tag, href] of css) {
  const source = await readFile(resolve(dist, href.replace(/^\.?\//, '')), 'utf8')
  html = html.replace(tag, () => `<style>${source}</style>`)
}

const scripts = [...html.matchAll(/<script[^>]+type="module"[^>]+src="([^"]+)"[^>]*><\/script>/g)]
for (const [tag, src] of scripts) {
  const source = await readFile(resolve(dist, src.replace(/^\.?\//, '')), 'utf8')
  html = html.replace(tag, () => `<script type="module">${escapeScript(source)}</script>`)
}

if (html.includes('src="/assets/') || html.includes('href="/assets/')) {
  throw new Error('the emitted page still references an external asset')
}
if (!html.includes('__RLVR_INITIAL_DATA__')) {
  throw new Error('the data marker is missing from the page')
}

await mkdir(dirname(target), { recursive: true })
await writeFile(target, html, 'utf8')
console.log(`wrote ${target} (${Math.round(html.length / 1024)} kB)`)
