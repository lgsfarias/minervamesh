#!/usr/bin/env node
/**
 * Baseline das simulações: abre cada módulo em Chrome headless, roda com os
 * parâmetros informados e registra métricas exibidas, hash das imagens e tempos.
 *
 * Uso:
 *   node scripts/baseline.mjs --label v1.0.0-defaults
 *   node scripts/baseline.mjs --label ns-cavity-tcc --only navier-stokes \
 *        --set scenario=cavity,L=1,H=1,nx=41,ny=41,Re=100,dt=0.01,total_steps=2000
 *
 * Saída: docs/v1.1.0/baseline/<label>.json (resumo humano em docs/v1.1.0/aceite.md)
 * Requer: puppeteer-core (devDependency), Google Chrome instalado, python3.
 * Cache do browser desabilitado: cada página baixa o Pyodide de novo (~5 s), mas nunca mede um .py antigo.
 */
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8123;
const OUT_DIR = resolve(ROOT, "docs/v1.1.0/baseline");

const MODULES = [
  // plt.show() do backend wasm do matplotlib: desenha num <canvas> anexado ao body, não num <img>
  { id: "mdf-conducao", path: "simulations/mdf/conducao-permanente-barra1d-geracao/index.html", run: "#runBtn", out: "body", canvas: true },
  { id: "mdf-onda", path: "simulations/mdf/equacao-onda/index.html", run: "#runWaveBtn", out: "#wave-output" },
  { id: "mef-conducao", path: "simulations/mef/conducao-permanente-barra1d-mef/index.html", run: "#runMEF1D", out: "#mef1d-output" },
  { id: "calor-transiente-1d", path: "simulations/mef/calor-transiente-1d/index.html", run: "#runHeatBtn", out: "#heat-output" },
  { id: "conveccao-difusao", path: "simulations/mef/conveccao-difusao-1d/index.html", run: "#runConveccaoDifusao1D", out: "#conveccao-difusao-output", extraText: "#pe-info" },
  { id: "viga", path: "simulations/mef/viga-1d-mef/index.html", run: "#runViga1D", out: "#viga1d-output" },
  { id: "vibracao", path: "simulations/mef/vibracao-1d-mef/index.html", run: "#runVibracao1D", out: "#vibracao1d-output" },
  { id: "transcalor", path: "simulations/mef/transcalor/index.html", run: "#run-btn", out: "#plot-output" },
  { id: "potencial", path: "simulations/mef/escoamento-fluido-2d/escoamento-potencial/index.html", run: "#run-btn", out: "#plot-output", extraText: "#lift-results" },
  { id: "navier-stokes", path: "simulations/mef/escoamento-fluido-2d/navier-stokes/index.html", run: "#run-btn", out: "#plot-output", doneText: /Conclu|Parada/ },
];

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const label = arg("label", `run-${Date.now()}`);
const only = arg("only", null);
const sets = Object.fromEntries((arg("set", "") || "").split(",").filter(Boolean).map((kv) => kv.split("=")));
const LOAD_TIMEOUT = Number(arg("load-timeout", 240)) * 1000;
const RUN_TIMEOUT = Number(arg("run-timeout", 900)) * 1000;

const sha = (s) => createHash("sha256").update(s).digest("hex").slice(0, 16);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  // Servidor local com Cache-Control: no-store (reforço do setCacheDisabled).
  const serverPy = [
    "import http.server, functools",
    "class H(http.server.SimpleHTTPRequestHandler):",
    "    def end_headers(self):",
    "        self.send_header('Cache-Control', 'no-store'); super().end_headers()",
    "    def log_message(self, *a): pass",
    `http.server.ThreadingHTTPServer(('127.0.0.1', ${PORT}), H).serve_forever()`,
  ].join("\n");
  const server = spawn("python3", ["-c", serverPy], { cwd: ROOT, stdio: "ignore" });
  await sleep(800);
  mkdirSync(OUT_DIR, { recursive: true });
  // protocolTimeout alto: enquanto o Pyodide roda, a thread principal da página fica bloqueada e o CDP não responde.
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, protocolTimeout: RUN_TIMEOUT + 60_000, args: ["--window-size=1400,1000"] });
  const results = [];
  try {
    for (const m of MODULES) {
      if (only && m.id !== only) continue;
      const page = await browser.newPage();
      const url = `http://127.0.0.1:${PORT}/${m.path}`;
      const r = { id: m.id, url, params: {}, errors: [] };
      try {
      await page.setViewport({ width: 1400, height: 1000 });
      await page.setCacheEnabled(false); // nunca medir um simulation.py antigo vindo do cache
      page.on("pageerror", (e) => r.errors.push(String(e.message).slice(0, 300)));
      page.on("console", (msg) => { if (msg.type() === "error") r.errors.push(msg.text().slice(0, 300)); });
      const t0 = Date.now();
      // Marca py:ready numa flag global: nem toda página tem o dialog #loading (MDF condução não tem)
      await page.evaluateOnNewDocument(() => { addEventListener("py:ready", () => { window.__pyReady = true; }); });
      await page.goto(url, { waitUntil: "domcontentloaded" });
      await page.waitForFunction(() => window.__pyReady === true, { timeout: LOAD_TIMEOUT, polling: 250 });
      r.load_s = +((Date.now() - t0) / 1000).toFixed(1);
      // Aplica overrides e registra todos os inputs
      for (const [k, v] of Object.entries(sets)) {
        const ok = await page.evaluate((k, v) => {
          const el = document.getElementById(k); if (!el) return false;
          el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); return true;
        }, k, v);
        if (!ok) r.errors.push(`override: input #${k} não existe`);
      }
      r.params = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll("input, select")].filter((e) => e.id).map((e) => [e.id, e.type === "checkbox" ? e.checked : e.value])));
      const before = await page.evaluate((sel) => [...document.querySelector(sel).querySelectorAll("img")].map((i) => i.src), m.out);
      await page.evaluate(() => { window.__minervamesh_result = undefined; });
      const t1 = Date.now();
      await page.click(m.run);
      await page.waitForFunction((sel, beforeJson, doneSrc, useCanvas) => {
        const root = document.querySelector(sel); if (!root) return false;
        if (useCanvas) return root.querySelectorAll("canvas").length > 0;
        const imgs = [...root.querySelectorAll("img")].map((i) => i.src).filter((s) => s.startsWith("data:"));
        const changed = JSON.stringify(imgs) !== beforeJson && imgs.length > 0;
        const dialogOpen = [...document.querySelectorAll("dialog")].some((d) => d.open);
        const txt = root.textContent || "";
        const busy = /Iniciando|Calculando|Executando|Carregando/.test(txt) && !/Conclu/.test(txt);
        const done = doneSrc ? new RegExp(doneSrc).test(txt) : true;
        return changed && !dialogOpen && !busy && done;
      }, { timeout: RUN_TIMEOUT, polling: 250 }, m.out, JSON.stringify(before), m.doneText ? m.doneText.source : null, !!m.canvas);
      if (m.canvas) await sleep(1500); // deixa o canvas terminar de desenhar
      r.run_s = +((Date.now() - t1) / 1000).toFixed(1);
      const snap = await page.evaluate((sel, extra) => {
        const root = document.querySelector(sel);
        const clone = root.cloneNode(true); clone.querySelectorAll("img, style, script").forEach((n) => n.remove());
        const text = clone.textContent.replace(/\s+/g, " ").trim();
        const imgs = [...root.querySelectorAll("img")].map((i) => i.src).filter((s) => s.startsWith("data:"))
          .concat([...root.querySelectorAll("canvas")].map((c) => { try { return c.toDataURL("image/png"); } catch { return "data:canvas-indisponivel"; } }));
        const extraText = extra ? (document.querySelector(extra)?.textContent || "").replace(/\s+/g, " ").trim() : "";
        return { text, imgs, extraText };
      }, m.out, m.extraText || null);
      r.text = m.canvas ? "(gráfico em canvas; comparar pelo hash)" : snap.text; if (snap.extraText) r.extra_text = snap.extraText;
      r.images = snap.imgs.map((s) => ({ kind: s.slice(5, s.indexOf(";")), bytes: s.length, sha256_16: sha(s) }));
      // Resultado numérico exportado pelo gancho _export_result (spec 2, passo 0)
      r.result = await page.evaluate(() => window.__minervamesh_result ?? null);
      r.result_sha256_16 = r.result ? sha(JSON.stringify(r.result)) : null;
      } catch (e) {
        r.failed = String(e.message).slice(0, 300);
      }
      results.push(r);
      console.log(`${m.id.padEnd(22)} load ${String(r.load_s ?? "-").padStart(6)}s  run ${String(r.run_s ?? "-").padStart(7)}s  imgs ${r.images?.length ?? 0}  result ${r.result_sha256_16 ?? "-"}  ${r.failed ? "FALHOU " + r.failed : r.errors.length ? "ERR " + r.errors[0] : ""}`);
      writeFileSync(resolve(OUT_DIR, `${label}.partial.json`), JSON.stringify({ label, results }, null, 2));
      await page.close();
    }
  } finally {
    await browser.close();
    server.kill();
  }
  const out = { label, date: new Date().toISOString(), overrides: sets, results };
  writeFileSync(resolve(OUT_DIR, `${label}.json`), JSON.stringify(out, null, 2));
  console.log(`\nGravado em docs/v1.1.0/baseline/${label}.json`);
}

main().catch((e) => { console.error(e); process.exit(1); });
