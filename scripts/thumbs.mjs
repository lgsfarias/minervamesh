#!/usr/bin/env node
/**
 * Miniaturas dos módulos (v1.2.0; três por card desde a v1.2.1): abre cada módulo em
 * Chrome headless, roda em inglês e salva só a figura, sem toolbar nem título da janela:
 *   assets/thumbs/<id>.png      figura completa da execução com os parâmetros padrão,
 *                               1000 px de largura (saída de exemplo antes do Run)
 *   assets/thumbs/<id>-<n>.webp miniatura n (1..3) do card do menu: região "thumbs[n-1]"
 *                               de assets/menu.json (só a área dos eixos), ajustada a
 *                               16:9 em 384 × 216, WebP (v1.2.2)
 *
 * Cada entrada de "thumbs" pode trazer "set" (id do input → valor): o módulo roda de
 * novo com esses parâmetros (número/texto → value, checkbox → checked, select → value)
 * antes do recorte. Entradas com o mesmo "set" compartilham uma execução.
 *
 * A figura é lida do próprio bitmap (canvas.toDataURL() no backend webagg, ou o
 * src base64 do <img>), por isso não depende do tamanho da janela. GIFs usam o
 * último quadro. Corte e redimensionamento em scripts/thumbs_crop.py (Pillow).
 *
 * Uso:
 *   node scripts/thumbs.mjs
 *   node scripts/thumbs.mjs --only navier-stokes
 * Requer: puppeteer-core (devDependency), Google Chrome, python3 com Pillow.
 */
import puppeteer from "puppeteer-core";
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8124;
const OUT_DIR = resolve(ROOT, "assets/thumbs");
const TMP = resolve(tmpdir(), "minervamesh-thumbs");

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const only = arg("only", null);
const LOAD_TIMEOUT = Number(arg("load-timeout", 240)) * 1000;
const RUN_TIMEOUT = Number(arg("run-timeout", 900)) * 1000;

const modules = JSON.parse(readFileSync(resolve(ROOT, "assets/menu.json"), "utf8")).modules;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Agrupa as miniaturas de um módulo por conjunto de parâmetros: uma execução por grupo.
// A execução padrão (sem "set") vem primeiro e é a única que gera a figura completa.
function runsOf(m) {
  const runs = new Map();
  (m.thumbs || []).forEach((t, i) => {
    const key = JSON.stringify(t.set || {});
    if (!runs.has(key)) runs.set(key, { set: t.set || null, crops: [] });
    runs.get(key).crops.push({ n: i + 1, ...t });
  });
  return [...runs.values()].sort((a, b) => (a.set ? 1 : 0) - (b.set ? 1 : 0));
}

// Roda o módulo da página aberta (já em py:ready) e devolve o bitmap da figura em data URL.
async function runAndCapture(page, m, set) {
  if (set) {
    await page.evaluate((set) => {
      for (const [id, v] of Object.entries(set)) {
        const e = document.getElementById(id);
        if (!e) throw new Error(`input #${id} não existe`);
        if (e.type === "checkbox") e.checked = v === true || v === "true";
        else e.value = String(v);
        e.dispatchEvent(new Event("input", { bubbles: true }));
        e.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }, set);
  }
  await page.click("#run-btn");
  await page.waitForFunction((doneSrc, useCanvas) => {
    const root = document.querySelector("#plot-output");
    if (!root) return false;
    if (useCanvas) return root.querySelectorAll("canvas").length > 0;
    const st = document.getElementById("mm-status");
    if (st && !st.classList.contains("hidden") && st.querySelector("#mm-status-bar")?.classList.contains("bg-mm-ok")) return true;
    const imgs = [...root.querySelectorAll("img")].filter((i) => i.src.startsWith("data:"));
    const txt = root.textContent || "";
    const busy = /Iniciando|Calculando|Executando|Carregando|Running|Starting/.test(txt) && !/Conclu|Finished|Complete/i.test(txt);
    const done = doneSrc ? new RegExp(doneSrc, "i").test(txt) : true;
    return imgs.length > 0 && !busy && done;
  }, { timeout: RUN_TIMEOUT, polling: 250 }, m.doneText || null, m.output === "canvas");
  await sleep(2000); // canvas termina de desenhar; GIF fica completo
  // Bitmap da figura: maior canvas (webagg) ou primeiro <img> base64
  return page.evaluate((useCanvas) => {
    const root = document.querySelector("#plot-output");
    if (useCanvas) {
      const cs = [...root.querySelectorAll("canvas")].sort((a, b) => b.width * b.height - a.width * a.height);
      return cs[0] ? cs[0].toDataURL("image/png") : null;
    }
    const img = [...root.querySelectorAll("img")].find((i) => i.src.startsWith("data:"));
    return img ? img.src : null;
  }, m.output === "canvas");
}

async function main() {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: ROOT, stdio: "ignore" });
  await sleep(800);
  mkdirSync(OUT_DIR, { recursive: true });
  mkdirSync(TMP, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, protocolTimeout: RUN_TIMEOUT + 60_000, args: ["--window-size=1400,1000"] });
  try {
    for (const m of modules) {
      if (only && m.id !== only) continue;
      for (const run of runsOf(m)) {
        const page = await browser.newPage();
        await page.setViewport({ width: 1400, height: 1000, deviceScaleFactor: 1 });
        await page.setCacheEnabled(false);
        const t0 = Date.now();
        const label = run.set ? `${m.id} ${JSON.stringify(run.set)}` : m.id;
        try {
          await page.evaluateOnNewDocument(() => { addEventListener("py:ready", () => { window.__pyReady = true; }); });
          await page.goto(`http://127.0.0.1:${PORT}/${m.path}?lang=en`, { waitUntil: "domcontentloaded" });
          await page.waitForFunction(() => window.__pyReady === true, { timeout: LOAD_TIMEOUT, polling: 250 });
          const dataUrl = await runAndCapture(page, m, run.set);
          if (!dataUrl) throw new Error("figura não encontrada em #plot-output");
          const [head, b64] = dataUrl.split(",", 2);
          const ext = /image\/gif/.test(head) ? "gif" : "png";
          const raw = resolve(TMP, `${m.id}.${ext}`);
          writeFileSync(raw, Buffer.from(b64, "base64"));
          const outs = [];
          for (const c of run.crops) {
            // Só a execução padrão grava a figura completa ("-" pula)
            const full = run.set ? "-" : resolve(OUT_DIR, `${m.id}.png`);
            // Sem -I: o Pillow pode estar no site-packages do usuário, que o modo isolado ignora
            const r = spawnSync("python3", [resolve(ROOT, "scripts/thumbs_crop.py"), raw, full, resolve(OUT_DIR, `${m.id}-${c.n}.webp`), String(c.x), String(c.y), String(c.w), String(c.h)], { encoding: "utf8" });
            if (r.status !== 0) throw new Error(`thumbs_crop.py: ${r.stderr.trim()}`);
            outs.push(`${m.id}-${c.n} ${r.stdout.trim()}`);
          }
          console.log(`${label} ok ${((Date.now() - t0) / 1000).toFixed(0)}s\n  ${outs.join("\n  ")}`);
        } catch (e) {
          console.log(`${label} ERRO ${String(e.message).slice(0, 200)}`);
        }
        await page.close();
      }
    }
  } finally {
    await browser.close();
    server.kill();
    rmSync(TMP, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
