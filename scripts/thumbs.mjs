#!/usr/bin/env node
/**
 * Miniaturas dos módulos (v1.2.0): abre cada módulo em Chrome headless, roda com
 * os parâmetros default em inglês e salva só a figura, sem toolbar nem título
 * da janela:
 *   assets/thumbs/<id>.png     figura completa, 1000 px de largura (saída de exemplo antes do Run)
 *   assets/thumbs/<id>-sm.png  recorte 4:3 de 480 px (card do menu; região "thumb" de assets/menu.json)
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

async function main() {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: ROOT, stdio: "ignore" });
  await sleep(800);
  mkdirSync(OUT_DIR, { recursive: true });
  mkdirSync(TMP, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, protocolTimeout: RUN_TIMEOUT + 60_000, args: ["--window-size=1400,1000"] });
  try {
    for (const m of modules) {
      if (only && m.id !== only) continue;
      const page = await browser.newPage();
      await page.setViewport({ width: 1400, height: 1000, deviceScaleFactor: 1 });
      await page.setCacheEnabled(false);
      const t0 = Date.now();
      try {
        await page.evaluateOnNewDocument(() => { addEventListener("py:ready", () => { window.__pyReady = true; }); });
        await page.goto(`http://127.0.0.1:${PORT}/${m.path}?lang=en`, { waitUntil: "domcontentloaded" });
        await page.waitForFunction(() => window.__pyReady === true, { timeout: LOAD_TIMEOUT, polling: 250 });
        await page.click("#run-btn");
        await page.waitForFunction((doneSrc, useCanvas) => {
          const root = document.querySelector("#plot-output");
          if (!root) return false;
          if (useCanvas) return root.querySelectorAll("canvas").length > 0;
          const st = document.getElementById("mm-status");
          if (st && !st.classList.contains("hidden") && st.querySelector("#mm-status-bar")?.classList.contains("bg-green-600")) return true;
          const imgs = [...root.querySelectorAll("img")].filter((i) => i.src.startsWith("data:"));
          const txt = root.textContent || "";
          const busy = /Iniciando|Calculando|Executando|Carregando|Running|Starting/.test(txt) && !/Conclu|Finished|Complete/i.test(txt);
          const done = doneSrc ? new RegExp(doneSrc, "i").test(txt) : true;
          return imgs.length > 0 && !busy && done;
        }, { timeout: RUN_TIMEOUT, polling: 250 }, m.doneText || null, m.output === "canvas");
        await sleep(2000); // canvas termina de desenhar; GIF fica completo
        // Bitmap da figura: maior canvas (webagg) ou primeiro <img> base64
        const dataUrl = await page.evaluate((useCanvas) => {
          const root = document.querySelector("#plot-output");
          if (useCanvas) {
            const cs = [...root.querySelectorAll("canvas")].sort((a, b) => b.width * b.height - a.width * a.height);
            return cs[0] ? cs[0].toDataURL("image/png") : null;
          }
          const img = [...root.querySelectorAll("img")].find((i) => i.src.startsWith("data:"));
          return img ? img.src : null;
        }, m.output === "canvas");
        if (!dataUrl) throw new Error("figura não encontrada em #plot-output");
        const [head, b64] = dataUrl.split(",", 2);
        const ext = /image\/gif/.test(head) ? "gif" : "png";
        const raw = resolve(TMP, `${m.id}.${ext}`);
        writeFileSync(raw, Buffer.from(b64, "base64"));
        const t = m.thumb || { x: 0, y: 0, w: 1, h: 1 };
        // Sem -I: o Pillow pode estar no site-packages do usuário, que o modo isolado ignora
        const r = spawnSync("python3", [resolve(ROOT, "scripts/thumbs_crop.py"), raw, resolve(OUT_DIR, `${m.id}.png`), resolve(OUT_DIR, `${m.id}-sm.png`), String(t.x), String(t.y), String(t.w), String(t.h)], { encoding: "utf8" });
        if (r.status !== 0) throw new Error(`thumbs_crop.py: ${r.stderr.trim()}`);
        console.log(`${m.id} ok ${((Date.now() - t0) / 1000).toFixed(0)}s ${r.stdout.trim()}`);
      } catch (e) {
        console.log(`${m.id} ERRO ${String(e.message).slice(0, 200)}`);
      }
      await page.close();
    }
  } finally {
    await browser.close();
    server.kill();
    rmSync(TMP, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
