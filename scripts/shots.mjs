#!/usr/bin/env node
/**
 * Capturas de tela das 11 páginas (menu + 10 módulos) para conferência visual
 * da v1.2.0: larguras de desktop, tablet e celular, nos dois idiomas e, quando a
 * página suportar, nos dois temas. Não executa simulação: captura o estado
 * logo após py:ready (ou após o carregamento, no menu).
 *
 * Uso:
 *   node scripts/shots.mjs --label antes
 *   node scripts/shots.mjs --label etapa4 --only navier-stokes --widths 390
 *   node scripts/shots.mjs --label etapa2 --themes light,dark
 *
 * Saída: docs/v1.2.0/shots/<label>/<página>_<largura>_<lang>_<tema>.png (pasta ignorada pelo git)
 * Requer: puppeteer-core (devDependency), Google Chrome instalado, python3.
 * Lista de módulos: assets/menu.json.
 */
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8125;

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const label = arg("label", `shots-${Date.now()}`);
const only = arg("only", null);
const widths = arg("widths", "1440,820,390").split(",").map(Number);
const langs = arg("langs", "pt,en").split(",");
const themes = arg("themes", "light").split(",");
const LOAD_TIMEOUT = Number(arg("load-timeout", 240)) * 1000;
const OUT_DIR = resolve(ROOT, "docs/v1.2.0/shots", label);

const menu = JSON.parse(readFileSync(resolve(ROOT, "assets/menu.json"), "utf8")).modules;
const PAGES = [{ id: "menu", path: "index.html", py: false }, ...menu.map((m) => ({ id: m.id, path: m.path, py: true }))];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: ROOT, stdio: "ignore" });
  await sleep(800);
  mkdirSync(OUT_DIR, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, protocolTimeout: LOAD_TIMEOUT + 60_000 });
  let n = 0;
  try {
    for (const p of PAGES) {
      if (only && p.id !== only) continue;
      for (const theme of themes) {
        for (const lang of langs) {
          const page = await browser.newPage();
          await page.setViewport({ width: widths[0], height: 900, deviceScaleFactor: 1 });
          await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: theme }]);
          // Tema explícito via localStorage (chave da v1.2.0); páginas antigas ignoram.
          await page.evaluateOnNewDocument((t) => {
            try { localStorage.setItem("minervamesh.theme", t); } catch {}
            addEventListener("py:ready", () => { window.__pyReady = true; });
          }, theme);
          const url = `http://127.0.0.1:${PORT}/${p.path}?lang=${lang}`;
          try {
            await page.goto(url, { waitUntil: "networkidle0", timeout: LOAD_TIMEOUT });
            if (p.py) await page.waitForFunction(() => window.__pyReady === true, { timeout: LOAD_TIMEOUT, polling: 250 });
            await sleep(1500); // MathJax e fontes
            for (const w of widths) {
              await page.setViewport({ width: w, height: 900, deviceScaleFactor: 1 });
              await sleep(400);
              const file = resolve(OUT_DIR, `${p.id}_${w}_${lang}_${theme}.png`);
              await page.screenshot({ path: file, fullPage: true });
              // Rolagem horizontal é defeito: registra no log quando o documento é mais largo que a janela.
              const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
              console.log(`${p.id} ${w}px ${lang} ${theme}${over > 0 ? `  OVERFLOW +${over}px` : ""}`);
              n++;
            }
          } catch (e) {
            console.log(`${p.id} ${lang} ${theme} ERRO ${String(e.message).slice(0, 160)}`);
          }
          await page.close();
        }
      }
    }
  } finally {
    await browser.close();
    server.kill();
  }
  console.log(`${n} capturas em ${OUT_DIR}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
