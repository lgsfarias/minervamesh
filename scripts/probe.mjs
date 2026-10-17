#!/usr/bin/env node
/**
 * Sonda de conferência visual (v1.2.0): abre um módulo, executa ações (abrir aba,
 * clicar em Executar e esperar o fim, trocar de tema) e captura a página inteira.
 * Complementa scripts/shots.mjs, que só captura o estado inicial.
 *
 * Uso:
 *   node scripts/probe.mjs --page navier-stokes --do run --out docs/v1.2.0/shots/probe/ns-run.png
 *   node scripts/probe.mjs --page viga --do tab:code --width 1440
 *   node scripts/probe.mjs --page viga --do run,tab:problem --width 390 --theme dark --lang pt
 *
 * --do: lista separada por vírgula: run | tab:params | tab:problem | tab:code | click:<seletor> | wait:<ms>
 * Requer: puppeteer-core (devDependency), Google Chrome, python3. Lista de módulos: assets/menu.json.
 */
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 8126;

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const pageId = arg("page", "navier-stokes");
const actions = (arg("do", "") || "").split(",").filter(Boolean);
const width = Number(arg("width", 1440));
const lang = arg("lang", "en");
const theme = arg("theme", "light");
const out = resolve(ROOT, arg("out", `docs/v1.2.0/shots/probe/${pageId}_${width}_${lang}_${theme}_${actions.join("+").replace(/[^a-z0-9+]/gi, "") || "idle"}.png`));
const RUN_TIMEOUT = Number(arg("run-timeout", 900)) * 1000;

const menu = JSON.parse(readFileSync(resolve(ROOT, "assets/menu.json"), "utf8")).modules;
const mod = pageId === "menu" ? { path: "index.html" } : menu.find((m) => m.id === pageId);
if (!mod) throw new Error(`módulo desconhecido: ${pageId}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: ROOT, stdio: "ignore" });
  await sleep(800);
  mkdirSync(dirname(out), { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, protocolTimeout: RUN_TIMEOUT + 60_000 });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => console.log("pageerror:", String(e.message).slice(0, 200)));
    page.on("console", (m) => { if (m.type() === "error") console.log("console.error:", m.text().slice(0, 200)); });
    await page.evaluateOnNewDocument((t) => {
      try { localStorage.setItem("minervamesh.theme", t); } catch {}
      addEventListener("py:ready", () => { window.__pyReady = true; });
    }, theme);
    await page.goto(`http://127.0.0.1:${PORT}/${mod.path}?lang=${lang}`, { waitUntil: "domcontentloaded" });
    if (pageId !== "menu") await page.waitForFunction(() => window.__pyReady === true, { timeout: 240_000, polling: 250 });
    await sleep(1500);
    for (const a of actions) {
      if (a === "run") {
        await page.click("#run-btn");
        await page.waitForFunction((doneSrc, useCanvas) => {
          const root = document.querySelector("#plot-output");
          if (!root) return false;
          const st = document.getElementById("mm-status");
          if (st && !st.classList.contains("hidden") && st.querySelector("#mm-status-bar")?.classList.contains("bg-mm-ok")) return true;
          if (useCanvas) return root.querySelectorAll("canvas").length > 0;
          const imgs = [...root.querySelectorAll("img")].filter((i) => i.src.startsWith("data:"));
          const txt = root.textContent || "";
          const busy = /Iniciando|Calculando|Executando|Carregando|Running|Starting/.test(txt) && !/Conclu|Finished|Complete/i.test(txt);
          const done = doneSrc ? new RegExp(doneSrc, "i").test(txt) : true;
          return imgs.length > 0 && !busy && done;
        }, { timeout: RUN_TIMEOUT, polling: 250 }, mod.doneText || null, mod.output === "canvas");
        await sleep(2000);
      } else if (a.startsWith("tab:")) {
        await page.click(`#mm-tab-${a.slice(4)}`);
        await sleep(1200);
      } else if (a.startsWith("click:")) {
        await page.click(a.slice(6));
        await sleep(800);
      } else if (a.startsWith("wait:")) {
        await sleep(Number(a.slice(5)));
      }
    }
    await page.screenshot({ path: out, fullPage: true });
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    console.log(`${out}${over > 0 ? `  OVERFLOW +${over}px` : ""}`);
  } finally {
    await browser.close();
    server.kill();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
