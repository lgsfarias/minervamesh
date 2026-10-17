#!/usr/bin/env node
/**
 * Verificação estática do i18n (v1.1.0, etapa 3):
 *   - paridade de chaves entre i18n.pt.json e i18n.en.json (comum e por módulo);
 *   - toda chave usada em data-i18n / data-i18n-attr / data-mm-title no HTML existe
 *     no JSON do módulo ou no comum;
 *   - chaves definidas e não usadas em lugar nenhum (órfãs; aviso, não erro);
 *   - toda chave usada em i18n_t("…") no simulation.py do módulo existe no JSON
 *     do módulo ou no comum (etapa 4).
 *
 * Uso: node scripts/i18n-check.mjs   (sai com código 1 se houver chave faltando)
 */
import { readFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const common = { pt: read(join(ROOT, "assets/i18n/common.pt.json")), en: read(join(ROOT, "assets/i18n/common.en.json")) };

function* walk(dir) {
  for (const n of readdirSync(dir)) {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (n === "index.html") yield p;
  }
}

function keysInHtml(html) {
  const keys = new Set();
  for (const m of html.matchAll(/data-i18n="([^"]+)"/g)) keys.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g))
    m[1].split(";").forEach((pair) => { const k = pair.split(":")[1]?.trim(); if (k) keys.add(k); });
  for (const m of html.matchAll(/data-mm-title="([^"]+)"/g)) keys.add(m[1]);
  for (const m of html.matchAll(/data-mm-sibling="[^"|]+\|([^"]+)"/g)) keys.add(m[1]);
  return keys;
}

// Chaves usadas pelo app.js (markup injetado)
const appJs = readFileSync(join(ROOT, "assets/app.js"), "utf8");
const appKeys = keysInHtml(appJs);
for (const m of appJs.matchAll(/\b(?:t|fmt)\("([^"]+)"/g)) appKeys.add(m[1]);
for (const m of appJs.matchAll(/setRunBusy\(true, "([^"]+)"/g)) appKeys.add(m[1]);
for (const m of appJs.matchAll(/dataset\.i18n = "([^"]+)"/g)) appKeys.add(m[1]);
// v1.2.0: chaves escolhidas em tempo de execução (botão de tema: "nav.theme.light" / "nav.theme.dark" como literais)
for (const m of appJs.matchAll(/"((?:nav|menu|result|tab|summary|code)\.[a-z0-9_.-]+)"/g)) appKeys.add(m[1]);
// v1.2.0: cards do menu montados a partir de assets/menu.json (buildMenu em app.js)
const menu = JSON.parse(readFileSync(join(ROOT, "assets/menu.json"), "utf8"));
for (const mod of menu.modules) {
  for (const k of [`menu.${mod.id}.title`, `menu.${mod.id}.desc`, `menu.tag.${mod.method}`, `menu.tag.${mod.dim}`, `menu.tag.${mod.regime}`]) appKeys.add(k);
}

let errors = 0;
const usedCommon = new Set(appKeys);

function parity(label, pt, en) {
  const onlyPt = Object.keys(pt).filter((k) => !(k in en));
  const onlyEn = Object.keys(en).filter((k) => !(k in pt));
  if (onlyPt.length || onlyEn.length) {
    errors++;
    console.log(`✗ ${label}: paridade PT×EN quebrada`);
    if (onlyPt.length) console.log(`    só em pt: ${onlyPt.join(", ")}`);
    if (onlyEn.length) console.log(`    só em en: ${onlyEn.join(", ")}`);
  }
}
parity("common", common.pt, common.en);

const pages = [join(ROOT, "index.html"), ...walk(join(ROOT, "simulations"))].filter((p) => !p.includes("gerador-malha"));
for (const page of pages) {
  const html = readFileSync(page, "utf8");
  const dir = dirname(page);
  const rel = page.slice(ROOT.length + 1);
  const hasModule = /data-mm-i18n/.test(html);
  let mod = { pt: {}, en: {} };
  if (hasModule) {
    for (const l of ["pt", "en"]) {
      const f = join(dir, `i18n.${l}.json`);
      if (!existsSync(f)) { errors++; console.log(`✗ ${rel}: falta i18n.${l}.json`); continue; }
      mod[l] = read(f);
    }
    parity(rel, mod.pt, mod.en);
  } else if (existsSync(join(dir, "i18n.pt.json"))) {
    errors++; console.log(`✗ ${rel}: tem i18n.pt.json mas o body não declara data-mm-i18n`);
  }
  const used = keysInHtml(html);
  // Chaves usadas pelo Python do módulo (i18n_t("py.…"), etapa 4)
  const pyFile = join(dir, "simulation.py");
  if (existsSync(pyFile)) {
    const py = readFileSync(pyFile, "utf8");
    for (const m of py.matchAll(/i18n_t\(\s*"([^"]+)"/g)) used.add(m[1]);
    for (const m of py.matchAll(/i18n_t\(\s*'([^']+)'/g)) used.add(m[1]);
    // Chaves montadas em tempo de execução: i18n_t("prefixo." + var) → aceita qualquer chave com o prefixo
    for (const m of py.matchAll(/i18n_t\(\s*["']([^"']+\.)["']\s*\+/g)) {
      const prefix = m[1];
      Object.keys(mod.pt).concat(Object.keys(common.pt)).forEach((k) => { if (k.startsWith(prefix)) used.add(k); });
    }
  }
  // Chave terminada em ponto é prefixo de i18n_t("prefixo." + var), não chave literal
  [...used].filter((k) => k.endsWith(".")).forEach((k) => used.delete(k));
  const missing = [...used].filter((k) => !(k in mod.pt) && !(k in common.pt));
  if (missing.length) { errors++; console.log(`✗ ${rel}: chaves sem tradução: ${missing.join(", ")}`); }
  used.forEach((k) => { if (!(k in mod.pt)) usedCommon.add(k); });
  if (hasModule) {
    const orphans = Object.keys(mod.pt).filter((k) => !used.has(k));
    if (orphans.length) console.log(`! ${rel}: chaves do módulo não usadas no HTML: ${orphans.join(", ")}`);
  }
  // Elemento com data-i18n contendo math ou markup filho (seria apagado pelo textContent)
  for (const m of html.matchAll(/<([a-z0-9]+)[^>]*data-i18n="([^"]+)"[^>]*>([\s\S]*?)<\/\1>/g)) {
    if (/\$|<[a-z]/.test(m[3])) { errors++; console.log(`✗ ${rel}: data-i18n="${m[2]}" contém math ou markup filho`); }
  }
  console.log(`✓ ${rel}: ${used.size} chaves (${hasModule ? Object.keys(mod.pt).length + " do módulo" : "só comum"})`);
}
const orphanCommon = Object.keys(common.pt).filter((k) => !usedCommon.has(k));
if (orphanCommon.length) console.log(`! common: chaves não usadas em HTML/app.js: ${orphanCommon.join(", ")}`);
console.log(errors ? `\n${errors} erro(s)` : "\nOK: paridade e cobertura completas");
process.exit(errors ? 1 : 0);
