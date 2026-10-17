#!/usr/bin/env node
/**
 * Compara duas rodadas de scripts/baseline.mjs (v1.1.0, critério de aceite da spec, seção 2).
 *
 * Uso: node scripts/compare-baseline.mjs docs/v1.1.0/baseline/v1.0.0-hook.json docs/v1.1.0/baseline/<rodada>.json
 *
 * Critério: `__minervamesh_result` (result_sha256_16) idêntico e texto exibido idêntico.
 * Rodada sem o gancho (fechamento da v1.1.0, result null em todos os módulos): só texto e imagens.
 * Do texto é descontado o prefixo "título da figura + formatos da toolbar" que o gráfico
 * interativo (etapa 2b) acrescenta à área de resultado nos 6 módulos estáticos.
 * Hash de imagem é informativo: muda com qualquer rótulo traduzido ou troca GIF→canvas.
 */
import { readFileSync } from "node:fs";

const [refPath, newPath] = process.argv.slice(2);
if (!refPath || !newPath) { console.error("uso: compare-baseline.mjs <referencia.json> <rodada.json>"); process.exit(2); }
const ref = JSON.parse(readFileSync(refPath, "utf8")).results;
const cur = JSON.parse(readFileSync(newPath, "utf8")).results;

const TOOLBAR = /^.*?epsjpegpgfpdfpngpsrawsvgtifwebp\s*/;
const noHook = cur.every((x) => x.result_sha256_16 == null);
if (noHook) console.log("rodada sem o gancho _export_result: critério é texto e imagens\n");
const norm = (r) => ((r.text || "").replace(TOOLBAR, "") + " | " + (r.extra_text || "")).replace(/\s+/g, " ").trim();
const imgs = (r) => JSON.stringify((r.images || []).map((i) => i.sha256_16));

let bad = 0;
for (const r of ref) {
  const c = cur.find((x) => x.id === r.id);
  if (!c) { console.log(`${r.id.padEnd(22)} AUSENTE na rodada`); bad++; continue; }
  const text = norm(r) === norm(c);
  const result = noHook || r.result_sha256_16 === c.result_sha256_16;
  const imgEq = imgs(r) === imgs(c);
  if (!text || !result || (noHook && !imgEq) || c.failed) bad++;
  console.log(
    `${r.id.padEnd(22)} result ${noHook ? "(sem gancho)" : result ? "igual" : "DIFERE"}  texto ${text ? "igual" : "DIFERE"}  imagens ${imgEq ? "iguais" : "diferem"}` +
    `  run ${c.run_s}s${c.failed ? "  FALHOU " + c.failed : ""}${c.errors?.length ? "  ERR " + c.errors[0] : ""}`
  );
  if (!result && !noHook) console.log(`   result: ${r.result_sha256_16} -> ${c.result_sha256_16}`);
  if (!text) { console.log(`   ref: ${norm(r).slice(0, 400)}`); console.log(`   new: ${norm(c).slice(0, 400)}`); }
}
console.log(bad ? `\n${bad} divergência(s)` : "\nOK: resultado numérico e texto exibido idênticos");
process.exit(bad ? 1 : 0);
