/**
 * MinervaMesh — camada compartilhada das páginas de simulação.
 *
 * Injeta o que antes era copiado em cada index.html: cabeçalho (voltar ao menu,
 * seletor de idioma, logo), modal de código-fonte, rodapé, o bloco de status
 * (barra de progresso + texto) abaixo do botão Executar, o MathJax e o Prism
 * (este só quando o modal de código abre). O PyScript fica no HTML. Também aplica as
 * strings traduzíveis (data-i18n) e expõe window.i18n e window.mmProgress para o Python.
 *
 * Contrato com a página:
 *   <script type="module" src="…/assets/app.js"></script>  (antes do core.js do PyScript)
 *   <body data-mm-module="<nome da pasta>" [data-mm-sibling="<href>|<chave nav.goto.*>"]>
 *     <div id="mm-app" class="max-w-5xl mx-auto"> … conteúdo do módulo … </div>
 *   Ids esperados no conteúdo: #run-btn, #show-code-btn, #plot-output.
 *   Bloco de descrição: <details data-mm-desc> (estado lembrado em localStorage, minervamesh.desc.<módulo>).
 *   Strings do módulo: <body data-mm-i18n data-mm-title="title"> carrega ./i18n.<lang>.json
 *   (chaves do módulo têm precedência sobre as de assets/i18n/common.<lang>.json).
 *
 * Menu (index.html da raiz): página sem <script type="py">. Recebe só strings,
 * seletor de idioma (em #mm-lang-slot) e rodapé; sem status nem cabeçalho de módulo.
 *
 * Contrato com o Python (simulation.py):
 *   js.i18n.t("py.…") para rótulos de gráfico e mensagens (etapa 4 da v1.1.0)
 *   js.mmProgress.start(modo, total) / .step(i, t) / .encoding() / .done() / .fail()
 *     modo "spinner" (solução única), "counter" (marcha com GIF) ou "inline" (frames ao vivo)
 */

const ROOT = new URL("../", import.meta.url); // app.js vive em assets/
const DEFAULT_LANG = "en"; // inglês por padrão (spec 5.1); ?lang=pt ou o seletor trocam
const LANGS = ["pt", "en"];
const LS_LANG = "minervamesh.lang";

// ---------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------
const strings = { common: {}, module: {} };
let lang = resolveLang();

function resolveLang() {
  const fromUrl = new URLSearchParams(location.search).get("lang");
  if (LANGS.includes(fromUrl)) return fromUrl;
  try {
    const saved = localStorage.getItem(LS_LANG);
    if (LANGS.includes(saved)) return saved;
  } catch {}
  return DEFAULT_LANG;
}

async function fetchJson(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : {};
  } catch {
    return {};
  }
}

async function loadStrings(l) {
  const [common, mod] = await Promise.all([
    fetchJson(new URL(`assets/i18n/common.${l}.json`, ROOT)),
    // Strings do módulo só quando a página declara data-mm-i18n (etapa 3), para não gerar 404
    "mmI18n" in document.body.dataset ? fetchJson(new URL(`i18n.${l}.json`, location.href)) : {},
  ]);
  strings.common = common;
  strings.module = mod;
}

function t(key) {
  if (key in strings.module) return strings.module[key];
  if (key in strings.common) return strings.common[key];
  return `[${key}]`;
}

function fmt(key, params) {
  let out = t(key);
  for (const [k, v] of Object.entries(params || {})) out = out.replaceAll(`{${k}}`, String(v));
  return out;
}

function applyStrings() {
  document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-attr]").forEach((el) => {
    // formato: "attr:chave;attr2:chave2"
    el.dataset.i18nAttr.split(";").forEach((pair) => {
      const [attr, key] = pair.split(":").map((s) => s.trim());
      if (attr && key) el.setAttribute(attr, t(key));
    });
  });
  const titleKey = document.body.dataset.mmTitle;
  if (titleKey && titleKey in { ...strings.common, ...strings.module }) {
    document.title = `${t("site.name")} · ${t(titleKey)}`;
  }
  document.querySelectorAll(".mm-lang button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  });
}

async function setLang(l) {
  if (!LANGS.includes(l)) return;
  lang = l;
  try {
    localStorage.setItem(LS_LANG, l);
  } catch {}
  await loadStrings(l);
  applyStrings();
  document.dispatchEvent(new CustomEvent("mm:langchange", { detail: { lang: l } }));
}

window.i18n = {
  get lang() {
    return lang;
  },
  t,
  setLang,
  apply: applyStrings,
};

// ---------------------------------------------------------------------------
// Ícones (SVG inline, substituem o Font Awesome)
// ---------------------------------------------------------------------------
const ICONS = {
  arrowLeft:
    '<svg class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M17 10a.75.75 0 0 1-.75.75H5.612l4.158 3.96a.75.75 0 1 1-1.04 1.08l-5.5-5.25a.75.75 0 0 1 0-1.08l5.5-5.25a.75.75 0 1 1 1.04 1.08L5.612 9.25H16.25A.75.75 0 0 1 17 10Z" clip-rule="evenodd"/></svg>',
  arrowRight:
    '<svg class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M3 10a.75.75 0 0 1 .75-.75h10.638L10.23 5.29a.75.75 0 1 1 1.04-1.08l5.5 5.25a.75.75 0 0 1 0 1.08l-5.5 5.25a.75.75 0 1 1-1.04-1.08l4.158-3.96H3.75A.75.75 0 0 1 3 10Z" clip-rule="evenodd"/></svg>',
  code:
    '<svg class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M6.28 5.22a.75.75 0 0 1 0 1.06L2.56 10l3.72 3.72a.75.75 0 0 1-1.06 1.06L.97 10.53a.75.75 0 0 1 0-1.06l4.25-4.25a.75.75 0 0 1 1.06 0Zm7.44 0a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L17.44 10l-3.72-3.72a.75.75 0 0 1 0-1.06ZM11.377 2.011a.75.75 0 0 1 .612.867l-2.5 14.5a.75.75 0 0 1-1.478-.255l2.5-14.5a.75.75 0 0 1 .866-.612Z" clip-rule="evenodd"/></svg>',
  close:
    '<svg class="h-6 w-6" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z"/></svg>',
};

// ---------------------------------------------------------------------------
// Markup injetado
// ---------------------------------------------------------------------------
function el(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html.trim();
  return tpl.content.firstElementChild;
}

function langSwitcher() {
  const sw = el(`<div class="mm-lang flex items-center gap-1" role="group" data-i18n-attr="aria-label:lang.label">
        <button type="button" data-lang="pt">PT</button><span class="text-gray-300">|</span><button type="button" data-lang="en">EN</button>
      </div>`);
  sw.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  return sw;
}

function injectHeader(app) {
  const sibling = document.body.dataset.mmSibling; // "href|chave"
  let siblingHtml = "";
  if (sibling) {
    const [href, key] = sibling.split("|");
    siblingHtml = `<span class="text-gray-300">|</span>
      <a href="${href}" class="flex items-center gap-1 text-gray-600 hover:text-blue-800 font-medium">
        <span data-i18n="${key}"></span>${ICONS.arrowRight}</a>`;
  }
  const logo = new URL("assets/logo_poli_ufrj.png", ROOT);
  const menu = new URL("index.html", ROOT);
  app.prepend(
    el(`<header class="mb-8 flex items-center justify-between gap-4">
          <nav class="flex items-center gap-3 text-sm">
            <a href="${menu}" class="flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium">
              ${ICONS.arrowLeft}<span data-i18n="nav.back"></span></a>
            ${siblingHtml}
          </nav>
          <div class="flex items-center gap-5">
            <img src="${logo}" class="h-10 w-auto opacity-80" data-i18n-attr="alt:logo.alt" />
          </div>
        </header>`)
  );
  app.querySelector("header > div").prepend(langSwitcher());
}

function injectFooter(app) {
  // No menu o #mm-app fica dentro de uma linha flex centralizada; o rodapé vai no body (flex-col).
  const anchor = document.querySelector('script[type="py"]') ? app : document.body;
  const place = (node) => (anchor === app ? app.after(node) : document.body.append(node));
  place(
    el(`<footer class="mt-8 w-full pb-6 text-center text-sm text-gray-500 opacity-80">
          <span data-i18n="site.name"></span> &middot; <span data-i18n="footer.developed_by"></span> &middot;
          <a href="https://github.com/lgsfarias/minervamesh" class="underline hover:text-gray-700" data-i18n="footer.source"></a>
          &middot; <span data-i18n="footer.license"></span>
        </footer>`)
  );
}

function injectCodeModal() {
  const modal = el(`<div id="codeModal" class="fixed inset-0 z-50 hidden items-center justify-center bg-black bg-opacity-50 p-4">
      <div class="max-h-[85vh] w-full max-w-5xl overflow-y-auto rounded-lg bg-white p-6 shadow-2xl">
        <div class="mb-4 flex items-center justify-between border-b pb-2">
          <h3 class="text-xl font-bold text-gray-800" data-i18n="code.title"></h3>
          <button id="closeModalBtn" type="button" class="text-gray-500 hover:text-gray-800" data-i18n-attr="aria-label:code.close">${ICONS.close}</button>
        </div>
        <p class="mb-3 text-sm text-gray-500" data-i18n="code.comments_pt"></p>
        <pre class="text-sm"><code class="language-python" id="sourceCode"></code></pre>
      </div>
    </div>`);
  document.body.append(modal);
  const show = () => {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    loadSourceCode();
  };
  const hide = () => {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  };
  document.getElementById("show-code-btn")?.addEventListener("click", show);
  modal.querySelector("#closeModalBtn").addEventListener("click", hide);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) hide();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) hide();
  });
}

async function loadSourceCode() {
  const code = document.getElementById("sourceCode");
  code.textContent = t("code.loading");
  try {
    const [, src] = await Promise.all([loadPrism(), fetch(new URL("simulation.py", location.href), { cache: "no-store" }).then((r) => r.text())]);
    code.textContent = src;
    if (window.Prism) Prism.highlightElement(code);
  } catch (e) {
    code.textContent = String(e);
  }
}

// ---------------------------------------------------------------------------
// Bibliotecas de CDN (um lugar só para versão e hash SRI; antes copiadas em cada index.html)
// ---------------------------------------------------------------------------
const CDN = {
  prismCss: ["https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css", "sha384-wFjoQjtV1y5jVHbt0p35Ui8aV8GVpEZkyF99OXWqP/eNJDU93D3Ugxkoyh6Y2I4A"],
  prismJs: ["https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js", "sha384-06z5D//U/xpvxZHuUz92xBvq3DqBBFi7Up53HRrbV7Jlv7Yvh/MZ7oenfUe9iCEt"],
  prismPython: ["https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js", "sha384-WJdEkJKrbsqw0evQ4GB6mlsKe5cGTxBOw4KAEIa52ZLB7DDpliGkwdme/HMa5n1m"],
  mathjax: ["https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js", "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"],
};

function loadScript([src, integrity], attrs = {}) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    Object.assign(el, { src, integrity, crossOrigin: "anonymous", onload: resolve, onerror: () => reject(new Error(`falha ao carregar ${src}`)) }, attrs);
    document.head.append(el);
  });
}

function loadStyle([href, integrity]) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("link");
    Object.assign(el, { rel: "stylesheet", href, integrity, crossOrigin: "anonymous", onload: resolve, onerror: () => reject(new Error(`falha ao carregar ${href}`)) });
    document.head.append(el);
  });
}

// Prism só é usado no modal "Ver código": carregado no primeiro clique, uma vez.
// O componente python depende do núcleo, por isso os dois são sequenciais.
let prismPromise = null;
function loadPrism() {
  if (!prismPromise) {
    prismPromise = Promise.all([loadStyle(CDN.prismCss), loadScript(CDN.prismJs).then(() => loadScript(CDN.prismPython))]).catch((e) => {
      console.warn(e); // sem realce, o código aparece em texto puro
    });
  }
  return prismPromise;
}

// MathJax (spec 7.3): configuração antes do script; só nas páginas de simulação.
// Delimitadores $…$ e $$…$$; o texto traduzível nunca divide elemento com math (spec 5.2).
function injectMathJax() {
  if (window.MathJax) return;
  window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\(", "\\)"]], displayMath: [["$$", "$$"], ["\\[", "\\]"]], processEscapes: true } };
  loadScript(CDN.mathjax, { id: "MathJax-script", async: true }).catch((e) => console.warn(e));
}

function decorateButtons() {
  const showCode = document.getElementById("show-code-btn");
  if (showCode && !showCode.querySelector("svg")) {
    showCode.classList.add("inline-flex", "items-center", "justify-center", "gap-2");
    showCode.insertAdjacentHTML("afterbegin", ICONS.code);
    const label = showCode.querySelector("[data-i18n]");
    if (!label) {
      const span = document.createElement("span");
      span.dataset.i18n = "btn.show_code";
      showCode.append(span);
    }
  }
}

// ---------------------------------------------------------------------------
// Progresso (spec 6): #mm-status, discreto, abaixo dos botões (carga inicial e
// tempo total); #mm-run, chamativo, acima de #plot-output durante a simulação.
// #mm-run fica fora de #plot-output porque o Python reescreve o innerHTML da área.
// ---------------------------------------------------------------------------
const progress = { mode: null, total: 0, t0: 0, last: 0, timer: null };

function setRunBusy(busy, labelKey) {
  const btn = document.getElementById("run-btn");
  if (!btn) return;
  btn.disabled = busy;
  btn.setAttribute("aria-busy", String(busy));
  btn.classList.toggle("mm-busy", busy);
  let spin = btn.querySelector(".mm-btn-spinner");
  if (busy && !spin) btn.prepend(el('<span class="mm-btn-spinner" aria-hidden="true"></span>'));
  if (!busy && spin) spin.remove();
  const label = btn.querySelector("[data-i18n]");
  if (label) {
    label.dataset.i18n = busy ? labelKey : "btn.run";
    label.textContent = t(label.dataset.i18n);
  }
}

function statusEl() {
  let box = document.getElementById("mm-status");
  if (!box) {
    const btn = document.getElementById("run-btn");
    if (!btn) return null;
    box = el(`<div id="mm-status" class="mt-3 hidden" aria-live="polite">
        <div class="h-1.5 w-full overflow-hidden rounded bg-gray-200"><div id="mm-status-bar" class="h-full w-0 rounded bg-blue-600 transition-[width] duration-300"></div></div>
        <div class="mt-1 flex justify-between gap-2 text-xs text-gray-600"><span id="mm-status-text"></span><span id="mm-status-time" class="shrink-0 font-mono tabular-nums"></span></div>
      </div>`);
    const group = btn.parentElement;
    const anchor = [...group.children].every((c) => c.tagName === "BUTTON") ? group : btn;
    anchor.after(box);
  }
  return box;
}

// frac: fração concluída (0–1) ou null para barra indeterminada
function statusShow(text, frac, time = "", done = false) {
  const box = statusEl();
  if (!box) return;
  box.classList.remove("hidden");
  const bar = box.querySelector("#mm-status-bar");
  bar.classList.toggle("mm-bar-indeterminate", frac == null);
  bar.classList.toggle("bg-green-600", done);
  bar.classList.toggle("bg-blue-600", !done);
  bar.style.width = frac == null ? "40%" : `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  box.querySelector("#mm-status-text").textContent = text;
  box.querySelector("#mm-status-time").textContent = time;
}

function statusHide() {
  document.getElementById("mm-status")?.classList.add("hidden");
}

function runEl() {
  let box = document.getElementById("mm-run");
  if (!box) {
    const out = document.getElementById("plot-output");
    if (!out) return null;
    box = el(`<div id="mm-run" class="mb-4 hidden rounded-lg border border-blue-200 bg-blue-50 p-4 text-left" aria-live="polite">
        <div class="flex items-center gap-4">
          <span class="mm-run-spinner" aria-hidden="true"></span>
          <div class="min-w-0 flex-1">
            <div class="flex items-baseline justify-between gap-3 font-medium text-blue-800"><span id="mm-run-text"></span><span id="mm-run-time" class="shrink-0 font-mono tabular-nums text-sm"></span></div>
            <div class="mt-2 h-2 w-full overflow-hidden rounded bg-blue-100"><div id="mm-run-bar" class="h-full w-0 rounded bg-blue-600 transition-[width] duration-300"></div></div>
          </div>
        </div>
      </div>`);
    out.before(box);
  }
  return box;
}

function runShow(text, frac, time = "") {
  const box = runEl();
  if (!box) return;
  box.classList.remove("hidden");
  const bar = box.querySelector("#mm-run-bar");
  bar.classList.toggle("mm-bar-indeterminate", frac == null);
  bar.style.width = frac == null ? "40%" : `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  box.querySelector("#mm-run-text").textContent = text;
  box.querySelector("#mm-run-time").textContent = time;
}

function runHide() {
  document.getElementById("mm-run")?.classList.add("hidden");
  document.getElementById("plot-output")?.classList.remove("mm-dim");
}

const elapsed = () => ((performance.now() - progress.t0) / 1000).toFixed(1);

function progressReset() {
  if (progress.timer) clearInterval(progress.timer);
  progress.timer = null;
  progress.mode = null;
}

function progressRenderInline() {
  runShow(fmt("sim.progress.step", { i: progress.last, n: progress.total }), progress.total ? progress.last / progress.total : null, `${elapsed()} s`);
}

window.mmProgress = {
  start(mode = "spinner", total = 0) {
    progressReset();
    progress.mode = mode;
    progress.total = Number(total) || 0;
    progress.t0 = performance.now();
    progress.last = 0;
    setRunBusy(true, "btn.running");
    statusHide();
    if (mode === "inline") {
      progressRenderInline();
      progress.timer = setInterval(progressRenderInline, 1000);
      return;
    }
    document.getElementById("plot-output")?.classList.add("mm-dim");
    if (mode === "counter") runShow(fmt("sim.progress.step", { i: 0, n: progress.total }), 0, "t = 0.000 s");
    else runShow(t("sim.running"), null);
  },
  step(i, tsim = 0) {
    if (progress.mode === "counter") {
      runShow(fmt("sim.progress.step", { i, n: progress.total }), progress.total ? Number(i) / progress.total : null, `t = ${Number(tsim).toFixed(3)} s`);
    } else if (progress.mode === "inline") {
      progress.last = Math.min(Number(i), progress.total);
      progressRenderInline();
    }
  },
  encoding() {
    if (progress.mode === "counter") runShow(t("sim.encoding"), null);
  },
  done() {
    if (!progress.mode) return;
    progressReset();
    setRunBusy(false);
    runHide();
    statusShow(t("sim.total_time"), 1, `${elapsed()} s`, true);
  },
  // Simulação lançou exceção: volta ao estado de repouso sem "Tempo total".
  // Zera o modo, então o done() do finally em seguida é no-op. Sem guarda de modo:
  // a exceção pode vir antes do start() (ex.: parâmetro inválido).
  fail() {
    progressReset();
    setRunBusy(false);
    runHide();
    statusHide();
  },
};

// Cronômetro desde o início da navegação (performance.now()): inclui o download
// iniciado antes de este módulo rodar. Sem porcentagem: o PyScript não expõe progresso.
let loadingTimer = null;
function loadingStart() {
  setRunBusy(true, "btn.preparing");
  const tick = () => statusShow(t("loading.status"), null, `${Math.floor(performance.now() / 1000)} s`);
  tick();
  loadingTimer = setInterval(tick, 1000);
}
function loadingStop() {
  if (loadingTimer) clearInterval(loadingTimer);
  loadingTimer = null;
  setRunBusy(false);
  statusHide();
}

// Bloco "Sobre o problema" (<details data-mm-desc>, spec 7.1): aberto/fechado lembrado por módulo.
const LS_DESC = "minervamesh.desc.";
function initDescMemory() {
  const d = document.querySelector("details[data-mm-desc]");
  if (!d) return;
  const key = LS_DESC + (document.body.dataset.mmModule || location.pathname);
  try {
    const saved = localStorage.getItem(key);
    if (saved === "0" || saved === "1") d.open = saved === "1";
  } catch {}
  d.addEventListener("toggle", () => {
    try {
      localStorage.setItem(key, d.open ? "1" : "0");
    } catch {}
  });
}

// ---------------------------------------------------------------------------
// Ciclo de vida
// ---------------------------------------------------------------------------
async function boot() {
  const app = document.getElementById("mm-app");
  if (!document.querySelector('script[type="py"]')) return bootMenu();
  // Backend interativo do Matplotlib (plt.show()) desenha dentro deste elemento em vez do body
  const out = document.getElementById("plot-output");
  if (out) {
    document.pyodideMplTarget = out;
    // A toolbar nasce com "eps" selecionado (ordem alfabética); png é o que o aluno espera
    new MutationObserver(() => {
      out.querySelectorAll("select.mpl-widget").forEach((sel) => {
        if (!sel.dataset.mmDefault) {
          sel.dataset.mmDefault = "1";
          if ([...sel.options].some((o) => o.value === "png")) sel.value = "png";
        }
      });
    }).observe(out, { childList: true, subtree: true });
  }
  injectMathJax();
  if (app) {
    injectHeader(app);
    injectFooter(app);
    injectCodeModal();
    decorateButtons();
  }
  initDescMemory();
  await loadStrings(lang);
  applyStrings();

  const onReady = () => {
    loadingStop();
    const out = document.getElementById("plot-output");
    if (out && out.textContent.trim() === "" && out.children.length === 0) {
      const ph = document.createElement("span");
      ph.dataset.i18n = "output.placeholder";
      ph.textContent = t("output.placeholder");
      out.append(ph);
    }
  };
  if (window.__pyReadyFired) onReady();
  else {
    loadingStart();
    addEventListener("py:ready", onReady, { once: true });
  }
}

// Menu: sem PyScript, logo sem dialog de carga nem cabeçalho de módulo.
async function bootMenu() {
  document.getElementById("mm-lang-slot")?.append(langSwitcher());
  const app = document.getElementById("mm-app");
  if (app) injectFooter(app);
  await loadStrings(lang);
  applyStrings();
}

// O listener é registrado no carregamento do módulo (antes do core.js) para
// não perder o py:ready caso o PyScript fique pronto antes do boot terminar.
addEventListener("py:ready", () => {
  window.__pyReadyFired = true;
}, { once: true });

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
