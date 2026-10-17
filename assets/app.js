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
 *   Bloco de descrição: <details data-mm-desc>; cards [data-mm-params] e [data-mm-result] (layoutModule reorganiza tudo, v1.2.0).
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

// Fontes (v1.2.0, spec 4): IBM Plex Sans e Mono via Google Fonts, com fallback de
// sistema no tailwind.config.js. Injetadas aqui para não repetir três <link> em cada
// página. Sem SRI: o CSS do Google Fonts varia por navegador.
(function injectFonts() {
  if (document.querySelector('link[data-mm-fonts]')) return;
  const links = [
    { rel: "preconnect", href: "https://fonts.googleapis.com" },
    { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
    { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" },
  ];
  for (const attrs of links) {
    const l = document.createElement("link");
    Object.assign(l, attrs);
    l.dataset.mmFonts = "1";
    document.head.append(l);
  }
})();
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
  // Subtítulo do card de resultado (h3 da página) só aparece quando diz algo além de "Resultado"
  const sub = document.querySelector(".mm-result-sub");
  if (sub) sub.hidden = sub.textContent.trim() === t("view.result");
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
  // Marca (v1.2.0): quadrado com malha cruzada, mesmo desenho do favicon.ico
  mark:
    '<svg class="h-7 w-7 shrink-0 text-mm-accent" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="2" width="22" height="22" rx="2"/><path d="M2 13h22M13 2v22M2 2l22 22M24 2 2 24"/></svg>',
  sun:
    '<svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
  moon:
    '<svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/></svg>',
  external:
    '<svg class="h-3.5 w-3.5" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 2H2.5A.5.5 0 0 0 2 2.5v9a.5.5 0 0 0 .5.5h9a.5.5 0 0 0 .5-.5V9M8 2h4v4M12 2 6.5 7.5"/></svg>',
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
  const sw = el(`<div class="mm-lang flex items-center" role="group" data-i18n-attr="aria-label:lang.label">
        <button type="button" data-lang="pt">PT</button><span class="text-mm-border">|</span><button type="button" data-lang="en">EN</button>
      </div>`);
  sw.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  return sw;
}

// Botão claro/escuro (v1.2.0, spec 4): ícone do tema para o qual o clique leva;
// aria-label traduzido via data-i18n-attr, atualizado a cada troca. window.mmTheme vem de assets/theme.js.
function themeButton() {
  const btn = el(`<button type="button" class="mm-theme flex h-11 w-11 items-center justify-center rounded-md text-mm-muted hover:bg-mm-surface2 hover:text-mm-text"></button>`);
  const render = () => {
    const dark = window.mmTheme?.get() === "dark";
    const key = dark ? "nav.theme.light" : "nav.theme.dark";
    btn.innerHTML = dark ? ICONS.sun : ICONS.moon;
    btn.setAttribute("aria-pressed", String(dark));
    btn.dataset.i18nAttr = `aria-label:${key}`;
    btn.setAttribute("aria-label", t(key));
  };
  btn.addEventListener("click", () => window.mmTheme?.toggle());
  addEventListener("mm:theme", render);
  render();
  return btn;
}

const GITHUB = "https://github.com/lgsfarias/minervamesh";

// Marca: ícone + nome, link para o menu
function brand() {
  const menu = new URL("index.html", ROOT);
  return el(`<a href="${menu}" class="flex min-h-[44px] items-center gap-2.5 text-mm-text" aria-label="MinervaMesh">
        ${ICONS.mark}<span class="text-[17px] font-semibold tracking-tight">MinervaMesh</span></a>`);
}

// Cabeçalho das páginas de simulação (v1.2.0, spec 5.1): barra de largura total no topo do body,
// com a marca, "Todas as simulações", o irmão (escoamentos), PT | EN, tema e GitHub.
function injectHeader({ menu: isMenu = false } = {}) {
  const sibling = document.body.dataset.mmSibling; // "href|chave"
  let siblingHtml = "";
  if (isMenu) {
    // Sem a marca: o hero logo abaixo já tem o lockup grande (decisão do autor, 17/10/2026)
    const header = el(`<header class="border-b border-mm-border bg-mm-surface">
          <div class="mx-auto flex min-h-[56px] max-w-[1360px] flex-wrap items-center justify-end gap-x-4 gap-y-1 px-4 md:px-6">
            <div class="flex items-center gap-3">
              <a href="${GITHUB}" class="flex min-h-[44px] items-center gap-1.5 text-sm font-medium text-mm-accent hover:text-mm-accent-strong">GitHub ${ICONS.external}</a>
            </div>
          </div>
        </header>`);
    header.querySelector("div > div").prepend(langSwitcher(), themeButton());
    document.body.prepend(header);
    return;
  }
  if (sibling) {
    const [href, key] = sibling.split("|");
    siblingHtml = `<span class="text-mm-border" aria-hidden="true">|</span>
      <a href="${href}" class="flex min-h-[44px] items-center gap-1 font-medium text-mm-muted hover:text-mm-text">
        <span data-i18n="${key}"></span>${ICONS.arrowRight}</a>`;
  }
  const menu = new URL("index.html", ROOT);
  const header = el(`<header class="border-b border-mm-border bg-mm-surface">
        <div class="mx-auto flex min-h-[56px] max-w-7xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 md:px-6">
          <div class="flex min-w-0 flex-wrap items-center gap-x-4">
            <nav class="flex flex-wrap items-center gap-3 text-sm">
              <a href="${menu}" class="flex min-h-[44px] items-center gap-1 font-medium text-mm-muted hover:text-mm-text">
                ${ICONS.arrowLeft}<span data-i18n="nav.all"></span></a>
              ${siblingHtml}
            </nav>
          </div>
          <div class="flex items-center gap-3">
            <a href="${GITHUB}" class="hidden min-h-[44px] items-center gap-1.5 text-sm font-medium text-mm-accent hover:text-mm-accent-strong sm:flex">GitHub ${ICONS.external}</a>
          </div>
        </div>
      </header>`);
  header.querySelector("div > div").prepend(brand());
  const right = header.querySelector("div > div + div");
  right.prepend(langSwitcher(), themeButton());
  document.body.prepend(header);
}

// Rodapé (spec 5.1): crédito, afiliação em texto (o logo da Poli saiu na v1.2.0), código e licença.
function injectFooter() {
  document.body.append(
    el(`<footer class="mt-auto border-t border-mm-border bg-mm-surface">
          <div class="mx-auto flex max-w-7xl flex-wrap justify-between gap-x-6 gap-y-2 px-4 py-5 text-sm text-mm-muted md:px-6">
            <p><span data-i18n="site.name"></span> &middot; <span data-i18n="footer.developed_by"></span> &middot; <span data-i18n="footer.affiliation"></span></p>
            <p class="flex flex-wrap gap-x-4">
              <a href="${GITHUB}" class="text-mm-accent hover:text-mm-accent-strong" data-i18n="footer.source"></a>
              <a href="${GITHUB}/blob/main/LICENSE" class="text-mm-accent hover:text-mm-accent-strong" data-i18n="footer.license"></a>
            </p>
          </div>
        </footer>`)
  );
}

// Popup do código-fonte (v1.2.0, revisão 2 da etapa 4): abre em #show-code-btn; X, Esc e clique
// fora fecham; o foco vai para o X ao abrir e volta ao botão ao fechar. Prism (com números de linha)
// carrega na primeira abertura.
function injectCodeModal() {
  const modal = el(`<div id="codeModal" class="fixed inset-0 z-50 hidden items-center justify-center bg-black/60 p-3 md:p-6">
      <div role="dialog" aria-modal="true" aria-labelledby="mm-code-title" class="flex h-[88vh] w-full max-w-[1100px] flex-col overflow-hidden rounded-xl border border-mm-border bg-mm-surface text-mm-text shadow-2xl">
        <div class="flex items-center justify-between gap-4 border-b border-mm-border py-3 pl-5 pr-3">
          <div class="min-w-0">
            <h3 id="mm-code-title" class="text-base font-semibold" data-i18n="code.title"></h3>
            <p class="text-[13px] text-mm-muted"><span class="font-mono">simulation.py</span> &middot; <span data-i18n="code.comments_pt"></span></p>
          </div>
          <button id="closeModalBtn" type="button" class="mm-btn-ghost h-11 w-11 shrink-0 px-0" data-i18n-attr="aria-label:code.close">${ICONS.close}</button>
        </div>
        <div class="mm-code min-h-0 flex-1 overflow-auto"><pre class="line-numbers"><code class="language-python" id="sourceCode"></code></pre></div>
      </div>
    </div>`);
  document.body.append(modal);
  const btn = document.getElementById("show-code-btn");
  const show = () => {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    document.body.classList.add("overflow-hidden");
    modal.querySelector("#closeModalBtn").focus();
    loadSourceCode();
  };
  const hide = () => {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
    document.body.classList.remove("overflow-hidden");
    btn?.focus();
  };
  btn?.addEventListener("click", show);
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
  prismLinesJs: ["https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.js", "sha384-6QJu8apxMmB9TiPVWzYKF5pRgKcz7snO0/QU+MrWmgBLECQjoa6erxX2VQ5t41Jd"],
  prismLinesCss: ["https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css", "sha384-nUkTNLI8COlMCRJ0FHIdX76If83145OTCLUx4gQyfnO0gGeO/sD9czGEUBxtkcUv"],
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
    // Plugin de números de linha depois do núcleo e do componente python (v1.2.0)
    prismPromise = Promise.all([loadStyle(CDN.prismCss), loadStyle(CDN.prismLinesCss), loadScript(CDN.prismJs).then(() => loadScript(CDN.prismPython)).then(() => loadScript(CDN.prismLinesJs))]).catch((e) => {
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

// ---------------------------------------------------------------------------
// Layout da página de simulação (v1.2.0, spec 5): o app.js reorganiza o DOM a
// partir dos marcadores da página (h2 do título, <details data-mm-desc>,
// [data-mm-params], [data-mm-result]) em: linha de título com badges, faixa de
// resumo (equação governante + esquema), painel lateral com abas Parâmetros /
// Problema / Código e card de resultado largo. Os ids que o Python usa
// (#run-btn, #stop-btn, #show-code-btn, #plot-output, inputs) só mudam de lugar.
// Página sem os marcadores fica com o layout antigo.
// ---------------------------------------------------------------------------
function layoutModule(app) {
  const params = app.querySelector("[data-mm-params]");
  const result = app.querySelector("[data-mm-result]");
  if (!params || !result) return false;
  const desc = app.querySelector("details[data-mm-desc]");
  const title = app.querySelector('h2[data-i18n="title"]') || app.querySelector("h2");

  // Scripts inline do módulo já rodaram; vão para o fim do body para não segurar contêineres vazios
  app.querySelectorAll('script:not([type="py"])').forEach((s) => document.body.append(s));

  // 1. Título + badges (preenchidas por fillBadges com assets/menu.json)
  const titleRow = el(`<div class="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2"><div id="mm-badges" class="flex flex-wrap gap-1.5"></div></div>`);
  if (title) {
    title.className = "text-[26px] font-semibold leading-tight tracking-tight text-mm-text";
    titleRow.prepend(title);
  }

  // 2. Faixa de resumo: [data-mm-summary] explícito ou clone do primeiro $$…$$ da seção da equação, mais os esquemas
  const summary = el(`<section class="mm-summary" aria-labelledby="mm-summary-title">
        <div class="mm-summary-main">
          <p id="mm-summary-title" class="mm-leg" data-i18n="desc.sec.equation"></p>
          <div class="mm-summary-eq"></div>
          <button type="button" id="mm-full-problem" class="mm-link"><span data-i18n="summary.full_problem"></span>${ICONS.arrowRight}</button>
        </div>
        <div class="mm-summary-scheme"></div>
      </section>`);
  const eqBox = summary.querySelector(".mm-summary-eq");
  const explicit = app.querySelector("[data-mm-summary]");
  if (explicit) eqBox.append(explicit);
  else if (desc) {
    const eqSection = [...desc.querySelectorAll("section")].find((s) => s.querySelector('[data-i18n="desc.sec.equation"]')) || desc;
    const math = [...eqSection.querySelectorAll("div")].find((d) => /\$\$/.test(d.textContent));
    if (math) eqBox.append(math.cloneNode(true));
  }
  const schemeBox = summary.querySelector(".mm-summary-scheme");
  desc?.querySelectorAll("figure.mm-scheme").forEach((fig) => {
    const c = fig.cloneNode(true); // mantém data-mm-scheme: a troca por cenário da página alcança o clone
    c.className = "mm-scheme" + (fig.classList.contains("hidden") ? " hidden" : "");
    schemeBox.append(c);
  });
  if (!schemeBox.children.length) schemeBox.remove();
  if (!eqBox.children.length) summary.querySelector("#mm-summary-title").remove();

  // 3. Painel lateral: só os parâmetros (decisão do autor em 17/10/2026: sem abas; o problema vai para
  //    a coluna larga e o código para um popup). O h3 "Parâmetros" da página vira o cabeçalho do painel.
  const aside = el(`<aside class="mm-aside" aria-labelledby="mm-params-title">
        <div class="mm-aside-head"></div>
        <div class="mm-panel"></div>
      </aside>`);
  const h3p = params.querySelector(":scope > h3");
  if (h3p) {
    h3p.className = "text-[15px] font-semibold text-mm-text";
    h3p.id = "mm-params-title";
    aside.querySelector(".mm-aside-head").append(h3p);
  } else aside.querySelector(".mm-aside-head").append(el(`<h3 id="mm-params-title" class="text-[15px] font-semibold text-mm-text" data-i18n="section.params"></h3>`));
  params.removeAttribute("class");
  params.classList.add("mm-params");
  aside.querySelector(".mm-panel").append(params);
  // "Ver código-fonte" fica abaixo do Executar, como botão secundário; abre o popup (injectCodeModal)
  const codeBtn = params.querySelector("#show-code-btn");
  if (codeBtn) {
    const group = codeBtn.parentElement;
    codeBtn.className = "mm-btn-ghost w-full";
    if (group && group !== params && [...group.children].every((c) => c.tagName === "BUTTON")) {
      // Sai da linha de botões: Executar fica sozinho (largura total) e o status entra entre os dois
      group.after(codeBtn);
      codeBtn.classList.add("mt-3");
      if (!group.children.length) group.remove();
    }
  }
  // Executar (e Parar, no NS) com a mesma classe em todas as páginas: as 10 traziam variantes
  // (flex-1, w-full, justify-center sem largura…). O "hidden"/"flex" do Parar é alternado pelo Python.
  const runBtn = params.querySelector("#run-btn");
  if (runBtn) {
    runBtn.className = "mm-btn-primary";
    const g = runBtn.parentElement;
    if (g !== params && [...g.children].every((c) => c.tagName === "BUTTON")) g.className = "mm-actions";
    else runBtn.classList.add("mt-4");
  }
  const stopBtn = params.querySelector("#stop-btn");
  if (stopBtn) stopBtn.className = "mm-btn-danger" + (stopBtn.classList.contains("hidden") ? " hidden" : "");

  // 4. Coluna larga: card com o seletor Resultado | Problema no cabeçalho. O conteúdo do antigo
  //    <details> vira o painel Problema, com a largura que o texto e os esquemas pedem.
  const resultCard = el(`<section class="mm-result">
        <div class="mm-result-head">
          <div class="mm-seg" role="tablist">
            <button type="button" role="tab" id="mm-view-result" aria-controls="mm-panel-result" aria-selected="true"><span data-i18n="view.result"></span></button>
            <button type="button" role="tab" id="mm-view-problem" aria-controls="mm-panel-problem" aria-selected="false" tabindex="-1"><span data-i18n="view.problem"></span></button>
          </div>
        </div>
        <div class="mm-result-body" role="tabpanel" id="mm-panel-result" aria-labelledby="mm-view-result"></div>
        <div class="mm-problem mm-prose" role="tabpanel" id="mm-panel-problem" aria-labelledby="mm-view-problem" hidden></div>
      </section>`);
  const body = resultCard.querySelector(".mm-result-body");
  const h3 = result.querySelector(":scope > h3, :scope > h2");
  if (h3) h3.className = "mm-result-sub";
  [...result.childNodes].forEach((n) => body.append(n));
  const out = body.querySelector("#plot-output");
  if (out) out.className = "mm-plot";
  const pProblem = resultCard.querySelector("#mm-panel-problem");
  if (desc) {
    const inner = desc.querySelector(":scope > div") || desc;
    [...inner.childNodes].forEach((n) => pProblem.append(n));
    desc.remove();
  } else resultCard.querySelector(".mm-seg").remove();

  // 5. Montagem; contêineres antigos que ficaram vazios saem
  const main = el(`<div class="mm-main"></div>`);
  main.append(aside, resultCard);
  [...app.children].forEach((c) => {
    if (c === title) return;
    if (!c.textContent.trim() && !c.querySelector("input, select, button, canvas, img, svg, [id]")) c.remove();
  });
  app.append(titleRow, summary, main);

  // Seletor Resultado | Problema: clique, setas; o link da faixa abre o problema; mmProgress.start volta ao resultado
  const views = [...resultCard.querySelectorAll('[role="tab"]')];
  const select = (name, focus = false) => {
    for (const tab of views) {
      const on = tab.id === `mm-view-${name}`;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !on;
      if (on && focus) tab.focus();
    }
  };
  views.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab.id.replace("mm-view-", "")));
    tab.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      select(views[(i + d + views.length) % views.length].id.replace("mm-view-", ""), true);
    });
  });
  showResultView = () => select("result");
  summary.querySelector("#mm-full-problem").addEventListener("click", () => {
    if (!desc) return;
    // Seleciona o problema, rola o card até o topo da janela e leva o foco ao seletor (pedido do autor, 17/10/2026)
    select("problem", true);
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  return true;
}

// Definida por layoutModule: mmProgress.start() chama para o card voltar ao resultado quando uma simulação começa
let showResultView = null;

// Badges do título (método, dimensão, regime) a partir de assets/menu.json, pela pasta do módulo
function fillBadges(menu) {
  const box = document.getElementById("mm-badges");
  const m = (menu?.modules || []).find((x) => x.module === document.body.dataset.mmModule);
  if (!box || !m) return;
  box.innerHTML = [
    `<span class="mm-tag mm-tag-${m.method}" data-i18n="menu.tag.${m.method}"></span>`,
    `<span class="mm-tag" data-i18n="menu.tag.${m.dim}"></span>`,
    `<span class="mm-tag" data-i18n="menu.tag.${m.regime}"></span>`,
  ].join("");
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
        <div class="h-1.5 w-full overflow-hidden rounded bg-mm-border"><div id="mm-status-bar" class="h-full w-0 rounded bg-mm-accent transition-[width] duration-300"></div></div>
        <div class="mt-1 flex justify-between gap-2 text-xs text-mm-muted"><span id="mm-status-text"></span><span id="mm-status-time" class="shrink-0 font-mono tabular-nums"></span></div>
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
  bar.classList.toggle("bg-mm-ok", done);
  bar.classList.toggle("bg-mm-accent", !done);
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
    box = el(`<div id="mm-run" class="mb-4 hidden rounded-lg border border-mm-accent bg-mm-accent-soft p-4 text-left" aria-live="polite">
        <div class="flex items-center gap-4">
          <span class="mm-run-spinner" aria-hidden="true"></span>
          <div class="min-w-0 flex-1">
            <div class="flex items-baseline justify-between gap-3 font-medium text-mm-accent"><span id="mm-run-text"></span><span id="mm-run-time" class="shrink-0 font-mono tabular-nums text-sm"></span></div>
            <div class="mt-2 h-2 w-full overflow-hidden rounded bg-mm-accent-soft"><div id="mm-run-bar" class="h-full w-0 rounded bg-mm-accent transition-[width] duration-300"></div></div>
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
    showResultView?.(); // card volta ao resultado se o problema estava aberto
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
  if (app) {
    // Cabeçalho e rodapé são barras de largura total no body; o conteúdo do módulo
    // fica em #mm-app com o padding que antes era do body (v1.2.0, etapa 3).
    document.body.classList.add("flex", "flex-col");
    app.classList.add("w-full", "flex-1", "px-4", "py-6", "md:px-6", "md:py-8");
    injectHeader();
    injectFooter();
    layoutModule(app); // antes do MathJax: a equação clonada para a faixa é tipografada uma vez só
    injectCodeModal();
  }
  injectMathJax();
  const [, menu] = await Promise.all([loadStrings(lang), fetchJson(new URL("assets/menu.json", ROOT))]);
  fillBadges(menu);
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
// Cards do menu (v1.2.0, spec 6): um por módulo de assets/menu.json; título e descrição
// pelas chaves menu.<id>.title/.desc; miniatura em assets/thumbs/<id>-sm.png (scripts/thumbs.mjs).
async function buildMenu() {
  const grid = document.getElementById("mm-grid");
  if (!grid) return;
  const data = await fetchJson(new URL("assets/menu.json", ROOT));
  for (const m of data.modules || []) {
    const href = new URL(m.path, ROOT);
    const img = new URL(`assets/thumbs/${m.id}-sm.png`, ROOT);
    grid.append(
      el(`<a href="${href}" class="mm-card" data-method="${m.method}">
          <div class="mm-card-thumb"><img src="${img}" alt="" width="480" height="360" loading="lazy" /></div>
          <div class="mm-card-body">
            <div class="flex flex-wrap gap-1.5">
              <span class="mm-tag mm-tag-${m.method}" data-i18n="menu.tag.${m.method}"></span>
              <span class="mm-tag" data-i18n="menu.tag.${m.dim}"></span>
              <span class="mm-tag" data-i18n="menu.tag.${m.regime}"></span>
            </div>
            <h2 class="mm-card-title" data-i18n="menu.${m.id}.title"></h2>
            <p class="mm-card-desc" data-i18n="menu.${m.id}.desc"></p>
          </div>
        </a>`)
    );
  }
  // Chips: filtram por método; aria-pressed marca o ativo
  const chips = [...document.querySelectorAll("#mm-filters .mm-chip")];
  for (const chip of chips) {
    chip.addEventListener("click", () => {
      const method = chip.dataset.method;
      chips.forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
      grid.querySelectorAll(".mm-card").forEach((card) => card.classList.toggle("hidden", method !== "all" && card.dataset.method !== method));
    });
  }
}

async function bootMenu() {
  document.body.classList.add("flex", "flex-col");
  injectHeader({ menu: true });
  await buildMenu();
  injectFooter();
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
