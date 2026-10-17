/**
 * MinervaMesh — camada compartilhada das páginas de simulação.
 *
 * Injeta o que antes era copiado em cada index.html: dialog de carga do
 * PyScript, dialog de execução (#sim-loading), cabeçalho (voltar ao menu,
 * seletor de idioma, logo), modal de código-fonte e rodapé. Também aplica as
 * strings traduzíveis (data-i18n) e expõe window.i18n para o Python.
 *
 * Contrato com a página:
 *   <script type="module" src="…/assets/app.js"></script>  (antes do core.js do PyScript)
 *   <body data-mm-module="<nome da pasta>" [data-mm-sibling="<href>|<chave nav.goto.*>"]>
 *     <div id="mm-app" class="max-w-5xl mx-auto"> … conteúdo do módulo … </div>
 *   Ids esperados no conteúdo: #run-btn, #show-code-btn, #plot-output.
 *
 * Contrato com o Python (simulation.py):
 *   document.getElementById("sim-loading").showModal() / .close()
 *   js.i18n.t("py.…") para rótulos de gráfico e mensagens (etapa 4 da v1.1.0)
 */

const ROOT = new URL("../", import.meta.url); // app.js vive em assets/
const DEFAULT_LANG = "pt"; // etapa 3 da v1.1.0 troca para "en"
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

function injectDialogs() {
  if (!document.getElementById("loading")) {
    document.body.prepend(
      el(`<dialog id="loading" class="mm-dialog">
            <div class="mm-box">
              <div class="mm-spinner"></div>
              <p class="text-gray-700 font-medium text-center" data-i18n="loading.pyscript"></p>
            </div>
          </dialog>`)
    );
  }
  if (!document.getElementById("sim-loading")) {
    document.body.prepend(
      el(`<dialog id="sim-loading" class="mm-dialog">
            <div class="mm-box">
              <div class="mm-spinner"></div>
              <p class="text-gray-700 font-medium text-center" data-i18n="sim.running"></p>
              <p id="mm-progress" class="text-sm text-gray-500 text-center hidden"></p>
            </div>
          </dialog>`)
    );
  }
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
            <div class="mm-lang flex items-center gap-1" role="group" data-i18n-attr="aria-label:lang.label">
              <button type="button" data-lang="pt">PT</button><span class="text-gray-300">|</span><button type="button" data-lang="en">EN</button>
            </div>
            <img src="${logo}" class="h-10 w-auto opacity-80" data-i18n-attr="alt:logo.alt" />
          </div>
        </header>`)
  );
  app.querySelectorAll(".mm-lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
}

function injectFooter(app) {
  app.after(
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
    const r = await fetch(new URL("simulation.py", location.href), { cache: "no-store" });
    code.textContent = await r.text();
    if (window.Prism) Prism.highlightElement(code);
  } catch (e) {
    code.textContent = String(e);
  }
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
// Ciclo de vida
// ---------------------------------------------------------------------------
async function boot() {
  const app = document.getElementById("mm-app");
  injectDialogs();
  // Backend wasm do Matplotlib (plt.show()) desenha dentro deste elemento em vez do body
  const out = document.getElementById("plot-output");
  if (out) document.pyodideMplTarget = out;
  if (app) {
    injectHeader(app);
    injectFooter(app);
    injectCodeModal();
    decorateButtons();
  }
  await loadStrings(lang);
  applyStrings();

  const loading = document.getElementById("loading");
  const onReady = () => {
    if (loading.open) loading.close();
    const out = document.getElementById("plot-output");
    if (out && out.textContent.trim() === "" && out.children.length === 0) {
      out.textContent = t("output.placeholder");
    }
  };
  if (window.__pyReadyFired) onReady();
  else addEventListener("py:ready", onReady, { once: true });
  if (!window.__pyReadyFired && !loading.open) loading.showModal();
}

// O listener é registrado no carregamento do módulo (antes do core.js) para
// não perder o py:ready caso o PyScript fique pronto antes do boot terminar.
addEventListener("py:ready", () => {
  window.__pyReadyFired = true;
}, { once: true });

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
