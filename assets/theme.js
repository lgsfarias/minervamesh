/**
 * MinervaMesh — tema claro/escuro (v1.2.0).
 *
 * Script clássico, síncrono, incluído no <head> de cada página antes da folha
 * de estilo: escreve <html data-theme="light|dark"> antes da primeira pintura,
 * para a página não piscar no tema errado. Valor inicial: escolha salva em
 * localStorage (minervamesh.theme) ou, na falta dela, prefers-color-scheme.
 *
 * window.mmTheme.get() / .set("light"|"dark") / .toggle(): usados pelo botão
 * do cabeçalho (assets/app.js). Enquanto o usuário não escolher, a página
 * acompanha mudanças do sistema.
 */
(function () {
  var KEY = "minervamesh.theme";
  var root = document.documentElement;
  var mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function saved() {
    try {
      var t = localStorage.getItem(KEY);
      return t === "light" || t === "dark" ? t : null;
    } catch (e) {
      return null;
    }
  }
  function system() {
    return mq && mq.matches ? "dark" : "light";
  }
  function apply(t) {
    root.dataset.theme = t;
    root.style.colorScheme = t; // barras de rolagem e controles nativos
  }

  apply(saved() || system());
  if (mq && mq.addEventListener) {
    mq.addEventListener("change", function () {
      if (!saved()) apply(system());
    });
  }

  window.mmTheme = {
    get: function () {
      return root.dataset.theme === "dark" ? "dark" : "light";
    },
    set: function (t) {
      apply(t);
      try {
        localStorage.setItem(KEY, t);
      } catch (e) {}
      window.dispatchEvent(new CustomEvent("mm:theme", { detail: t }));
    },
    toggle: function () {
      this.set(this.get() === "dark" ? "light" : "dark");
    },
  };
})();
