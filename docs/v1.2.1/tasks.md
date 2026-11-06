# Tarefas da v1.2.1

Acompanha a [especificação](spec.md). Estados: `[ ]` pendente, `[~]` em
andamento, `[x]` concluído, `[-]` descartado (com motivo).

Branch de trabalho: `feat/v1.2.1`. Nenhum commit sem autorização do autor.

## Etapa 1 — Miniaturas (06/11/2026)

- [x] `assets/menu.json`: `thumb` → `thumbs` (três regiões por módulo, `set` nas variantes de parâmetro) e `eq` nos sete módulos 1D
- [x] `scripts/thumbs.mjs`: uma execução por conjunto de parâmetros (`set` aplicado via DOM: value, checked ou select, com `input` e `change`), figura inteira só na execução padrão, `<id>-<n>.png` por região
- [x] `scripts/thumbs_crop.py`: região encaixada em 16:9 (480 × 270) com faixas brancas, `-` pula a figura inteira
- [x] Teste em `mef-conducao`: Q = 0 (reta) e Q = 12 (parábola) corretos; figura inteira idêntica à commitada
- [x] Geração completa (10 figuras inteiras + 30 miniaturas, 22 execuções, ~35 min). Figuras inteiras idênticas às da v1.2.0 em 5 módulos; nos 5 de `mmplot` diferem pelo dpi do canvas (spec 5)
- [x] Conferência a olho das 30 (folhas de contato): quatro ajustes e regeneração dos módulos afetados: transiente 2D (variantes por k em vez de dt, que oscilava com dt = 0,02; quadrado inteiro em vez de faixa), convecção-difusão (sem SUPG só oscila com Peₑ > 2: 10 elementos; Pe = 5 na segunda), potencial (Cp só o trecho central) e viga (momento sem os rótulos do eixo)
- [x] `assets/thumbs/*-sm.png` removidos (10 arquivos)

## Etapa 2 — Menu (06/11/2026)

- [x] `assets/app.js`: `fetchScheme` (primeiro `figure.mm-scheme svg` da página do módulo, `DOMParser`), card com esquema + equação + faixa de três miniaturas; `injectMathJax()` no `bootMenu` depois dos cards
- [x] `assets/tailwind.src.css`: `.mm-card` vertical, `.mm-card-scheme` (200 px), `.mm-card-eq`, `.mm-card-strip`, `line-clamp-2`; override `.mm-card .mm-scheme svg` depois do bloco `.mm-scheme`; MathJax sem margem no card
- [x] `index.html`: grade `minmax(340px, 1fr)`
- [x] Esquemas MEF (elementos + cota `h`) e MDF (nós + estêncil) nas páginas dos módulos; classes `el`, `elh`, `node`, `nodeh`
- [x] `npm run build:css`
- [x] Captura do menu a 1440 e 390 px, claro e escuro: 10 esquemas, 7 equações, overflow 0, sem erro de JS (só 404 das miniaturas ainda não geradas)

## Etapa 3 — Fechamento

- [x] README (miniaturas e `menu.json`), CHANGELOG (entrada da v1.2.1)
- [x] `node scripts/i18n-check.mjs` limpo (o aviso `result.title` órfã já existia na v1.2.0)
- [x] Captura final do menu com as 30 miniaturas, 1440 e 390 px, claro e escuro: overflow 0, sem erro de console nem de rede (06/11/2026)
- [x] Conferência visual pelo autor no `localhost:8000` (06/11/2026: "ficou ótimo")
- [x] Três commits na `feat/v1.2.1`, fast-forward na `main`, tag `v1.2.1` anotada (ba2cc6f) e `gh-pages` espelhado (06/11/2026, autorizado pelo autor)
- [x] Fork do labmfa: `sync_minervamesh.sh v1.2.1` com a linha do modo embutido, build do Pelican (59 páginas, sem aviso), menu embutido, página e viga conferidos em Chrome headless; commit 8bc6946 na `feat/minervamesh`, descrição do PR #1 atualizada e comentário ao professor (06/11/2026)
