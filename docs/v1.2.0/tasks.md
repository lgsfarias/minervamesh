# Tarefas da v1.2.0

Acompanha a [especificação](spec.md). Estados: `[ ]` pendente, `[~]` em
andamento, `[x]` concluído, `[-]` descartado (com motivo). Atualizado ao final
de cada etapa; a data fica na linha da etapa.

Branch de trabalho: `feat/v1.2.0`. Nenhum commit sem autorização do autor.

## Etapa 0 — Leitura da spec pelo autor

- [x] Decisões em aberto fechadas pelo autor (17/10/2026): botão de tema no cabeçalho; cards montados por dados (`assets/menu.json`); miniaturas versionadas em `assets/thumbs/`
- [ ] Criar a branch `feat/v1.2.0` a partir da `main`

## Etapa 1 — Preparação (iniciada em 17/10/2026)

- [x] Branch `feat/v1.2.0` criada a partir da `main` (17/10/2026)
- [x] `assets/menu.json`: lista única dos 10 módulos (id, path, método, dimensão, regime, tipo de saída, região `thumb`)
- [x] `scripts/shots.mjs`: capturas das 11 páginas × 1440/820/390 × PT/EN (× temas via `--themes`, quando existirem); acusa `OVERFLOW` quando a página rola na horizontal; saída em `docs/v1.2.0/shots/<label>/` (ignorada pelo git)
- [x] `scripts/thumbs.mjs` + `scripts/thumbs_crop.py`: bitmap da figura (canvas ou img), figura inteira a 1000 px e recorte 4:3 do gráfico mais interessante a 480 × 360
- [x] Capturas "antes" (66 em `docs/v1.2.0/shots/antes/` + 11 a 390 px em `antes-390/`): overflow horizontal só no transiente 2D, potencial e Navier-Stokes (17/10/2026)
- [x] Miniaturas geradas na `main` atual (`assets/thumbs/`, 20 PNGs quantizados a 256 cores, 1,3 MB); recortes ajustados após conferência a olho: convecção-difusao e potencial → gráfico de cima, viga → deflexão, vibração → modo 5 (o 3 fica para quando o default mudar), NS → linhas de corrente; região mais baixa que 4:3 ganha faixas brancas em vez de perder o eixo x

## Etapa 2 — Tokens e tema (spec 4) (concluída em 17/10/2026)

- [x] Variáveis CSS claro/escuro em `assets/tailwind.src.css` (`@layer base`, `:root` e `[data-theme="dark"]`); `theme.extend.colors.mm`, `fontFamily` e `darkMode: ["selector", '[data-theme="dark"]']` no `tailwind.config.js`
- [x] Fontes IBM Plex Sans/Mono via `<link>` (preconnect + css2) injetado no topo do `app.js`; sem SRI porque o CSS do Google Fonts varia por navegador
- [x] `assets/theme.js` (script clássico no `<head>` das 11 páginas, antes do `tailwind.css`): `data-theme` antes da primeira pintura, `prefers-color-scheme` como valor inicial, `localStorage` `minervamesh.theme`, `window.mmTheme.get/set/toggle` para o botão da etapa 3
- [x] Esquemas SVG (`.mm-scheme`) com tokens (`--mm-hot` para a borda quente)
- [x] Mapeamento mecânico das 11 páginas e do `app.js` (script Python de uma vez, 560 trocas): `gray-900/800/700 → mm-text`, `gray-600/500/400 → mm-muted`, `gray-300/200/100 (borda) → mm-border`, `blue-800/700/600 → mm-accent`, `blue-50/100 → mm-accent-soft`, `white (texto em botão) → mm-on-accent`, `green → mm-ok`, `red → mm-danger`, `indigo → mm-accent`; nenhuma classe de cor antiga restou nos HTML/JS (as das strings Python ficam para a lista caso a caso)
- [x] `scripts/baseline.mjs` e `scripts/thumbs.mjs` detectam o fim da execução por `bg-mm-ok` (antes `bg-green-600`)
- [x] Build e conferência em viga e Navier-Stokes nos dois temas (`docs/v1.2.0/shots/etapa2/`): layout inalterado, MathJax e esquemas legíveis no escuro

## Etapa 3 — Cabeçalho, rodapé e marca (spec 4, 5.1) (concluída em 17/10/2026)

- [x] Cabeçalho das simulações vira barra de largura total no `body` (marca com ícone de malha · "Todas as simulações" · irmão · PT | EN · tema · GitHub); o padding que era do `body` passou para `#mm-app` nas 10 páginas
- [x] Remover `assets/logo_poli_ufrj.png`, `logo.alt`, `nav.back` (substituída por `nav.all`) e o `<img>` do menu
- [x] Rodapé em barra, com afiliação em texto (`footer.affiliation`) e links para o código e a licença
- [x] `favicon.ico` regenerado a partir do ícone de malha (Pillow, 16/32/48/64)
- [x] Botão de tema (lua/sol, `aria-pressed`, `aria-label` traduzido) ao lado de PT | EN nas simulações e no menu, via `window.mmTheme` de `assets/theme.js`. O `i18n-check` avisa `nav.theme.*` como órfãs porque as chaves são escolhidas em tempo de execução; é aviso, não erro
- [x] Conferido em viga (claro/escuro, 1440 e 390), menu e Navier-Stokes PT (`docs/v1.2.0/shots/etapa3/`); a 390 px o cabeçalho quebra em duas linhas sem overflow. O menu em si só muda na etapa 7

## Etapa 4 — Layout da página de simulação (spec 5.1 a 5.3, 5.5, 5.6)

- [ ] `app.js`: faixa de resumo (clone da primeira `$$…$$` + esquema ativo; `data-mm-summary` explícito tem precedência)
- [ ] `app.js`: abas Parameters | Problem | Code; `<details data-mm-desc>` vira painel Problem; modal de código vira painel Code; `localStorage` `minervamesh.desc.*` desativado
- [ ] `data-mm-params` nas 10 páginas; `data-mm-summary` no Navier-Stokes
- [ ] Card de resultado: cabeçalho "Result" + metadados; figura em placa branca, largura total; validação abaixo
- [ ] CSS: `mjx-container[display="true"]` com rolagem; `#plot-output canvas/img` com `max-width: 100%`; toolbar do webagg com tokens
- [ ] Inline math longa do NS movida para `$$…$$`
- [ ] Esquema por cenário (potencial e NS) continua trocando na faixa de resumo
- [ ] MathJax injetado depois da reorganização do DOM
- [ ] Abrir e executar as 10 páginas (Stop e presets no NS)

## Etapa 5 — Estados da área de resultado (spec 5.4)

- [ ] `<figure id="mm-example">` com `assets/thumbs/<módulo>.png` inserida no `boot()`; removida em `mmProgress.start()`
- [ ] Estilos de carregando, executando (`#mm-run`, `mm-dim`) e concluído (`#mm-status` verde)
- [ ] `scrollIntoView` em `#plot-output` ao terminar, só quando fora da tela

## Etapa 6 — Celular (spec 5.1)

- [ ] Ordem resultado → abas → inputs abaixo de 768 px
- [ ] Grupo Run/Stop + `#mm-status` em `#mm-runbar` fixo no rodapé da tela (ids preservados)
- [ ] Alvos de toque ≥ 44 px; inputs em duas colunas; sem rolagem horizontal a 390 px nas 10 páginas

## Etapa 7 — Menu (spec 6) (adiantada a pedido do autor; concluída em 17/10/2026)

- [x] `index.html` novo: hero com lockup, chips, contêiner da grade, `<noscript>` com os 10 links; cabeçalho (variante menu, sem navegação) e rodapé vêm do `app.js`
- [x] `buildMenu()` no `app.js` monta os cards a partir de `assets/menu.json`; componentes `.mm-chip`, `.mm-card`, `.mm-tag` no CSS; chaves `menu.hero.*`, `menu.filter.*`, `menu.note`, `menu.tag.*` (PT/EN); `menu.subtitle`, `menu.choose`, `menu.mdf`, `menu.mef` removidas
- [-] `scripts/baseline.mjs` lendo `assets/menu.json`: fica como está (tem campos próprios, `legacy`, `extraText`); só `thumbs.mjs` e `shots.mjs` leem o JSON
- [x] Miniaturas em `assets/thumbs/` (etapa 1)
- [x] Filtro por método com `aria-pressed`
- [x] Conferido a 1440 (5 colunas), 820 (3 colunas, último card sozinho) e 390 (cards horizontais), claro e escuro (`docs/v1.2.0/shots/etapa7/`)
- [x] Ajustes do autor (17/10/2026): sem a frase "Cada caso compara…" ao lado dos chips (`menu.note` removida) e sem a marca no cabeçalho do menu, porque o hero já tem o lockup grande

## Etapa 8 — i18n (spec 7)

- [ ] Chaves novas em `common.pt.json` e `common.en.json`
- [ ] `scripts/i18n-check.mjs` sem pendências
- [ ] Revisão do EN por agente revisor independente

## Etapa 9 — Modo embutido (spec 5.7)

- [ ] `data-mm-embedded`: sem cabeçalho da plataforma; linha com "All simulations", PT | EN e tema
- [ ] Mecanismo de inserção do atributo nas cópias do fork do labmfa (ler a doc do Pelican antes)
- [ ] Build do Pelican e abertura das páginas embutidas

## Etapa 10 — Fechamento (spec 3)

- [ ] `scripts/baseline.mjs --label v1.2.0-final` (+ canal do NS) e `compare-baseline.mjs` contra `final-sem-gancho`
- [ ] Capturas "depois" com `scripts/shots.mjs`; conferência visual pelo autor
- [ ] `docs/v1.2.0/aceite.md` com a comparação e eventuais divergências de hash explicadas
- [ ] `CHANGELOG.md`, tag `v1.2.0`, espelho `gh-pages`
- [ ] `scripts/sync_minervamesh.sh` no fork do labmfa e atualização do PR #1

## Lista "caso a caso" (fora desta versão)

- [ ] Navier-Stokes: subplots ψ / ω / streamlines lado a lado
- [ ] Classes Tailwind literais nas strings Python (`text-green-600`, `rounded shadow w-full`) → tokens
- [ ] Demais plotagens a revisar com o autor, uma por uma
