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

## Etapa 2 — Tokens e tema (spec 4)

- [ ] Variáveis CSS claro/escuro em `assets/tailwind.src.css`; `theme.extend.colors.mm` no `tailwind.config.js`
- [ ] Fontes IBM Plex via `<link>` injetado pelo `app.js`
- [ ] `data-theme` em `<html>` antes da primeira pintura; `prefers-color-scheme` + `localStorage` `minervamesh.theme`
- [ ] Esquemas SVG (`.mm-scheme`) com tokens
- [ ] Trocar `gray-*`/`blue-*`/`indigo-*` das 11 páginas e do `app.js` pelos tokens; build e conferência nos dois temas

## Etapa 3 — Cabeçalho, rodapé e marca (spec 4, 5.1)

- [ ] Wordmark + ícone de malha no `app.js` (simulações e menu)
- [ ] Remover `assets/logo_poli_ufrj.png`, `logo.alt` e o `<img>` do menu
- [ ] Afiliação em texto no rodapé (`footer.affiliation`)
- [ ] Favicon novo a partir do ícone
- [ ] Botão de tema ao lado de PT | EN (sol/lua, `aria-pressed`), ligado pelo `app.js`; script inline no `<head>` aplica `data-theme` antes da primeira pintura

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

## Etapa 7 — Menu (spec 6)

- [ ] `index.html` novo: hero com lockup, chips, contêiner da grade, `<noscript>` com os links, rodapé
- [ ] `assets/menu.json` (id, path, method, dim, regime) e montagem dos cards no `bootMenu()`; `scripts/thumbs.mjs` e `scripts/baseline.mjs` lendo a mesma lista
- [ ] Miniaturas em `assets/thumbs/`
- [ ] Filtro por método com `aria-pressed`

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
