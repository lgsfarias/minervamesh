# Especificação da v1.2.0 — repaginação da interface

Data: 2026-10-17. Estado: decisões em aberto fechadas pelo autor em
17/10/2026 (seção 2); nada implementado.
Protótipo aprovado em sessão de 17/10/2026 no Claude Design:
<https://claude.ai/artifact/UDaG8qsi2XwV1V7GDcUYgR> (10 pranchetas: menu desktop e
celular; simulação desktop, tablet, celular, escuro, embutida no LabMFA; estados
primeira abertura, pronto e executando).

## 1. Contexto e diagnóstico

A v1.1.0 entregou bilíngue, progresso, descrições padronizadas e esquemas. O
passe de design ficou anotado para "uma v1.2" (tasks da v1.1.0, etapa 2). O
levantamento de 17/10/2026 (capturas em 1440, 820 e 390 px) mostrou:

1. **Overflow horizontal a 390 px**: título cortado, logo da Poli fora da tela,
   equações inline estourando a largura (só `$$…$$` tem rolagem própria).
   Medido por `scripts/shots.mjs` em 17/10/2026: transiente 2D (+78 px),
   potencial (+44 px) e Navier-Stokes (+43 px); os sete módulos 1D e o menu
   não rolam, mas cortam o título e o logo dentro da janela.
2. **Resultado abaixo da dobra**: no desktop a primeira tela é o bloco "Sobre o
   problema"; no celular o resultado fica cerca de duas telas abaixo.
3. **Menu** esconde as 10 simulações em dois `<details>` fechados.
4. **Figura interativa (webagg) cortada** a 1400 px nos módulos de `mmplot`
   (viga, vibração, condução MEF, convecção-difusão, potencial): o canvas é
   mais largo que a coluna 3/4 e a área corta à direita.
5. Nada é mostrado na área de resultado antes do primeiro Run além de
   "Clique em Executar".

## 2. Decisões do autor (17/10/2026)

- **Identidade**: só a marca MinervaMesh (wordmark + ícone de malha). O logo da
  Poli sai do topo; a afiliação vai em texto no rodapé.
- **"Sobre o problema"**: faixa compacta (equação governante + esquema do
  domínio) sempre visível acima do resultado; texto completo numa aba
  "Problema" ao lado de "Parâmetros" e "Código" no painel lateral.
- **Menu**: galeria de cards com miniatura real do resultado, badges de método
  (FEM/FDM), dimensão (1D/2D) e regime; chips de filtro por método.
- **Alvo**: standalone (GitHub Pages) **e** embutida no site do LabMFA; **com
  modo escuro** (figuras do Matplotlib continuam claras sobre placa branca).
- **Resultado**: a figura ocupa toda a largura da área; validação embaixo.
- **Aba Código**: só o código e o aviso de comentários em PT; sem link para o
  GitHub dentro da simulação.
- Plotagens (ex.: subplots do Navier-Stokes lado a lado) ficam para depois,
  caso a caso.
- **Tema**: botão claro/escuro no cabeçalho, ao lado de PT | EN; o sistema
  dá só o valor inicial.
- **Cards do menu**: montados por dados pelo `app.js` a partir de uma lista
  única (critério do autor: o que for mais fácil de manter).
- **Miniaturas**: versionadas em `assets/thumbs/`.

## 3. Regra de ouro e critério de aceite

Continua valendo: **nenhuma alteração numérica** em `simulation.py`,
`common.py` e `geometry.py`. Esta versão, em princípio, **não toca em nenhum
`.py`**; se uma etapa precisar (ex.: classe Tailwind dentro de uma string
HTML montada no Python), é edição de apresentação, em commit separado.

Aceite, antes da tag:

1. Referência numérica: `docs/v1.1.0/baseline/final-sem-gancho.partial.json`
   (texto exibido e hash das imagens, PT, defaults da v1.1.0) e o do canal
   (`final-sem-gancho-ns-canal.partial.json`). Como o gancho
   `__minervamesh_result` foi removido no fechamento da v1.1.0, o critério é
   **texto exibido idêntico** e **hash das imagens idêntico**; esta versão
   não muda figuras, então o hash tem de bater.
2. `node scripts/baseline.mjs --label v1.2.0-final` (e `--only navier-stokes
   --set` para o canal, como na v1.1.0) ao final; comparar com
   `scripts/compare-baseline.mjs`.
3. Captura das 10 páginas e do menu em 1440, 820 e 390 px, PT e EN, claro e
   escuro, por script (`scripts/shots.mjs`, novo), e conferência visual pelo
   autor: sem rolagem horizontal, resultado visível na primeira tela do
   desktop, Run acessível sem rolar no celular.
4. Strings novas em EN revisadas por agente revisor independente, como na
   v1.1.0 (o autor não lê inglês).
5. `scripts/i18n-check.mjs` sem pendências.

## 4. Sistema visual

Arquivo: `assets/tailwind.src.css` (tokens) e `tailwind.config.js` (cores
mapeadas para as variáveis).

- **Tipografia**: IBM Plex Sans (400, 500, 600) e IBM Plex Mono (400, 500)
  via Google Fonts, um `<link>` injetado pelo `app.js`; fallback de sistema.
  Equações continuam no MathJax.
- **Tokens** como variáveis CSS em `:root` (claro) e `[data-theme="dark"]`
  (escuro), com os valores do protótipo:

  | token | claro | escuro |
  |---|---|---|
  | `--mm-bg` | `#f3f5f8` | `#0f1216` |
  | `--mm-surface` | `#ffffff` | `#171b21` |
  | `--mm-surface2` | `#f8f9fb` | `#1d222a` |
  | `--mm-text` | `#15181d` | `#e7eaee` |
  | `--mm-muted` | `#5b6470` | `#a2a9b4` |
  | `--mm-border` | `#dde1e6` | `#2a3038` |
  | `--mm-accent` | `#2a5ad8` | `#8fabf7` |
  | `--mm-accent-strong` | `#1e47b5` | `#b3c6fa` |
  | `--mm-accent-soft` | `#e6ecfb` | `#1d2a4a` |
  | `--mm-ok` / `--mm-ok-soft` | `#1f7a3d` / `#e3f3e8` | `#6fd392` / `#163421` |
  | `--mm-fem-bg` / `--mm-fem-fg` | `#e3eafb` / `#1e47b5` | `#1d2a4a` / `#b3c6fa` |
  | `--mm-fdm-bg` / `--mm-fdm-fg` | `#fbeedc` / `#8a4b0a` | `#3a2a12` / `#f3c98a` |
  | `--mm-plate` | `#ffffff` | `#ffffff` (figuras sempre claras) |

  No Tailwind: `theme.extend.colors.mm = { bg: "var(--mm-bg)", … }` e as
  páginas usam `bg-mm-surface`, `text-mm-muted` etc. Classes `gray-*`,
  `blue-*`, `indigo-*` somem das páginas.
- **Tema**: botão claro/escuro no cabeçalho ao lado de PT | EN (ícone sol/lua,
  `aria-label`, `aria-pressed`); valor inicial de `prefers-color-scheme`,
  escolha em `localStorage` (`minervamesh.theme`), aplicada em
  `<html data-theme>` antes da primeira pintura (script inline curto no
  `<head>` de cada página, para não piscar; o `app.js` só liga o botão).
- **Esquemas SVG** (`.mm-scheme`): cores fixas de hoje passam a usar os tokens
  (traço `--mm-text`, preenchimento `--mm-accent-soft`, rótulos `--mm-muted`),
  para funcionarem no escuro.
- **Ícone de malha** (wordmark): SVG inline no `app.js`, mesmo desenho do
  protótipo; `favicon.ico` regenerado a partir dele.

## 5. Página de simulação

### 5.1 Estrutura (layout do protótipo)

```
header   wordmark · "All simulations" · (irmão: Potential flow →) · PT|EN · tema · GitHub
title    h1 + badges (método, dimensão, regime)
summary  faixa: equação governante + esquema + "Full problem statement →"
main     aside (abas Parameters | Problem | Code)  |  result (figura larga + validação)
footer   crédito · afiliação · GitHub · MIT
```

- `aside`: `flex: 1 1 280px; max-width: 320px`. `result`: `flex: 999 1 440px;
  min-width: 0`. Com `gap` 24 e gutter `clamp(16px, 2vw, 24px)`: lado a lado
  a partir de ~790 px (tablet retrato), empilhado abaixo.
- No empilhado (celular) a ordem visual é **resultado → abas → inputs**, e o
  grupo Run/Stop + `#mm-status` vira barra fixa no rodapé da tela
  (`position: sticky; bottom: 0`). Implementação: `order` no flex para
  compact (`@media (max-width: 767px)`) e o grupo de botões movido pelo
  `app.js` para um contêiner `#mm-runbar` fora do formulário nesse
  breakpoint (ids preservados). Ao terminar a simulação no celular, rolar
  até `#plot-output` (`scrollIntoView`, só se o resultado estiver fora da
  tela).

### 5.2 O que o `app.js` passa a montar

Hoje cada `index.html` tem o `<details data-mm-desc>` com as seções, o painel
de parâmetros e a área de resultado escritos à mão. Para não reescrever as 10
páginas, o `app.js` **reorganiza o DOM** a partir de marcadores:

| marcador na página | destino montado pelo `app.js` |
|---|---|
| `<details data-mm-desc>` (como hoje) | conteúdo movido para o painel da aba **Problem**; o `<details>` deixa de existir e o `localStorage` `minervamesh.desc.*` deixa de ser usado |
| primeiro `$$…$$` da seção "Equação governante" + `figure.mm-scheme` | **clonados** para a faixa `summary` (o esquema clonado respeita a troca por cenário dos módulos de escoamento: o `app.js` clona o esquema ativo e reage ao `change` do select) |
| `[data-mm-params]` (novo atributo no card de parâmetros) | painel da aba **Parameters** |
| `#show-code-btn` + modal de código | aba **Code**: o `<pre>` do modal passa a viver no painel; o modal é removido. `#show-code-btn` continua existindo (o Python não o usa, mas o id está no contrato) como botão "View the whole file" dentro da aba, que expande o `<pre>` à altura total |
| `#plot-output` (como hoje) | card de resultado, com cabeçalho "Result" + linha de metadados |

Páginas que precisam de um bloco de resumo diferente do automático
(Navier-Stokes mostra as duas equações) podem declarar `<div data-mm-summary>`
explícito; o `app.js` usa esse em vez de clonar.

Os ids do contrato com o Python **não mudam**: `run-btn`, `stop-btn`,
`show-code-btn`, `plot-output`, `loadCase`, `scenario`, `geo_type` e todos os
ids de input. `#mm-status` fica sob o Run; `#mm-run` acima de `#plot-output`.
Os módulos que fazem `innerHTML` em `#plot-output` continuam funcionando porque
o card de resultado envolve `#plot-output`, não o substitui.

### 5.3 Abas

Três abas, rótulos de uma palavra, dois indicadores de seleção (cor + sublinhado
de 2 px), `role="tablist"`/`tab`/`tabpanel`, navegação por setas. Estado
inicial: Parameters. O link "Full problem statement" da faixa e os títulos de
seção do resumo abrem a aba Problem. A aba ativa **não** é lembrada em
`localStorage` (sempre volta a Parameters).

### 5.4 Estados da área de resultado

| estado | Run / `#mm-status` | área de resultado |
|---|---|---|
| carregando (antes de `py:ready`) | "Preparing…" com spinner; barra indeterminada + "Loading the Python environment" + cronômetro (como hoje, restilizado) | **saída de exemplo** com os defaults, esmaecida (`opacity: .28`), com card central "Example output with the default parameters / Press Run to compute the fields with your parameters." |
| pronto | "Run"; status oculto | idem |
| executando | "Running…" com spinner; Stop ativo onde existe; barra de passos | `#mm-run` acima da figura (spinner, passo, barra, tempo simulado); figura anterior esmaecida (`mm-dim`) ou quadros ao vivo (NS) |
| concluído | barra verde "Simulation finished" + tempo | figura em placa branca, largura total (`max-width: 100%`, `max-height: 860px` no desktop); validação abaixo, numa faixa `--mm-ok-soft` |

A saída de exemplo é a mesma imagem da miniatura do menu, em resolução maior
(`assets/thumbs/<módulo>.png`, seção 6.2), inserida pelo `app.js` em
`#plot-output` como `<figure id="mm-example">` no `boot()`; `mmProgress.start()`
a remove. Substitui o `output.placeholder` de hoje (chave fica para o caso sem
imagem).

### 5.5 Figura interativa (webagg)

- `#plot-output canvas, #plot-output img { max-width: 100% }` e o contêiner da
  figura com `overflow: hidden` horizontal: corrige o corte da seção 1.4 sem
  mexer no `mmplot.plot_fig` (que já reduz o dpi ao container; o corte
  acontece porque o canvas nasce antes do layout final).
- Toolbar do webagg com os tokens (já existe CSS; só troca de cores).
- Métricas de validação que os `.py` montam em HTML herdam o estilo por
  seletores `#plot-output …` no CSS (sem mexer nos `.py`). Exceções a
  verificar na etapa 5: classes Tailwind literais dentro das strings Python
  (`text-green-600`, `rounded shadow w-full`) continuam compilando porque o
  `content` do Tailwind inclui os `.py`; a cor fica fora dos tokens e vira
  item da lista "plotagens caso a caso".

### 5.6 MathJax em telas estreitas

```css
mjx-container[display="true"] { display: block; overflow-x: auto; overflow-y: hidden; max-width: 100%; padding: 2px 0; }
```

Inline math longa (NS, definição de u, v, ω) vai para `$$…$$` ou para a faixa
de resumo; sem outra mudança de conteúdo.

### 5.7 Modo embutido (LabMFA)

`<body data-mm-embedded>`: o `app.js` não injeta o cabeçalho da plataforma
(o site hospedeiro tem o seu) e mantém só uma linha com "All simulations",
PT | EN e tema acima do título. O `scripts/sync_minervamesh.sh` do fork passa
a inserir o atributo nas cópias (ou o `pelicanconf`/template do Pelican; decidir
na etapa 9 lendo a doc do Pelican antes, regra de 06/10/2026).

## 6. Menu

### 6.1 Página

`index.html` da raiz reescrita:

- cabeçalho (wordmark, PT | EN, tema, GitHub);
- hero: lockup grande (ícone + "MinervaMesh"), subtítulo "Numerical
  simulations that run entirely in your browser.", parágrafo de uma frase e
  a linha mono "Python · NumPy · SciPy · Matplotlib → Pyodide (WebAssembly) ·
  nothing to install, no server";
- chips All (10) / Finite differences (2) / Finite elements (8), `aria-pressed`;
- grade `repeat(auto-fit, minmax(min(240px, 100%), 1fr))`, gap 20: 5 colunas a
  1440, 3 a 820, 1 a 390;
- card = `<a>` com miniatura (`aspect-ratio: 4/3`, `object-fit: cover`,
  `object-position: top`), badges, título, descrição (3 linhas, `line-clamp`).
  Miniatura `flex: 1 1 140px`, texto `flex: 999 1 160px`: vertical no
  desktop, horizontal no celular, sem media query;
- rodapé igual ao das simulações.

Dados dos cards: as chaves `menu.*.title` e `menu.*.desc` existentes; novas
chaves `menu.tag.*` (fem, fdm, 1d, 2d, steady, transient, static, eigenvalue)
e `menu.filter.*`. A lista dos 10 módulos fica em `assets/menu.json` (um
objeto por módulo: `id`, `path`, `method`, `dim`, `regime`; título e
descrição vêm das chaves `menu.<id>.*`), e o `app.js` monta os cards no
`bootMenu()`. Decisão do autor em 17/10/2026, por manutenção: acrescentar ou
reordenar um módulo é uma linha no JSON, não um bloco HTML; o mesmo JSON serve
ao `scripts/thumbs.mjs` e pode substituir a lista `MODULES` do
`scripts/baseline.mjs`. O `index.html` do menu fica com o hero, os chips, o
contêiner da grade vazio e um `<noscript>` com os 10 links em texto.

### 6.2 Miniaturas

`scripts/thumbs.mjs` (novo, derivado de `scripts/baseline.mjs`): abre cada
módulo em Chrome headless, roda com os defaults em EN e lê **o bitmap da
figura** (`canvas.toDataURL()` no webagg, ou o `src` base64 do `<img>`), sem
toolbar e sem título da janela e independente do tamanho da janela. O
pós-processamento é de `scripts/thumbs_crop.py` (Pillow, já presente no
python3 do sistema): `assets/thumbs/<módulo>.png` com a figura inteira a
1000 px de largura (saída de exemplo antes do Run) e `<módulo>-sm.png`, um
recorte 4:3 de 480 × 360 px para o card. Para os módulos de GIF, o último
quadro. Quando a figura tem vários gráficos, o card mostra **o mais
interessante** (decisão do autor em 17/10/2026), definido pela região
`thumb` (frações x, y, w, h) de cada módulo em `assets/menu.json`: viga →
deflexão (topo), vibração → modo 3 (base), potencial → função corrente com a
malha (topo), Navier-Stokes → linhas de corrente (base). As imagens são
versionadas no repositório (≈1–2 MB no total) e regeneradas sempre que uma
plotagem mudar.

## 7. Internacionalização

Chaves novas em `assets/i18n/common.*.json` (PT e EN, revisão EN obrigatória):

- `nav.all`, `nav.theme.light`, `nav.theme.dark`;
- `tab.params`, `tab.problem`, `tab.code`, `summary.full_problem`,
  `code.view_whole`;
- `result.title`, `result.no_run`, `result.live`, `result.example.title`,
  `result.example.body`, `result.meta.step` (`step {i} / {n}`);
- `menu.hero.*`, `menu.filter.*`, `menu.tag.*`, `footer.affiliation`;
- `badge.*` por módulo (método, dimensão, regime) ou derivadas de
  `data-mm-tags` no `<body>` de cada página.

`logo.alt` e o arquivo `assets/logo_poli_ufrj.png` saem.

## 8. Ordem de implementação

Cada etapa termina com `npm run build:css`, abertura das 10 páginas e
conferência manual. Commits só com autorização; um commit por etapa, `.py`
(se houver) em commit próprio.

1. **Preparação**: branch `feat/v1.2.0`; `scripts/shots.mjs` (capturas das 11
   páginas × 3 larguras × PT/EN × claro/escuro) e `scripts/thumbs.mjs`;
   rodar os dois na `main` atual para ter o "antes" (capturas em
   `docs/v1.2.0/antes/`, fora do git, ou só no scratchpad).
2. **Tokens e tema**: variáveis CSS, `tailwind.config.js`, fontes,
   `data-theme`, botão de tema, esquemas SVG com tokens. Páginas ainda com
   o layout antigo, mas já nas cores novas (claro e escuro).
3. **Cabeçalho, rodapé e marca**: wordmark + ícone no `app.js`; logo da Poli
   e `logo.alt` removidos; favicon novo; afiliação no rodapé.
4. **Layout da simulação**: faixa de resumo, abas, reorganização do DOM
   (seção 5.2), card de resultado largo com validação embaixo, CSS do
   MathJax e do webagg. Atributo `data-mm-params` nas 10 páginas e
   `data-mm-summary` onde o automático não servir (NS). Conferir os dois
   módulos de escoamento (esquema por cenário) e o NS (Stop, presets).
5. **Estados**: saída de exemplo antes do Run, estilos de carregando /
   executando / concluído, `scrollIntoView` no celular.
6. **Celular**: ordem resultado → abas → inputs, barra Run fixa, alvos de
   44 px, revisão dos inputs em duas colunas.
7. **Menu**: `index.html` novo, miniaturas, chips, cards por dados.
8. **i18n**: chaves novas PT/EN, `scripts/i18n-check.mjs`, revisão EN por
   agente.
9. **Modo embutido**: `data-mm-embedded`, mecanismo de inserção no fork do
   labmfa (ler a doc do Pelican antes), teste do build do Pelican.
10. **Fechamento**: limpeza dos `simulation.py` exibidos ao aluno (comentários
    que só repetem o código saem; blocos repetidos de i18n e progresso passam
    a um `assets/mmcommon.py` compartilhado; comentários sobre o método ficam;
    commits separados, baseline depois), baseline final e comparação
    (seção 3), capturas "depois", `CHANGELOG.md`, tag `v1.2.0`, espelho
    `gh-pages`, sync no fork e atualização do PR #1.

## 9. Fora de escopo

- Qualquer mudança de plotagem (subplots lado a lado, cores das figuras,
  classes Tailwind dentro de strings Python): lista "caso a caso" para depois.
- Atualização do PyScript e do Prism (já anotadas no `CLAUDE.md`).
- Botão Parar fora do Navier-Stokes.
- Tradução dos comentários dos `.py`.
- Ilustração decorativa; as miniaturas são saídas reais.

## 10. Riscos e cuidados

- **Reorganizar o DOM antes do PyScript**: o `app.js` já roda antes do
  `core.js`, e os `.py` só buscam ids no clique; mover nós preserva ids e
  listeners. Mesmo assim cada página é aberta e executada na etapa 4, e a
  baseline roda no fechamento.
- **MathJax e nós clonados**: clonar a equação para a faixa antes do typeset
  gera dois `$$…$$` tipografados independentes; o `app.js` injeta o MathJax
  depois da reorganização (hoje injeta antes).
- **`mmplot` mede `#plot-output` no clique** (`clientWidth`): o card de
  resultado precisa ter largura final antes do Run, o que vale para flex sem
  animação. Em celular, se a área estiver oculta por `display:none` em alguma
  aba, a medida dá 0 e o dpi cai para 100; a área de resultado nunca fica em
  aba, só o painel lateral.
- **Hash de imagem** na baseline: esta versão não altera figuras, mas uma
  mudança involuntária de largura no `clientWidth` muda o dpi e o hash dos
  módulos de `mmplot`. Se o hash divergir e o texto não, comparar a imagem a
  olho e registrar a causa em `docs/v1.2.0/aceite.md`.
- **Tema escuro e imagens dos `.py`**: GIFs e PNGs com fundo branco ficam na
  placa branca por design; nada a fazer nas figuras.
- **Offline** (cópia da defesa): as fontes do Google não carregam sem
  internet; o fallback de sistema cobre. Não há dependência nova de runtime.
