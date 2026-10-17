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

## Etapa 4 — Layout da página de simulação (spec 5.1 a 5.3, 5.5, 5.6) (17/10/2026)

- [x] `layoutModule()` no `app.js`: linha de título + badges (de `assets/menu.json`, campo `module`), faixa de resumo (clone do primeiro `$$…$$` da seção da equação + clones dos esquemas; `[data-mm-summary]` explícito tem precedência), painel lateral com abas, card de resultado; contêineres antigos vazios removidos; scripts inline do módulo movidos para o fim do `body`
- [x] ~~Abas Parameters | Problem | Code no painel lateral~~ **Revisão 2 (autor, 17/10/2026, após ver a aba Code "tripinha")**: o painel lateral fica só com Parâmetros (o h3 da página vira o cabeçalho); o problema vai para a **coluna larga**, num seletor Resultado | Problema no cabeçalho do card (`role=tab`, setas; `mmProgress.start()` volta ao resultado; o link da faixa abre o problema); o código volta a ser **popup** (`injectCodeModal`: até 1100 px, 88% da altura, X/Esc/clique fora, foco no X e de volta ao botão, Prism com plugin de números de linha de cdnjs com SRI). `#show-code-btn` fica abaixo do Executar como botão secundário. O link "Enunciado completo" da faixa seleciona Problema, rola o card até o topo e leva o foco ao seletor (pedido do autor). Subtítulo do card (h3 da página) só aparece quando difere de "Resultado". `[hidden] { display: none !important }` no CSS base, porque o `flex` utilitário vencia o atributo. Protótipo atualizado (pranchetas "Revision 2"). `initDescMemory` e a chave `minervamesh.desc.*` saem; chaves `tab.*`, `code.view_whole`, `code.collapse` trocadas por `view.result`, `view.problem`
- [x] `data-mm-params` e `data-mm-result` nas 10 páginas (script Python, cards localizados por `#run-btn` e `#plot-output`)
- [-] `data-mm-summary` no Navier-Stokes: o clone automático já mostra as duas equações principais; as definições de u, v, ω ficam na aba Problem
- [x] Card de resultado: cabeçalho fino com o h3 da página; `#plot-output` vira `.mm-plot` (figura em placa clara, largura total, validação abaixo). Blocos de validação montados pelos `.py` (`bg-blue-50` etc.) recebem os tokens por CSS em `.mm-result-body`, sem tocar nos `.py`
- [x] CSS: `mjx-container[display="true"]` com rolagem própria; `.mm-plot canvas/img` com `max-width: 100%`; inputs uniformes em `.mm-params` (as páginas tinham classes variadas, viga sem borda)
- [x] Botões uniformes (achado do autor na condução MDF, 17/10/2026): `layoutModule` troca as classes de `#run-btn` e `#stop-btn` por `.mm-btn-primary` / `.mm-btn-danger` (largura total, 44 px) e o grupo por `.mm-actions`; as 10 páginas traziam quatro variantes (`flex-1` em linha, `w-full`, `justify-center` sem largura, botão solto). Conferido por folha de contato dos 10 painéis. A condução MDF ainda tinha os inputs em três colunas (resto do layout antigo de largura total): agora duas, como as demais
- [-] Inline math longa do NS movida para `$$…$$`: a 320 px o MathJax já quebra a linha nas vírgulas; fica como está
- [x] Esquema por cenário (potencial e NS) continua trocando na faixa de resumo: os clones mantêm `data-mm-scheme` e o `updateUI` da página alcança todos
- [x] MathJax injetado depois da reorganização do DOM
- [x] `scripts/probe.mjs` (novo): abre um módulo, executa ações (`run`, `tab:*`, `click:*`) e captura; usado para conferir viga (run, aba Code), NS (aba Problem a 820) e condução MEF (run no escuro)
- [x] Baseline completa (`docs/v1.2.0/baseline/etapa4.json`, `scripts/baseline.mjs` agora grava em `docs/v1.2.0/baseline/`) comparada com `final-sem-gancho`: **texto idêntico nos 10 módulos**; hash de imagem diferente em 5 (condução MDF, convecção-difusão, viga, vibração, potencial), exatamente os módulos de `mmplot` cuja figura é mais larga que a coluna antiga: `plot_fig` calcula o dpi pelo `clientWidth` de `#plot-output`, que cresceu com o card largo (risco previsto na spec, seção 10). Condução MEF (figura de 800 px, dpi já no teto) e os cinco de GIF/PNG batem. Registrar em `aceite.md` no fechamento
- [x] Overflow horizontal a 390 px: zerado no transiente 2D, no potencial e no NS (`shots --label etapa4`)

## Etapa 5 — Estados da área de resultado (spec 5.4) (concluída em 17/10/2026)

- [x] `insertExample()` no `app.js`: `<figure id="mm-example">` com `assets/thumbs/<id>.png` (figura inteira, opacidade 0,3, placa clara) e card central "Saída de exemplo com os parâmetros padrão / Clique em Executar…" (`result.example.*` PT/EN); entra no `boot()` depois do `menu.json`, antes do `applyStrings`; `mmProgress.start()` a remove. O `output.placeholder` continua como fallback para página sem miniatura
- [x] Estilos de carregando (Run "Preparando…" + barra indeterminada), executando (`#mm-run` com spinner, passo, barra e tempo acima da figura; Stop ativo no NS) e concluído (`#mm-status` verde) já vinham da v1.1.0 e passaram pelos tokens na etapa 2; conferidos com `probe.mjs` (NS pronto, NS aos 6 s de execução, viga a 390 px no escuro)
- [x] `mmProgress.done()`: `scrollIntoView` em `#plot-output` só quando o card está fora da tela (celular, parâmetros acima)

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
- [x] Ordem dos cards invertida (autor, 17/10/2026): do caso mais rico ao mais simples (Navier-Stokes, potencial, transiente 2D, vibração, viga, convecção-difusão, transiente 1D, condução MEF, onda, condução MDF); a ordem didática continua pelos chips e badges. É só a ordem do array em `assets/menu.json`

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
