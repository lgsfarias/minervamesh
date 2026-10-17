# Tarefas da v1.1.0

Acompanha a [especificação](spec.md). Estados: `[ ]` pendente, `[~]` em
andamento, `[x]` concluído, `[-]` descartado (com motivo). Atualizado ao final
de cada etapa; a data fica na linha da etapa.

Branch de trabalho: `feat/v1.1.0`. Nenhum commit sem autorização do autor.

## Etapa 1 — Baseline na v1.0.0 (concluída em 17/10/2026; ferramenta `scripts/baseline.mjs`)

- [x] Fixar o conjunto de parâmetros por módulo (defaults da v1.0.0) e registrar em `docs/v1.1.0/aceite.md`
- [x] Rodar as 10 simulações e anotar as métricas exibidas (ou hash da imagem final onde não há métrica)
- [x] Medir tempo do transiente 2D
- [x] Medir tempo do Navier-Stokes: canal default antigo e cavidade com valores do TCC
- [x] Testar candidatos a default da cavidade: escolhido 31×31, Δt 0,02, 300 passos, steps_per_frame 10 (~55 s). Tabela em `docs/v1.1.0/aceite.md`
- [x] Registrar a tabela "antes" em `docs/v1.1.0/aceite.md` (CHANGELOG fica curto, decisão do autor)

## Etapa 1b — Gancho de exportação de resultados (spec 2, passo 0) (concluída em 17/10/2026)

- [x] `window.__minervamesh_result` ao final do `run` em cada `simulation.py` (commit só com isso)
- [x] `scripts/baseline.mjs` lê e grava o objeto exportado
- [x] Rodar baseline: hashes idênticos aos da `v1.0.0` (valida o gancho); gravar `v1.0.0-hook` como referência numérica

## Etapa 2 — Camada compartilhada e homogeneização (spec 4, 4.1, 4.2) (concluída em 17/10/2026; menu fica para a etapa 3)

- [x] `assets/app.js`: cabeçalho, dialog de carga, dialog `sim-loading`, modal de código, rodapé, ícones SVG
- [x] `assets/i18n/common.pt.json` e `common.en.json`
- [x] Incluir `app.js` nas 10 páginas e no menu; remover markup duplicado (menu concluído na etapa 3, em 17/10/2026)
- [x] Remover Font Awesome de todas as páginas
- [x] Padronizar `<title>`, largura do container e padding do `<body>`
- [x] Layout largo: `max-w-7xl` em todas e grade parâmetros/resultado 1/4 + 3/4 a partir de `lg` (decisão de 17/10/2026; passe de design completo fica para uma v1.2)
- [x] Figuras grandes (viga, vibração): ajuste por dpi no `mmplot.plot_fig`, preservando o layout original da figura
- [x] Vibração: tabela de frequências sai da figura e vira bloco HTML de validação, como nos outros módulos (decisão de 17/10/2026). Anotado para depois: subplot de propriedades só quando forem variáveis
- [x] Padronizar ids (`run-btn`, `show-code-btn`, `plot-output`) nos HTML e nos `getElementById` dos `.py` (commit separado)
- [x] Atualizar `MODULES` em `scripts/baseline.mjs` para os ids novos
- [x] Abrir e executar as 10 páginas após a renomeação de ids (baseline completa idêntica à referência, 17/10/2026)
- [x] Gráfico interativo nos 6 módulos estáticos (spec 4.2) via `assets/mmplot.py` compartilhado; `imageio` fora do `config.toml` deles; resultado numérico idêntico nos 6 (17/10/2026)
- [x] CSS da toolbar e título da figura; `png` pré-selecionado (app.js)

## Etapa 3 — i18n dos HTML (spec 5) (concluída em 17/10/2026)

- [x] `data-i18n` nas 10 páginas e no menu (texto traduzível e math/`<sub>`/`<strong>` em elementos separados, spec 5.2)
- [x] `i18n.pt.json` e `i18n.en.json` em cada módulo (`data-mm-i18n` no body); paridade e cobertura verificadas por `scripts/i18n-check.mjs`
- [x] Seletor PT | EN, `?lang=`, `localStorage`, `<html lang>` e `<title>` dinâmicos (modo menu no `app.js`: sem dialog de carga nem cabeçalho de módulo)
- [x] Menu com linha de descrição por módulo, bilíngue (`menu.*` em `common.*.json`; cada linha confirmada contra o código do módulo)
- [x] Aviso no modal de código de que os comentários estão em português (`code.comments_pt`)
- [x] Revisão terminológica do EN por agente revisor independente (17/10/2026): 4 correções de termo (propped cantilever, fixed-pinned), SUPG com barra (Brooks & Hughes 1982), Navier-Stokes descrito como semi-implícito (confere com o código), demais ajustes de consistência. O autor não lê inglês; a revisão substitui a leitura dele
- [x] `DEFAULT_LANG = "en"` em `assets/app.js`

## Etapa 4 — i18n dos `.py` (spec 5.2; commits separados) (concluída em 17/10/2026)

- [x] Guarda `i18n_t(key)` no topo de cada `simulation.py` (nome `i18n_t` em vez de `t`: colisão com a variável de tempo; ver spec 5.2)
- [x] Trocar literais de rótulos de gráfico e mensagens por `i18n_t("py.…")`; PT dos `py.*` byte a byte igual aos literais anteriores, para a comparação de texto continuar válida
- [x] `scripts/baseline.mjs --lang` (default `pt`) e `scripts/compare-baseline.mjs`; `scripts/i18n-check.mjs` passa a varrer os `i18n_t(...)` dos `.py`
- [x] EN das chaves `py.*` revisado pelo agente revisor (1 correção de sentido, 4 de estilo)
- [x] Conferir métricas contra a baseline (17/10/2026: `__minervamesh_result` e texto idênticos nos 10 módulos)
- [x] Decisão do autor em 17/10/2026 (feito na etapa 6): `prop_type` cru em "propriedades constant" (viga, vibração) traduzido via `option.prop_type.*`, e o `geo_type.capitalize()` ("Naca4412") do título do potencial via `option.geo_type.*`; a partir daqui a divergência de texto aceita na comparação inclui "propriedades constante" na viga e na vibração

## Etapa 5 — Progresso (spec 6) (concluída em 17/10/2026)

- [x] Carga inicial: tamanho dos pacotes medido (18 a 44 MB na rede, tabela em `docs/v1.1.0/aceite.md`). Decisão do autor em 17/10/2026: **sem dialog**; botão Executar desabilitado com spinner e "Preparando...", linha discreta com barra indeterminada e cronômetro abaixo dos botões. O texto sobre download e cache foi descartado (poluía)
- [x] Componente de progresso no `app.js` (`window.mmProgress`) com modos `spinner`, `counter`, `inline`; durante a execução o painel chamativo fica acima de `#plot-output` (onde o resultado vai aparecer) e o resultado anterior esmaece; ao terminar, "Tempo total" sob o botão
- [x] Contador e yield em `calor-transiente-1d`, `transcalor`, `equacao-onda` (commits separados); transiente 2D abandona o spinner próprio
- [x] `asyncio.sleep(0)` repinta no Pyodide (teste em `docs/v1.1.0/aceite.md`); adotado
- [x] Navier-Stokes: contador `min(passo, total)` e cronômetro (modo `inline`)
- [x] Escoamento potencial e os outros 5 módulos de solução única no modo `spinner` (pedido do autor em 17/10/2026: padronizar o "carregando" de todas; a condução MDF passa de `create_proxy` síncrono a `@when` assíncrono para o estado do botão pintar). Os dialogs `#loading` e `#sim-loading` saem do `app.js`
- [x] Conferir métricas e tempo do transiente 2D contra a baseline (17/10/2026: `__minervamesh_result` e texto idênticos nos 10 módulos; transiente 2D 6,4 s → 6,7–7,0 s)
- [x] EN das 7 chaves novas (`btn.preparing`, `btn.running`, `loading.status`, `sim.running`, `sim.progress.step`, `sim.encoding`, `sim.total_time`) revisado pelo agente revisor terminológico (17/10/2026): sem correção
- [x] Invólucro `*_guarded` com `try/finally` nos 8 handlers que não tinham: o botão Executar volta ao normal se a simulação lançar exceção
- [x] Modo de falha (17/10/2026): `mmProgress.fail()` reabilita o botão e esconde painel e linha de status sem "Tempo total"; os 10 invólucros tratam a exceção e mostram `py.error` em `#plot-output` (antes, nos 8 sem `try/except` próprio, o erro só ia ao console)

## Etapa 6 — Descrições padronizadas (spec 7) (concluída em 17/10/2026)

- [x] MathJax 3.2.2 com SRI em todas as páginas (mesmo hash das páginas de escoamento; delimitadores `$…$`, `$$…$$`, `\(…\)`, `\[…\]`); regra de separação texto × math verificada em headless: número de `mjx-container` idêntico em PT e EN nas 10 páginas, nenhum `$` cru, nenhum `mjx-merror`
- [x] Template `<details open data-mm-desc>` nas 10 páginas, com memória em `localStorage` (`minervamesh.desc.<módulo>`, `initDescMemory` no `app.js`); testado fechar/recarregar/abrir/recarregar nas 10
- [x] Textos PT reescritos a partir do `resultados.tex` (8 MEF) e do código (2 MDF); textos antigos (`desc.*`) removidos dos JSONs; `section.physical_params` sai do comum; `section.method` vira "Método numérico"
- [x] Verificação em fonte confiável do que não está no TCC, fontes registradas em `docs/v1.1.0/aceite.md` ("Fontes verificadas")
- [x] Conferência de cada condição de contorno e inicial contra o solver; divergências código × TCC registradas em `docs/v1.1.0/aceite.md` (saída do potencial é natural, não Dirichlet; engastada-apoiada da vibração = engastada-engastada; nota "malha refinada" da vibração corrigida)
- [x] Tradução EN revisada pelo agente revisor terminológico (o autor não lê inglês)
- [x] Navier-Stokes: default cavidade (31×31, Δt 0,02, 300 passos, 10 passos/quadro, só no `index.html`; cavidade passa a ser a primeira opção do seletor); preset "Caso de validação (Ghia et al., 1982), ~6 min" (41×41, Δt 0,01, 2000 passos, 50 passos/quadro) via seletor + botão Carregar no `index.html`

## Etapa 7 — Esquemas SVG (spec 8) (concluída em 17/10/2026)

- [x] 7 esquemas 1D (condução MDF, onda com senoide cheia e pulso tracejado, condução MEF, transiente 1D com k₁/k₂, convecção-difusão, viga simplesmente apoiada, vibração livre-livre), cada um como `<figure class="mm-scheme" data-mm-scheme>` ao lado do parágrafo da seção "Domínio"; CSS compartilhado em `tailwind.src.css` (fora de `@layer`); rótulos só com símbolos, `<tspan>` para sub/sobrescritos; `<defs>` com ids prefixados por esquema
- [x] Transiente 2D (quadrado unitário, laterais com gradiente e a expressão linear em y)
- [x] Potencial: cilindro, aerofólio; troca em `toggleGeoParams()`
- [x] Navier-Stokes: cavidade, canal com obstáculo (cilindro + retângulo w × h tracejado), degrau; troca em `updateUI()` (cavidade / canal / degrau; o preset Ghia cai na cavidade)
- [x] Legenda "dimensões definidas pelos parâmetros" (`scheme.parametric`) nas páginas de escoamento e `aria-label` traduzível (`scheme.alt`) em todos os SVGs; EN revisado pelo agente revisor terminológico
- [x] Remover `assets/conducao_esquema.png` e `assets/placeholder.png`, a `<img>` da condução MDF e a chave `img.alt`
- [x] Verificado em headless (17/10/2026): esquema visível correto em cada estado dos seletores, sem erros de console, sem chaves faltando, figura abaixo do texto em 390 px de largura

## Etapa 8 — Rótulos dos inputs (spec 9) (concluída em 17/10/2026)

- [x] Símbolo e unidade em todos os rótulos (76 chaves `input.*` nos 10 módulos), PT e EN, no padrão `Nome símbolo [unidade]` sem dois-pontos; símbolos iguais aos das descrições (nₑ, nₓ, k₁, k₂, Δt, tₘₐₓ, T₀, Tₗ, ρcₚ); adimensionais sem colchetes (Re, CFL, θ; α e Q da condução MDF, módulo sem unidades); subscritos sem glifo Unicode (c_x, c_y, n_x, n_y, T_bottom, T_top, c_v, U∞) em `<sub>` fora do `data-i18n`; texto fallback dos `<label>` nos HTML atualizado junto; EN revisado pelo agente revisor terminológico

## Etapa 9 — Limpezas (spec 10) (concluída em 17/10/2026)

- [x] CDN num lugar só (17/10/2026, pedido do autor): Prism carregado pelo `app.js` no primeiro clique em "Ver código" e MathJax injetado pelo `app.js`, ambos com SRI; os `<head>` perdem 6 tags de CDN e ficam com PyScript, Tailwind e `app.js`. Verificado em headless: mesma contagem de `mjx-container` nas 10 páginas, Prism só pedido ao abrir o modal, menu sem MathJax nem Prism
- [x] Remover `plotly` do transiente 2D (9,7 MB do PyPI na carga, medição da etapa 5); auditoria dos demais `config.toml` contra os `import` dos `.py` (incluindo `common.py`, `geometry.py` e `mmplot.py`): todos os outros pacotes são usados, nada mais a remover
- [x] Recompilar Tailwind (feito ao fim de cada etapa que tocou HTML; `assets/tailwind.css` versionado)
- [x] README: parágrafo de resumo em inglês no topo (revisado pelo agente revisor terminológico) e linhas desatualizadas pela v1.1.0 corrigidas (`assets/`, arquivos `i18n.*.json` por módulo, MathJax, passos de uso com idioma, "Executar" e "Ver código-fonte")

## Etapa 10 — Fechamento

- [ ] Métricas "depois" em `docs/v1.1.0/aceite.md`; entrada curta no `CHANGELOG.md`
- [ ] Remover o gancho `_export_result` dos 10 `simulation.py` (revert do commit da etapa 1b), depois de registrada a comparação final; rodar a baseline uma última vez para confirmar texto e imagens idênticos aos da rodada "depois"
- [ ] Tag `v1.1.0`
- [ ] Espelhar `gh-pages`
- [ ] `scripts/sync_minervamesh.sh` no fork do labmfa e atualização do draft PR #1
