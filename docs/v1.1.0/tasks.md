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

## Etapa 4 — i18n dos `.py` (spec 5.2; commits separados)

- [ ] Guarda `t(key)` no topo de cada `simulation.py`
- [ ] Trocar literais de rótulos de gráfico e mensagens por `t("py.…")`
- [ ] Conferir métricas contra a baseline

## Etapa 5 — Progresso (spec 6)

- [ ] Carga inicial: cronômetro e texto sobre download e cache (medir tamanho dos pacotes)
- [ ] Componente de progresso no `app.js` com modos `spinner`, `counter`, `inline`
- [ ] Contador e yield em `calor-transiente-1d`, `transcalor`, `equacao-onda` (commits separados)
- [ ] Verificar se `asyncio.sleep(0)` repinta; senão usar `0.01`
- [ ] Navier-Stokes: contador e cronômetro inline
- [ ] Escoamento potencial: passa a abrir o `sim-loading`
- [ ] Conferir métricas e tempo do transiente 2D contra a baseline

## Etapa 6 — Descrições padronizadas (spec 7)

- [ ] MathJax 3.2.2 com SRI em todas as páginas; regra de separação texto × math
- [ ] Template `<details open>` com memória nas 10 páginas
- [ ] Textos PT reescritos a partir do `resultados.tex` (8 MEF) e do código (2 MDF)
- [ ] Verificação em fonte confiável do que não está no TCC, fontes registradas em `docs/v1.1.0/aceite.md`
- [ ] Conferência de cada condição de contorno e inicial contra o solver
- [ ] Tradução EN; revisão do EN pelo autor
- [ ] Navier-Stokes: default cavidade leve; preset "Caso de validação (Ghia et al., 1982)"

## Etapa 7 — Esquemas SVG (spec 8)

- [ ] 6 esquemas 1D
- [ ] Transiente 2D
- [ ] Potencial: cilindro, aerofólio
- [ ] Navier-Stokes: cavidade, canal com obstáculo, degrau; troca pelo seletor de cenário
- [ ] Remover `assets/conducao_esquema.png` e `assets/placeholder.png`

## Etapa 8 — Rótulos dos inputs (spec 9)

- [ ] Símbolo e unidade em todos os rótulos, PT e EN, símbolos iguais aos das equações

## Etapa 9 — Limpezas (spec 10)

- [ ] Remover `plotly` do transiente 2D; auditar os demais `config.toml`
- [ ] Recompilar Tailwind
- [ ] README: parágrafo de resumo em inglês no topo

## Etapa 10 — Fechamento

- [ ] Métricas "depois" em `docs/v1.1.0/aceite.md`; entrada curta no `CHANGELOG.md`
- [ ] Remover o gancho `_export_result` dos 10 `simulation.py` (revert do commit da etapa 1b), depois de registrada a comparação final; rodar a baseline uma última vez para confirmar texto e imagens idênticos aos da rodada "depois"
- [ ] Tag `v1.1.0`
- [ ] Espelhar `gh-pages`
- [ ] `scripts/sync_minervamesh.sh` no fork do labmfa e atualização do draft PR #1
