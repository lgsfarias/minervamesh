# Especificação da v1.1.0 — hospedagem no site do LabMFA

Data: 2026-10-17. Estado: entendimento aprovado pelo autor em 17/10/2026; este
documento ainda pendente de leitura do autor e de implementação.

## 1. Contexto

O Prof. Gustavo Rabello dos Anjos condicionou a hospedagem da plataforma em
`labmfa.coppe.ufrj.br` a quatro pontos:

1. Simulações bilíngues, com inglês como idioma padrão (o site é em inglês).
2. Indicador de progresso: simulações demoradas ficam sem retorno e parecem
   defeito do site.
3. Cada caso deve exibir a equação resolvida, o domínio, as condições de
   contorno e a condição inicial.
4. Padronizar as descrições dos problemas, inclusive decidir se haverá imagens.

Esta especificação registra as decisões tomadas em sessão de refinamento com o
autor em 17/10/2026 e o que será entregue na `v1.1.0`. O draft PR
`gustavorabello/labmfa#1` só sai do draft depois desta versão sincronizada.

## 2. Regra de ouro e critério de aceite

A regra "nenhuma alteração numérica nos `simulation.py`, `common.py` e
`geometry.py`" continua valendo. Ficam **liberadas edições não numéricas**:

- trocar literais de texto por chamadas `js.i18n.t("...")`;
- escrever no DOM (contador de progresso) e inserir `await asyncio.sleep(0)`
  dentro de loops de tempo;
- remover pacotes não usados do `config.toml`.

Critério de aceite, obrigatório antes da tag:

1. Antes de qualquer edição, fixar um **conjunto de parâmetros por módulo**
   (os defaults da `v1.0.0`) e registrá-lo em `docs/v1.1.0/aceite.md`. Rodar as 10 simulações na `v1.0.0` com esses valores e
   anotar as métricas de erro exibidas sob o gráfico (erro máximo, erro L²,
   etc., conforme cada módulo mostra).
2. Ao final, repetir **com os mesmos valores registrados**, digitando-os se
   os defaults da página tiverem mudado (caso do Navier-Stokes, seção 7.2).
   As métricas têm de ser **idênticas** (mesmos dígitos exibidos).
3. Módulos ou cenários sem métrica exibida (ex.: Navier-Stokes em canal, que
   não tem referência analítica): comparar a imagem final renderizada (hash do
   PNG base64 do último frame, ou diff de pixels) em vez de métricas.
4. Navier-Stokes: registrar dois conjuntos, o default antigo (canal, 500
   passos) e o da cavidade com os valores do TCC, e medir o tempo de execução
   dos dois.
5. A tabela "antes × depois" vai em `docs/v1.1.0/aceite.md` e, resumida, na descrição do PR.
6. No transiente 2D, medir também o tempo total de execução antes e depois
   para confirmar que o yield ao browser não encarece a simulação de forma
   perceptível.

Edições em `.py` vão em commits separados dos commits de UI, para o diff
numérico ser revisável isoladamente.

Limite do hash de imagem: qualquer mudança não numérica que toque no gráfico
(rótulos traduzidos, GIF de 1 frame virando PNG) muda o hash sem mudar o
resultado. Por isso a comparação por hash só vale até a primeira dessas
edições. Para os módulos sem métrica em texto (equação da onda, vibração,
condução MDF) e como reforço nos demais, o protocolo tem um passo zero:

0. **Gancho de exportação** (primeiro commit em `.py`, sem nenhuma outra
   mudança): ao final de cada `run`, o Python grava em
   `js.window.__minervamesh_result` um dicionário com os arrays e escalares
   que definem o resultado (ex.: vetor `T` nodal, frequências naturais,
   deslocamento da viga, `u` final da onda, `ψ_min` e sua posição). O
   `scripts/baseline.mjs` lê e grava esse objeto. Serialização: converter com
   `pyodide.ffi.to_js(result, dict_converter=js.Object.fromEntries)`, arrays
   NumPy via `.tolist()`, escalares como `float`; um dicionário Python puro
   viraria um `PyProxy` que o JS não consegue serializar. A escrita acontece
   **depois** de o gráfico ser gerado e inserido no DOM, para não poder
   influenciar a renderização. Como o gancho não toca em
   nenhum gráfico, o hash das imagens tem de ficar idêntico ao da `v1.0.0`,
   o que valida o próprio gancho. A partir daí, "métricas idênticas" significa
   texto exibido **e** `__minervamesh_result` idênticos, e o hash deixa de ser
   critério. O gancho é **temporário**: decisão do autor em 17/10/2026 de
   removê-lo no fechamento (etapa 10), depois da comparação final, para o
   `simulation.py` exibido no modal "Ver código" voltar a conter só o
   método. O commit da etapa 1b fica como receita para reinseri-lo quando
   uma mudança futura (ex.: atualização do PyScript) precisar de verificação
   numérica.

Ferramenta: `scripts/baseline.mjs` (Node + `puppeteer-core`, devDependency)
abre cada módulo em Chrome headless, roda com os parâmetros informados e grava
em `docs/v1.1.0/baseline/<label>.json` as métricas exibidas, o hash das
imagens geradas e os tempos de carga e execução. A mesma ferramenta produz os dados das tabelas "antes" e "depois" de
`docs/v1.1.0/aceite.md`.

## 3. Inventário dos módulos

| Módulo | Método | Tipo | Saída | Pacotes hoje | Fonte da formulação |
|---|---|---|---|---|---|
| `mdf/conducao-permanente-barra1d-geracao` | MDF | solução única | PNG matplotlib | numpy, matplotlib | código (não está no TCC) |
| `mdf/equacao-onda` | MDF | marcha (~445 passos default) | GIF (~50 frames) | numpy, matplotlib, imageio | código (não está no TCC) |
| `mef/conducao-permanente-barra1d-mef` | MEF | solução única | GIF 1 frame | numpy, matplotlib, imageio | `resultados.tex` 71–78 |
| `mef/calor-transiente-1d` | MEF | marcha (200 passos default) | GIF (~50 frames) | numpy, matplotlib, imageio | `resultados.tex` 159–173 |
| `mef/conveccao-difusao-1d` | MEF | solução única | GIF 1 frame | numpy, matplotlib, imageio | `resultados.tex` 253–257 |
| `mef/viga-1d-mef` | MEF | solução única | GIF 1 frame | numpy, matplotlib, imageio | `resultados.tex` 349–375 |
| `mef/vibracao-1d-mef` | MEF | solução única (autovalor) | GIF 1 frame | numpy, matplotlib, imageio, scipy | `resultados.tex` 424–458 |
| `mef/transcalor` | MEF | marcha (50 passos default) | GIF (1 frame/passo) | numpy, matplotlib, scipy, **plotly**, imageio | `resultados.tex` 515–525 |
| `mef/escoamento-fluido-2d/escoamento-potencial` | MEF | solução única | PNG | numpy, matplotlib, scipy | `resultados.tex` 604–612 |
| `mef/escoamento-fluido-2d/navier-stokes` | MEF | marcha (500 passos default) | frames ao vivo | numpy, matplotlib, scipy | `resultados.tex` 705–714 |

Linhas de `resultados.tex` referem-se a
`../tcc-minervamesh/relatorio/Textual/resultados.tex`.

Estado atual do feedback de progresso:

- Dos quatro módulos de marcha, só o Navier-Stokes faz `await` dentro do loop
  e mostra frames ao vivo. Os outros três rodam o loop inteiro bloqueando o
  browser e ainda codificam o GIF ao final.
- O transiente 2D tem o dialog `sim-loading` no HTML mas nunca o abre; usa um
  spinner próprio dentro de `#plot-output`.
- Os dois módulos de escoamento não têm o dialog `sim-loading`.
- O dialog de carga inicial ("Carregando PyScript...") fecha no evento
  `py:ready`. O PyScript 2025.8.1 não documenta nenhum evento de progresso de
  download de pacotes, então barra de porcentagem na carga inicial **não é
  possível**. Fonte: documentação oficial,
  <https://docs.pyscript.net/2025.8.1/user-guide/plugins/> (hooks `onReady`,
  `onBeforeRun`, `onAfterRun`, nenhum de progresso) e
  <https://docs.pyscript.net/2025.8.1/user-guide/dom/>.
- Modo worker foi descartado: para acessar o DOM exige headers COOP/COEP (ou
  `mini-coi.js`) e o professor publica por SFTP num servidor que não controla.
  Fonte: <https://docs.pyscript.net/2025.8.1/user-guide/workers/>.

## 4. Camada compartilhada

Cada simulação continua auto-contida no que é específico (HTML, `simulation.py`,
`config.toml`, strings do módulo, SVG do domínio). O que hoje está copiado em
10 páginas passa para `assets/`:

```
assets/
├── app.js                 # i18n, seletor de idioma, cronômetros, modais, <details> com memória
├── i18n/
│   ├── common.pt.json     # menu, voltar, executar, ver código, carregando, progresso, rodapé
│   └── common.en.json
├── tailwind.src.css / tailwind.css (recompilar: npm run build:css)
└── logo_poli_ufrj.png
simulations/<módulo>/
├── index.html
├── simulation.py
├── config.toml
├── i18n.pt.json           # título, descrição, rótulos de inputs, rótulos de gráfico (py.*)
└── i18n.en.json
```

Cada `index.html` passa a incluir `<script type="module" src="…/assets/app.js">`
antes do PyScript. O `app.js` precisa estar aplicado antes de `py:ready` para o
primeiro "Run" já ler o idioma certo.

Remover: `assets/conducao_esquema.png` (usada só pela condução MDF, substituída
por SVG) e `assets/placeholder.png` (sem uso).

### 4.1 Homogeneização das 10 páginas

Levantamento de 17/10/2026 das divergências entre as páginas, herdadas da
correria do TCC. Todas são resolvidas nesta versão, e o padrão passa a ser o
que a camada compartilhada impõe:

| Item | Hoje | Padrão v1.1.0 |
|---|---|---|
| Dialog de carga inicial `#loading` | igual em 9 páginas, copiado 9×; a condução MDF não tem nenhum indicador de carga | markup e CSS injetados pelo `app.js` em todas (etapa 2); na etapa 5 o dialog sai e vira estado do botão Executar (seção 6.1) |
| Mecanismo de saída do gráfico | condução MDF usa `plt.show()` do backend interativo do Matplotlib (desenha em `<canvas>` anexado ao body, acumulando figuras); 5 módulos estáticos geram **GIF de 1 frame** via imageio; potencial e Navier-Stokes geram PNG base64 | **Gráfico interativo** nos 6 módulos estáticos (seção 4.2); GIF nos 3 animados; Navier-Stokes avaliado à parte. `imageio` sai do `config.toml` dos estáticos |
| Dialog `#sim-loading` | existe em 7 páginas; transiente 2D tem e não usa; potencial e Navier-Stokes não têm | controlado pelo `app.js` em todas as 10 (etapa 2); na etapa 5 o dialog sai e vira painel acima do resultado (seção 6.2) |
| Ícones Font Awesome (`<i class="fas …">`) | usados em todas, mas só 3 páginas carregam a biblioteca; nas outras 7 o ícone nunca aparece | remover Font Awesome; ícones via SVG inline (seta de voltar, código, fechar) no `app.js` |
| `<title>` | formatos variados ("Condução Permanente (Barra 1D com Geração Interna)", "Transiente 2D") | `MinervaMesh · <título do módulo>`, traduzível |
| Largura do container e grade | `max-w-4xl`, `5xl`, `6xl`, `7xl`; grade parâmetros/resultado 1/3 + 2/3 | `max-w-7xl` em todas e grade 1/4 + 3/4 a partir de `lg` (decisão do autor em 17/10/2026: o gráfico interativo precisa de largura; a condução MDF mantém parâmetros em linha e resultado em largura total) |
| Padding do `<body>` | `p-6`, `p-8`, `p-6 md:p-10` | `p-6 md:p-10` em todas |
| Cabeçalho | link "Voltar para Menu" com e sem ícone; logo da Poli em posição absoluta; link cruzado só nos de escoamento | cabeçalho único: voltar ao menu, título, seletor PT/EN, logo; link cruzado potencial ⇄ Navier-Stokes mantido como item do cabeçalho |
| Ids de botão e saída | `runBtn`, `runWaveBtn`, `runHeatBtn`, `run-btn`; `wave-output`, `heat-output`, `plot-output`, `conveccao-difusao-output` | `run-btn`, `show-code-btn`, `plot-output` em todas; os `.py` mudam só o `getElementById` correspondente (edição não numérica) |
| Modal de código-fonte | copiado 10× com pequenas variações | injetado pelo `app.js`, carrega `simulation.py` da pasta atual |
| Bloco "Sobre o problema" | prosa livre, estrutura diferente em cada página | template da seção 7 |
| MathJax | só nas 2 páginas de escoamento | em todas (seção 7.3) |
| Rodapé | copiado 10× | injetado pelo `app.js` |
| `<html lang>` | `pt-BR` fixo | definido pelo `app.js` conforme idioma |

### 4.2 Gráfico interativo (decisão do autor em 17/10/2026)

O backend do Matplotlib que o Pyodide traz (`plt.show()`) desenha a figura num
`<canvas>` com toolbar: zoom por retângulo, pan, histórico de vistas,
coordenadas sob o cursor, alça de redimensionar e exportação em png, svg, pdf,
eps, jpeg, webp, tif, ps, pgf e raw. O autor considerou isso um diferencial
grande para o aluno e decidiu adotá-lo nos **6 módulos estáticos** (condução
MDF, condução MEF, convecção-difusão, viga, vibração, potencial), no lugar da
imagem PNG/GIF. Não exige atualizar o PyScript: já funciona na 2025.8.1.

A lógica comum fica em `assets/mmplot.py` (`plot_begin`, `plot_size`, `plot_show`,
`plot_append_html`), buscado por cada módulo via `[[fetch]]` no `config.toml`,
como já acontece com `common.py` nos módulos de escoamento. Cada `simulation.py`
importa e chama; o contrato (edição não numérica):

- `plt.close("all")` e limpeza de `#plot-output` antes de criar a figura: o
  backend remove o `<div>` da figura anterior, então não acumula gráficos.
- `figsize` original preservado e **dpi** reduzido a partir de `clientWidth` da
  área de resultado (`plot_fig`), porque o backend não é responsivo. Encolher o
  figsize deixava títulos e rótulos (em pontos) grandes demais e sobrepostos nas
  figuras altas (viga, vibração); reduzir o dpi escala a figura inteira como uma
  imagem.
- título do módulo como `Figure.set_label` **e** `set_window_title` antes do
  `plt.show()`: o backend manda o rótulo da figura ao conectar o canvas e, se o
  label estiver vazio, volta para "Figure N". A figura **não** é fechada depois
  do `show`.
- Métricas em HTML são anexadas abaixo da figura (`plot_append_html`), não
  substituem o conteúdo da área; o CSS tira o `flex` de `#plot-output` quando há
  figura (`:has(.mpl-toolbar)`), para figura e métricas empilharem.
- `app.js` aponta `document.pyodideMplTarget` para `#plot-output` e pré-seleciona
  `png` no seletor de formato; o CSS da camada compartilhada estiliza título e
  toolbar.

Passe de design completo da interface (barra lateral de parâmetros, resultado em
largura total, descrição em painel) fica para uma v1.2, depois da plataforma no
ar; nesta versão o ajuste é o container largo e a grade 1/4 + 3/4 (seção 4.1).

Fora desta versão, como experimento próprio: Navier-Stokes com figura
persistente atualizada a cada quadro (`fig.canvas.draw()`), que tende a ser
mais rápido que codificar PNG por quadro; recriar a figura a cada quadro foi
testado e não serve (último quadro em branco). Os 3 módulos de GIF (onda,
transiente 1D e 2D) continuam em GIF, que o aluno baixa pronto.

O utilitário `simulations/extra/gerador-malha` está fora do menu e fora do escopo da
v1.1.0: não recebe a camada compartilhada nem os ids novos.

O `index.html` de cada módulo fica reduzido a: head mínimo (PyScript, Tailwind,
MathJax, `app.js`), bloco de descrição + SVG, painel de parâmetros, área de
resultado e o `<script type="py">`. Tudo o mais vem do `app.js`.

## 5. Internacionalização

### 5.1 Comportamento

- Idiomas: `en` (padrão) e `pt`. Padrão inglês em todo lugar, inclusive no
  espelho `gh-pages`. Sem detecção pelo idioma do browser.
- Resolução do idioma, em ordem: `?lang=pt|en` na URL → `localStorage`
  (`minervamesh.lang`) → `en`.
- Seletor "PT | EN" no cabeçalho de todas as páginas, ao lado do link de
  voltar ao menu. Texto, sem bandeiras. Troca sem recarregar: o JS reaplica
  os `data-i18n`, atualiza `<html lang>` e `<title>`, e salva no
  `localStorage`.
- O Python lê o idioma na hora do "Run", via `js.i18n.t(...)`. Resultado já
  renderizado não é traduzido retroativamente; o próximo "Run" sai no idioma
  novo.
- Comentários dentro dos `simulation.py` permanecem em PT-BR. O modal
  "Ver código" ganha uma linha (traduzível) avisando que os comentários do
  código estão em português. A tradução dos comentários fica para uma
  entrega posterior.

### 5.2 Mecanismo

- HTML: `data-i18n="chave"` para texto, `data-i18n-attr="placeholder:chave"`
  para atributos quando necessário. Chaves do módulo têm precedência sobre as
  de `common`.
- JSON plano, chaves com ponto: `"desc.equation.title"`, `"input.L"`,
  `"py.plot.title"`. Equações em MathJax ficam no HTML, fora do JSON, porque
  são neutras de idioma.
- **Texto traduzível e math nunca dividem o mesmo elemento.** Um elemento com
  `data-i18n` não pode conter `$…$`, porque o MathJax substitui o conteúdo
  por nós próprios e a próxima troca de idioma os apagaria. Padrão: `<p
  data-i18n="…">texto</p>` seguido de `<div>$$…$$</div>` separado.
- `app.js` expõe `window.i18n = { lang, t(key), setLang(lang), apply() }`.
  `t` devolve a chave entre colchetes quando não encontra, para o buraco ficar
  visível em teste.
- Python: no topo de cada `simulation.py`, um guarda de três linhas:
  `_i18n = getattr(js, "i18n", None)` e `def i18n_t(key): return
  str(_i18n.t(key)) if _i18n else key`. Os literais viram
  `i18n_t("py.plot.xlabel")`; strings com valores usam `str.format`
  (`i18n_t("py.plot.title").format(t=t)`), com os mesmos placeholders em PT e
  EN. Assim, se o `app.js` falhar, a simulação roda com as chaves no lugar dos
  textos em vez de quebrar. O nome é `i18n_t`, não `t`, porque `t` é a
  variável de tempo em `render_frame` da onda e do transiente 1D (e `tr` e `_`
  também já são usados no potencial e no Navier-Stokes). O guarda fica
  separado do gancho de exportação (etapa 1b), que é revertido no fechamento.
  Em `common.py` e `geometry.py` não há strings de UI; não mudam.

### 5.3 Conteúdo

- O PT-BR dos 10 blocos de descrição é reescrito a partir do `resultados.tex`
  (MEF) e do código (MDF), seguindo o template da seção 7. O EN é traduzido a
  partir dele. O autor revisa o EN antes do PR.
- Menu (`index.html` raiz): títulos traduzidos e uma linha de descrição por
  módulo (fenômeno ou equação, método, tipo de referência: analítica, Ghia et
  al. 1982, etc.).
- README continua em PT-BR e ganha um parágrafo de resumo em inglês no topo,
  com link para a plataforma no site do LabMFA.

## 6. Indicador de progresso

Revisado na implementação (decisão do autor em 17/10/2026): **nenhum dialog**.
Os dialogs `#loading` e `#sim-loading` saíram do `app.js`. A carga inicial é
discreta, junto ao botão; a execução é chamativa, na área do resultado, para
levar o olhar ao lugar onde a figura vai aparecer.

### 6.1 Carga inicial do PyScript

Em todas as páginas, enquanto o `py:ready` não chega:

- o botão Executar fica desabilitado, com spinner e o rótulo "Preparando...";
- logo abaixo do grupo de botões, uma linha discreta (`#mm-status`) com barra
  fina indeterminada, o texto "Preparando o ambiente de simulação..." e o
  cronômetro desde o início da navegação, atualizado a cada segundo;
- sem barra de porcentagem (não há evento para isso) e **sem texto** sobre o
  tamanho do download (descartado por poluir; a medição fica em
  `docs/v1.1.0/aceite.md`).

### 6.2 Durante a simulação

Um componente (`window.mmProgress`, chamado pelo Python via
`progress("start"|"step"|"encoding"|"done")`) com três modos. Em todos, o
botão Executar fica desabilitado com spinner e "Executando...", e um painel
chamativo (`#mm-run`: spinner, texto, número à direita e barra) é inserido
logo **acima de `#plot-output`**, fora dele porque o Python reescreve o
`innerHTML` da área. Ao terminar, o painel some e a linha discreta sob o botão
mostra "Tempo total | … s" com a barra cheia em verde.

- `counter` (`calor-transiente-1d`, `transcalor`, `equacao-onda`): "Passo i /
  N | t = … s" e barra proporcional; o resultado anterior esmaece. Dentro do
  loop, na mesma cadência em que os frames são amostrados (e não a cada
  passo), o Python atualiza o contador e cede o controle ao browser com
  `await asyncio.sleep(0)`, verificado empiricamente (tabela em
  `docs/v1.1.0/aceite.md`: repinta e não custa tempo). Terminados os passos, a
  mensagem muda para "Codificando animação..." até o `imageio.mimsave`
  terminar e o `<img>` ser inserido. O transiente 2D abandona o spinner
  próprio.
- `inline` (`navier-stokes`): frames ao vivo continuam visíveis (sem esmaecer)
  e com o botão Parar; o painel mostra "Passo min(i, N) / N" e o cronômetro.
  O contador podia passar do total ("402 / 400") porque o loop avança
  `steps_per_frame` passos por quadro; o texto da validação também passa a
  mostrar `min(passo, total)`, sem mudar o loop. O botão Parar **não** é
  generalizado para os outros módulos nesta versão.
- `spinner` (os seis de solução única, inclusive o escoamento potencial e a
  condução MDF, que não tinham indicador): "Executando simulação..." com barra
  indeterminada; resultado anterior esmaece. Pedido do autor em 17/10/2026:
  padronizar o "carregando" de todas as simulações, então nenhum módulo abre
  dialog por conta própria.

## 7. Descrição padronizada do problema

### 7.1 Template

Bloco `<details open>` acima do painel de parâmetros, em todas as 10 páginas,
com estado aberto/fechado lembrado em `localStorage`
(`minervamesh.desc.<módulo>`). Seções, nesta ordem e com estes títulos
(traduzíveis):

1. **Equação governante** — a EDP/EDO resolvida, em MathJax.
2. **Domínio** — intervalo ou região, com o esquema SVG (seção 8) ao lado.
3. **Condições de contorno** — escritas com símbolos, nunca com o valor
   numérico atual dos inputs (ex.: `T(0) = T₀`, não `T(0) = 0`).
4. **Condição inicial** — só nos transientes.
5. **Método numérico** — uma ou duas frases: discretização, integração no
   tempo, estabilização.
6. **Solução de referência** — a fórmula analítica ou a referência
   bibliográfica (Ghia et al. 1982 na cavidade). Só a fórmula e sua origem;
   **sem** comentários sobre propriedades do método (superconvergência,
   reprodução polinomial, etc.), que não são escopo da simulação.

Os símbolos usados nas equações e condições de contorno são os mesmos dos
rótulos dos inputs (seção 9).

### 7.2 Fonte e verificação

- MEF: texto do `resultados.tex` nas linhas da tabela da seção 3. Lacuna
  conhecida: na viga, as condições de contorno (`w = 0`, `M = 0` para apoio
  simples) só aparecem dentro da solução analítica (linha 375); a descrição
  deve cobrir também os outros casos de contorno oferecidos pelo seletor da
  página.
- MDF: derivar do código. Condução com geração: `−α T'' = Q` em `[0, L]`,
  `T(0)=0`, `T(L)=1` (fixos no código). Onda: `u_tt = c² u_xx` em `[0, L]`,
  `u(0,t) = u(L,t) = 0`, condição inicial de deslocamento senoide ou pulso
  (seletor) e de velocidade inicial `v₀` (conferir no código se é nula e
  declarar), passo `Δt = CFL·Δx/c`.
- Cada condição de contorno e inicial escrita no bloco é conferida contra o
  que o solver impõe, linha a linha, antes de entrar. Textos antigos do
  "Sobre o problema" não são reaproveitados sem essa conferência.
- **O que não estiver no TCC é verificado em fonte confiável na internet**
  (decisão do autor em 17/10/2026). Vale para as equações, soluções
  analíticas e nomes de método dos dois módulos MDF, e para qualquer fórmula
  ou atribuição a autor que o `resultados.tex` não cubra. Fontes aceitas:
  livros-texto consolidados (Incropera para condução, Reddy ou Zienkiewicz
  para MEF, LeVeque ou Strikwerda para MDF), artigos originais com DOI
  (Ghia et al. 1982; Brooks e Hughes 1982 para SUPG) e documentação oficial
  de software. Conhecimento geral do assistente não é fonte final. Cada
  verificação registra onde foi feita (página do livro, DOI ou URL) em
  `docs/v1.1.0/aceite.md`, e o que não puder ser confirmado é sinalizado ao autor em
  vez de preenchido.
- Navier-Stokes: o cenário default passa a ser **cavidade**, `Re = 100`,
  com malha e número de passos escolhidos na etapa de baseline. Como o módulo
  mostra frames ao vivo, **não há alvo de tempo total**: o autor decidiu em
  17/10/2026 que uma execução de algumas dezenas de segundos melhora a
  experiência, desde que o quadro atualize com frequência (ajustar
  `steps_per_frame` para cerca de 1 atualização por segundo) e o contador de
  passos esteja visível. Não precisa reproduzir os parâmetros do TCC. Os
  valores do TCC (`resultados.tex` 726–732: malha 41×41, `Δt = 0,01`, 2000
  passos) entram como preset "Caso de validação (Ghia et al., 1982)" no
  seletor de casos, com o tempo medido indicado (344 s no Chrome headless em
  17/10/2026). Candidatos medidos em `docs/v1.1.0/aceite.md`; **default escolhido pelo
  autor em 17/10/2026**: malha 31×31, `Δt = 0,02`, 300 passos,
  `steps_per_frame = 10` (~55 s, ψ_min a 2,4 % de Ghia). Mudança só em
  `index.html`.
  A descrição cobre os cenários com uma frase cada e um SVG por cenário
  (seção 8).

### 7.3 Renderização

MathJax 3.2.2 (`tex-mml-chtml.js` do jsDelivr, com o mesmo hash SRI já usado
nas páginas de escoamento) em todas as 10 páginas e no menu, se o menu exibir
equações. Configurar delimitadores `$…$` e `$$…$$`. Com a regra de separação
da seção 5.2, a troca de idioma nunca toca em elementos com math, então não é
preciso retypeset.

## 8. Esquemas SVG do domínio

Um SVG inline por módulo (e por cenário, nos de escoamento), no estilo do
protótipo aprovado em 17/10/2026:

- fundo do domínio `#eff6ff`, contorno `#1f2937`;
- paredes sólidas hachuradas; contornos com velocidade prescrita em azul
  `#2563eb` com setas; contornos quentes em vermelho `#dc2626`, frios em azul;
- eixos `x`, `y` discretos em cinza;
- rótulos **só com símbolos** (`u = v = 0`, `T = T_top`, `ψ = 0 on ∂Ω`), para
  não precisarem de tradução. Palavras só quando inevitável, e então com
  `data-i18n`;
- sobrescritos e subscritos (`Ω = [0, 1]²`, `T_top`, `k₁`) com `<tspan>`
  nativo do SVG (`dy` e `font-size` reduzido), não com caracteres Unicode de
  tamanho fixo nem HTML sobreposto (que desalinha quando o SVG escala).
  Frações ficam em linha (`∂T/∂n = 0`); nenhum dos 13 desenhos precisa de
  fração empilhada.

Lista de desenhos (13):

| Módulo | Esquema |
|---|---|
| condução permanente MDF | barra `[0, L]`, `T(0)=0`, `T(L)=1`, geração `Q` distribuída |
| equação da onda | corda `[0, L]`, extremidades fixas, perfil inicial (senoide ou pulso) |
| condução permanente MEF | barra, `T₀`, `T_L`, `Q` |
| transiente 1D | barra, `T(0,t)=0`, `T(L,t)=1`, divisão `k₁ / k₂` em `L/2` |
| convecção-difusão | barra, `T₀`, `T_L`, seta de `u` |
| viga | viga com apoios/engaste conforme o caso default do seletor (conferir no `index.html`), carga `q` |
| vibração | barra com o contorno do caso default do seletor (conferir no `index.html`; o TCC está inconsistente entre livre-livre e bi-engastada), `E, ρ, A` |
| transiente 2D | quadrado `[0, L]²`, `T_top`, `T_bottom`, laterais com perfil linear, `T = 0` em `t = 0` |
| potencial, cilindro | canal `L × H`, entrada `U∞`, cilindro de raio `r` em `(cx, cy)` |
| potencial, aerofólio | mesmo canal com silhueta de aerofólio (serve para NACA 0012 e 4412) |
| Navier-Stokes, cavidade | o do protótipo |
| Navier-Stokes, canal + obstáculo | canal com cilindro/retângulo (um desenho, obstáculo genérico) |
| Navier-Stokes, degrau | canal com degrau para trás |

Os esquemas de escoamento trocam com o seletor de cenário/geometria da
página (JS mostra/esconde o SVG correspondente). Legenda informa que as
dimensões são paramétricas.

## 9. Rótulos dos inputs

Padronizar em todas as páginas: `Nome símbolo [unidade]`, com o mesmo símbolo
das equações. Exemplos:

| Hoje | Novo (PT) | Novo (EN) |
|---|---|---|
| Comprimento (L) | Comprimento L [m] | Length L [m] |
| Número de elementos | Número de elementos nₑ | Number of elements nₑ |
| Condutividade primeira metade (k₁) | Condutividade k₁ (x < L/2) [W/(m·K)] | Conductivity k₁ (x < L/2) [W/(m·K)] |
| Passo de tempo (dt) | Passo de tempo Δt [s] | Time step Δt [s] |
| Nós X | Nós em x, nₓ | Nodes in x, nₓ |

Sem tabela de parâmetros separada no bloco de descrição. Grandezas
adimensionais (Re, CFL, θ) ficam sem colchetes.

## 10. Limpezas de infraestrutura

- Remover `plotly` do `config.toml` do transiente 2D (não é importado em lugar
  nenhum). Auditar os demais `config.toml` com `grep import` e remover o que
  não for usado. Sujeito ao critério de aceite da seção 2.
- Recompilar `assets/tailwind.css` (`npm run build:css`) após todas as mudanças
  de HTML; o `tailwind.config.js` já varre `.html` e `.py`.
- Todo script novo de CDN com hash SRI, como na `v1.0.0`.

## 11. Ordem de implementação

Cada etapa termina com build do Tailwind, abertura das 10 páginas e
conferência manual. Nada é commitado sem autorização do autor.

1. **Baseline**: fixar os conjuntos de parâmetros, rodar as 10 simulações na
   `v1.0.0`, registrar métricas (ou hash da imagem) e os tempos do transiente
   2D e dos dois casos do Navier-Stokes (canal default antigo e cavidade com
   os valores do TCC). Medir também candidatos a default da cavidade e fixar
   um com cadência de ~1 quadro/s e duração de algumas dezenas de segundos.
   Registrar a tabela "antes" em `docs/v1.1.0/aceite.md` (CHANGELOG fica curto, decisão do autor).
2. **Camada compartilhada e homogeneização** (seção 4.1): `assets/app.js`,
   `assets/i18n/common.*.json`, cabeçalho, modais, rodapé, ids padronizados,
   remoção do Font Awesome, larguras e títulos uniformes. Os `.py` mudam só
   os `getElementById` dos ids renomeados.
3. **i18n dos HTML**: `data-i18n` e `i18n.*.json` por módulo, seletor,
   `?lang=`, `localStorage`, menu com descrições. Textos ainda os atuais,
   traduzidos.
4. **i18n dos `.py`** (commits separados): trocar literais por `i18n.t`.
   Conferir métricas.
5. **Progresso**: carga inicial (cronômetro + texto); contador e yield nos
   quatro módulos de marcha (commits separados); Navier-Stokes só contador.
   Conferir métricas e tempo do transiente 2D.
6. **Descrições**: template, textos reescritos a partir do TCC e do código,
   MathJax em todas as páginas, default do Navier-Stokes para cavidade.
   Conferência de cada condição de contorno contra o solver e verificação em
   fonte externa confiável do que não está no TCC (seção 7.2), com as fontes
   registradas em `docs/v1.1.0/aceite.md`.
7. **SVGs**: 13 esquemas, troca por cenário nos de escoamento, remoção das
   imagens antigas.
8. **Inputs**: rótulos com símbolo e unidade.
9. **Limpezas**: `plotly`, auditoria dos `config.toml`, README com resumo EN.
10. **Fechamento**: métricas "depois" em `docs/v1.1.0/aceite.md` e entrada
    curta no `CHANGELOG.md`, revisão do EN pelo
    autor, tag `v1.1.0`, espelho `gh-pages`, `scripts/sync_minervamesh.sh` no
    fork do labmfa, atualização do draft PR #1.

## 12. Fora de escopo desta versão

- Tradução dos comentários dos `simulation.py`.
- Botão Parar fora do Navier-Stokes.
- Migração dos módulos de GIF para frames ao vivo.
- Modo worker do PyScript.
- Atualização do PyScript (2026.x) e do Prism (1.30.0), já anotadas no
  `CLAUDE.md`.
- Ilustração decorativa (gerada por IA ou não) para o menu ou para a página do
  LabMFA.
- Esta especificação é versionada neste repositório mas **não** entra no PR do
  labmfa.

## 13. Riscos e cuidados

- Textos antigos de "Sobre o problema" foram escritos por IA em versões
  anteriores e já continham erros; por isso a conferência contra o solver é
  obrigatória, não opcional.
- `js.i18n` precisa existir antes do primeiro `Run`; o guarda da seção 5.2
  garante que, se o `app.js` falhar, a simulação roda mostrando as chaves em
  vez de quebrar.
- Renomear ids de botão e saída exige tocar nos `getElementById` dos `.py`;
  é edição não numérica, mas erro de id quebra o módulo em silêncio. Cada
  página é aberta e executada após a etapa 2.
- O yield dentro do loop muda a ordem de repintura, não a aritmética; ainda
  assim as métricas são conferidas módulo a módulo.
- Achado lateral no TCC, fora deste repo: em `resultados.tex` a legenda da
  tabela da vibração diz "bi-engastada" (linha 439) e o texto diz
  "livre-livre" (450 e 458). Corrigir no repositório do TCC.
