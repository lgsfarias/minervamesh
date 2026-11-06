# Especificação da v1.2.1 — cards do menu com o problema físico

Data: 2026-11-06. Origem: avaliação da v1.2.0 pelo orientador em 19/10/2026
("ficou bom demais", com um único pedido: nos cards, o problema físico
ilustrado no lugar do resultado, e três pequenas figuras com os resultados
abaixo). Protótipo no Claude Design, com as duas páginas (proposta inicial e
sugestões aplicadas): <https://claude.ai/artifact/Weim6sDjJf39pd4f2pbW7j>.

## 1. Decisões do autor (19/10/2026)

- As três miniaturas são os **painéis da figura padrão** quando o módulo
  tem três ou mais subplots; nos módulos de figura única, **três execuções
  com parâmetros diferentes**.
- Sugestões do protótipo aceitas na íntegra: miniaturas 16:9 recortadas só
  na área dos eixos e sem legenda; descrição em duas linhas; esquemas de
  condução MEF e MDF diferenciados; equação governante sob o esquema nos
  casos 1D, em tipografia matemática (MathJax na plataforma).

## 2. O card

De cima para baixo:

1. **Esquema do problema físico** (`figure.mm-scheme.mm-card-scheme`, altura
   fixa de 200 px): o `app.js` busca a página de cada módulo e copia o
   primeiro `figure.mm-scheme svg` (o da geometria padrão; potencial e
   Navier-Stokes têm outros, paramétricos, depois dele). Uma fonte só: o SVG
   continua na aba Problema do módulo, com as classes e os tokens de cor da
   v1.2.0, por isso funciona nos dois temas. Nos módulos 1D a equação
   governante (`menu.json` → `eq`, LaTeX em display) vai abaixo do SVG, no
   espaço que sobrava; nos 2D o esquema já preenche o painel.
2. **Faixa de três miniaturas** (`assets/thumbs/<id>-1..3.png`, 480 × 270,
   `aspect-video`, `object-cover`): fundo `surface2`, opacidade 0,9 em
   repouso e 1 no hover, sem legenda.
3. **Texto**: badges, título e descrição em duas linhas (`line-clamp-2`); o
   texto completo segue na página do módulo.

Grade: `minmax(340px, 1fr)`: três colunas a 1360 px, duas em tablet, uma no
celular. O card é sempre vertical.

## 3. Miniaturas (`assets/menu.json` → `thumbs`, `scripts/thumbs.mjs`)

Cada módulo lista três entradas `{x, y, w, h}` (frações da figura, só a
área dos eixos, sem título, barra de cores nem legenda) e, opcionalmente,
`set` (id do input → valor) para rodar com outros parâmetros. Entradas com o
mesmo `set` compartilham uma execução; a execução sem `set` é a padrão e a
única que grava a figura inteira (`<id>.png`, saída de exemplo da v1.2.0).
`scripts/thumbs_crop.py` encaixa a região em 16:9 com faixas brancas, nunca
corta. Os `-sm.png` da v1.2.0 saem.

Campos 2D quadrados (transiente 2D) entram inteiros, com faixas brancas
laterais: a primeira tentativa, uma faixa 16:9 do topo do quadrado, perdia a
forma do campo e ampliava a figura (ficava borrada). Variar o passo de tempo
também foi descartado: dt = 0,02 oscila (Crank-Nicolson com dt grande para a
malha), por isso as variantes mudam k com o dt padrão, o que equivale a olhar
instantes diferentes com o mesmo custo de execução.

| Módulo | Miniatura 1 | Miniatura 2 | Miniatura 3 |
|---|---|---|---|
| Navier-Stokes | ψ | ω | linhas de corrente |
| Potencial | ψ | linhas de corrente | Cp × teoria |
| Condução transiente 2D | k = 1 (padrão, t = 0,05 s) | k = 0,25 (difusão lenta, camada quente rasa) | k = 2,5 (difusão rápida, perto do permanente) |
| Vibração | modo 2 | modo 3 | modo 4 |
| Viga | deflexão | momento fletor | cortante |
| Convecção-difusão | Pe = 50, SUPG (padrão) | u = 0,005 (Pe = 5) | sem SUPG com 10 elementos (Peₑ = 5: oscila) |
| Calor transiente 1D | k₁ = 5, k₂ = 1 (padrão) | k₁ = k₂ = 1 | k₁ = 1, k₂ = 5 |
| Condução permanente MEF | Q = 4 (padrão) | Q = 0 | Q = 12 |
| Onda | senoide, t = 2,0 s (padrão) | senoide, t = 1,0 s | pulso |
| Condução permanente MDF | nx = 10 (padrão) | nx = 5 | nx = 40 |

## 4. Esquemas MEF × MDF

Os dois esquemas de condução permanente 1D eram idênticos. Agora o MEF mostra
a barra dividida em dez elementos, um deles em destaque com a cota `h`; o
MDF mostra os nós e o estêncil de três pontos (i−1, i, i+1) em destaque. O
rótulo Q (ou Q, k) subiu para o topo. Classes novas em `.mm-scheme`: `el`,
`elh`, `node`, `nodeh`. Vale também na aba Problema de cada módulo.

## 5. Regra de ouro e aceite

Nenhuma alteração em `simulation.py`, `common.py` e `geometry.py`. Aceite:

1. `assets/thumbs/<id>.png` (figura inteira) regenerado **idêntico** ao da
   v1.2.0 nos módulos de figura estática (mesmo Pyodide, mesmos defaults):
   prova de que a plataforma segue reproduzindo os resultados. Resultado em
   06/11/2026: idênticos byte a byte em Navier-Stokes, transiente 2D, calor
   transiente 1D, onda e condução MEF; diferentes em potencial, viga,
   vibração, convecção-difusão e condução MDF, os módulos de `mmplot`
   (canvas webagg), cujo bitmap depende da largura da coluna de resultado no
   momento da captura (dpi), o mesmo efeito documentado em
   `docs/v1.2.0/aceite.md`. As figuras de 17/10 foram capturadas no layout
   antigo; as curvas são as mesmas.
2. Menu sem overflow horizontal a 390 px, sem erro de console, 10 esquemas
   e 7 equações presentes, nos dois temas (`scripts/shots.mjs`).
3. `node scripts/i18n-check.mjs` limpo (não há chave nova).
4. Conferência visual pelo autor no `localhost`.
