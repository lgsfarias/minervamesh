# Changelog

Registro de versões da plataforma MinervaMesh. Formato livre, em português,
uma seção por versão, da mais recente para a mais antiga.

## v1.2.0 (17/10/2026)

Repaginação da interface (protótipo aprovado pelo autor no Claude Design;
especificação em `docs/v1.2.0/spec.md`):

- Menu em galeria: card por simulação com miniatura real do resultado,
  badges de método, dimensão e regime, e chips de filtro; marca MinervaMesh
  própria (o logo da Poli saiu; a afiliação vai no rodapé).
- Página de simulação com o resultado como foco: faixa com a equação
  governante e o esquema do domínio sempre visível; enunciado completo na
  coluna larga, num seletor Resultado | Problema; código-fonte em popup com
  números de linha; figura em largura total com a validação embaixo.
- Saída de exemplo (parâmetros padrão) antes do primeiro Run, no lugar da
  área vazia.
- Tema claro e escuro (botão no cabeçalho, segue o sistema por padrão) e
  tipografia IBM Plex; tokens de cor em variáveis CSS.
- Celular: resultado antes dos parâmetros, barra do Run fixa no rodapé da
  janela, equações com rolagem própria; nenhuma página rola na horizontal.
- Modo embutido (`data-mm-embedded`) para a hospedagem no site do LabMFA.
- `simulation.py` exibidos ao aluno mais enxutos: o bloco repetido de i18n e
  progresso (15 linhas por módulo) virou `assets/mmcommon.py`; no transiente
  1D, dois comentários duplicados saíram e três enganosos foram corrigidos.
  Os comentários que explicam os passos do método ficaram. Nenhuma linha
  numérica mudou (AST idêntica, `scripts/ast-check.py`).
- Resultados numéricos inalterados em relação à v1.1.0 (texto das métricas
  idêntico nos 10 módulos; verificação em `docs/v1.2.0/aceite.md`).
- Ferramentas: `scripts/shots.mjs` (capturas), `scripts/thumbs.mjs`
  (miniaturas), `scripts/probe.mjs` (sonda de estados), `scripts/ast-check.py`.

## v1.1.0 (17/10/2026)

Melhorias para a hospedagem no site do LabMFA:

- Interface bilíngue (inglês padrão, português a um clique).
- Indicador de progresso na carga inicial e nas simulações com marcha no tempo.
- Descrição padronizada de cada problema: equação, domínio, condições de
  contorno e inicial, método numérico e solução de referência, com esquema do
  domínio em SVG (um por cenário nos módulos de escoamento).
- Rótulos dos parâmetros com símbolo e unidade, iguais aos das equações.
- Gráfico interativo do Matplotlib (zoom, pan, exportação) nos seis módulos
  de solução única.
- Páginas homogeneizadas (cabeçalho, modais, ícones, larguras, ids) com a
  camada compartilhada em `assets/app.js`.
- Navier-Stokes abre na cavidade (Re = 100) com preset de validação Ghia et
  al. (1982).
- `plotly` removido das dependências do transiente 2D (carga ~10 MB menor).
- Resultados numéricos inalterados em relação à v1.0.0 (verificação em
  `docs/v1.1.0/aceite.md`).

## v1.0.0 (06/10/2026)

Preparação para produção: Tailwind compilado localmente, hashes SRI nos
scripts de CDN, rodapé com autoria e licença, gerador de malha fora do menu.
Tag `v1.0.0`; espelho em `gh-pages`.

## v1.0-tcc (set/2026)

Estado da plataforma que gerou as figuras do relatório do TCC
(`lgsfarias/tcc-minervamesh`).
