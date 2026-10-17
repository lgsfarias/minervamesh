# Changelog

Registro de versões da plataforma MinervaMesh. Formato livre, em português,
uma seção por versão, da mais recente para a mais antiga.

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
