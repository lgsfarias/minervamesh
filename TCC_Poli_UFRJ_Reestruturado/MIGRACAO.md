# Migração do TCC MinervaMesh para o template oficial Poli/UFRJ

Este diretório contém o TCC **MinervaMesh** reestruturado sobre o template oficial atual
da Escola Politécnica/UFRJ (classe `poli.cls` v2.0). O projeto original em
`../tcc_minervamesh/` permanece **intacto**.

## Como compilar

```bash
cd TCC_Poli_UFRJ_Reestruturado
latexmk -pdf thesis.tex
```

O `latexmkrc` já cuida do ciclo `pdflatex → bibtex → makeindex (lista de abreviaturas) →
pdflatex ×2`. Build manual equivalente:

```bash
pdflatex thesis && bibtex thesis && makeindex -s poli.ist -o thesis.lab thesis.abx && pdflatex thesis && pdflatex thesis
```

Resultado: `thesis.pdf` (210 páginas, idêntico em conteúdo ao `Principal.pdf` antigo).

## Estrutura

```
thesis.tex              # raiz: preâmbulo, metadados, includes
thesis.bib              # bibliografia (cópia de biblio.bib, 46 entradas)
poli.cls / poli-unsrt.bst / poli.ist / latexmkrc   # template oficial (não editar)
Logo_poli.jpg / Logo_mec.jpg                       # logos usados pela capa/folha de aprovação
Pre-textual/  dedic.tex  thanks.tex  resumo.tex  abstract.tex
Textual/      introducao  revisao_bibliografica  fundamentacao_teorica
              resultados  arquitetura  conclusao
Pos-textual/  apendice_modulos_1d  apendice_modulos_2d  apendice_benchmark
Imagens/      benchmark_scaling.pdf + prints/*.png (21 figuras)
```

## Principais mudanças estruturais (template antigo → novo)

| Antes | Depois |
|---|---|
| `\documentclass{book}` + `TesePack.tex` + `.sty` legados (`pagina`, `nomes`, `indentfirst`, `fancyheadings`) | `\documentclass[grad,pdftex]{poli}` (layout, margens, cabeçalhos e rótulos PT-BR pela classe) |
| Capa e folha de aprovação escritas à mão (`Capa.tex`) | **Geradas automaticamente** pela classe a partir dos metadados (`\title`, `\author`, `\advisor`, `\examiner`, `\department`, `\date`) via `\maketitle`/`\frontmatter` |
| Sem ficha catalográfica | **Ficha catalográfica gerada automaticamente** (com as 5 palavras-chave de `\keyword`) |
| Resumo/Abstract como `\chapter`-like manuais em `Preambulo.tex` | Ambientes nativos `abstract` / `foreignabstract` (cabeçalho institucional automático) |
| Seção "Siglas" manual | Lista de Abreviaturas nativa via `\abbrev{}{}` + `\printloabbreviations` |
| Bibliografia `coppe.bst` | `poli-unsrt.bst` (numérico por ordem de aparição) |
| `\includegraphics{figuras/prints/...}` | Figuras em `Imagens/`; `\graphicspath{{Imagens/}{Imagens/prints/}}` |
| — | Bloco `\lstset` com `pythonstyle`/`dirstyle` e mapeamento `literate` de símbolos Unicode **portado do `TesePack.tex`** para o `thesis.tex` (necessário para as listagens Python dos apêndices) |
| `\nocite{*}` do template de exemplo | Removido — apenas referências efetivamente citadas aparecem |

O conteúdo acadêmico (texto, equações, tabelas, figuras, labels, citações, apêndices) foi
migrado **1:1, sem alteração**. Os 6 capítulos foram preservados na mesma ordem.

## Validação realizada

- Compila sem erros fatais; **0 referências indefinidas, 0 citações indefinidas, 0 labels duplicados**.
- 46 entradas bibliográficas, todas citadas (sem órfãs).
- 21 figuras na Lista de Figuras, 13 tabelas na Lista de Tabelas, 6 capítulos + 3 apêndices no sumário.
- Verificação visual: capa, folha de aprovação (3 assinaturas), ficha, resumo/abstract, páginas de
  conteúdo com figuras, tabelas, equações, referências cruzadas e listagens Python (acentos OK).

## Ajustes manuais restantes (decisão do autor)

1. **Examinadores**: `\examiner{Prof.}{Examinador Um}{D.Sc.}` e `{Examinador Dois}` em `thesis.tex`
   são *placeholders* — substituir pelos nomes e graus reais da banca.
2. **Labels `cap4`/`cap5`**: mantidos verbatim do original (são semânticos: `cap5`=Resultados,
   `cap4`=Arquitetura) para preservar todas as `\ref` cruzadas. Renomear quebraria referências —
   só alterar se renomear todas as ocorrências em conjunto.
3. **Palavras-chave**: aparecem agora na ficha catalográfica (padrão do template), não ao final do
   resumo como no documento antigo. Se desejar exibi-las também ao fim do resumo, adicionar manualmente.
4. **Dedicatória**: condensada para o formato curto em itálico do comando `\dedication{}` do template
   (conteúdo preservado em substância). Revisar redação se desejar.
5. **`biblio.bib`/`thesis.bib`**: aviso benigno do BibTeX `empty pages in Clough1960` (campo `pages`
   ausente nessa entrada, pré-existente). Preencher se o dado for conhecido — não foi inventado.
6. **Imagens não usadas**: `Imagens/minerva.pdf`, `Logo_poli2.jpg`, `Imagens/newton-method.png`,
   `Imagens/PINNExemploDesenho.png` são resíduos do template de exemplo — podem ser removidos.
