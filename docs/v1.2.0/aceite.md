# Aceite da v1.2.0: verificação numérica e visual

Material de desenvolvimento desta versão. Acompanha a [especificação](spec.md)
(seção 3) e as [tarefas](tasks.md). Não entra no changelog nem no PR do LabMFA.

## Critério

A v1.2.0 é uma repaginação de interface. Os `simulation.py` receberam só
duas edições, ambas não numéricas e em commit próprio: o bloco repetido de
i18n/progresso virou `assets/mmcommon.py`, e no transiente 1D saíram dois
comentários duplicados e três comentários enganosos foram corrigidos
(`python3 scripts/ast-check.py 3af7591` reproduz a verificação: AST idêntica
à do commit da refatoração nos 12 `.py`, docstrings incluídas).

Referência numérica: a rodada final da v1.1.0
(`docs/v1.1.0/baseline/final-sem-gancho.partial.json` e
`final-sem-gancho-ns-canal.partial.json`), defaults da v1.1.0 em PT. Como o
gancho `__minervamesh_result` foi removido no fechamento da v1.1.0, o
critério é **texto das métricas exibidas idêntico** (texto é o que o aluno
compara com a solução analítica) e, como reforço, o hash SHA-256 das imagens.
Rodadas desta versão em `docs/v1.2.0/baseline/` (`scripts/baseline.mjs`,
Chrome headless).

## Resultado final (17/10/2026)

`node scripts/baseline.mjs --label v1.2.0-final` × referência:

| Módulo | Execução v1.1.0 (s) | Execução v1.2.0 (s) | Métricas exibidas | Imagem |
|---|---|---|---|---|
| mdf-conducao | 1.8 | 2.5 | igual | difere (dpi, ver abaixo) |
| mdf-onda | 5.0 | 8.9 | igual | igual |
| mef-conducao | 0.3 | 0.4 | igual | igual |
| calor-transiente-1d | 7.8 | 12.8 | igual | igual |
| conveccao-difusao | 0.6 | 0.9 | igual | difere (dpi) |
| viga | 0.6 | 1.6 | igual | difere (dpi) |
| vibracao | 0.8 | 1.5 | igual | difere (dpi) |
| transcalor | 7.0 | 11.6 | igual | igual |
| potencial | 1.4 | 1.4 | igual | difere (dpi) |
| navier-stokes (cavidade, default) | 54.0 | 55.8 | igual | igual |
| navier-stokes (canal antigo, `--set scenario=channel,L=2.0,H=1.0,nx=60,ny=30,Re=100.0,dt=0.001,steps_per_frame=10,total_steps=500`) | 130.0 | 132.0 | igual | igual |

Os tempos da v1.2.0 foram medidos com outro processo ocupando a máquina
(agente de limpeza de comentários em paralelo) e não são comparáveis; o do
Navier-Stokes, medido sozinho, bate com o da v1.1.0.

**Texto das métricas idêntico nos 10 módulos e nas duas configurações do
Navier-Stokes.** É o critério de aceite; cumprido.

### Por que cinco hashes de imagem mudaram

Os cinco módulos cuja imagem difere (condução MDF, convecção-difusão, viga,
vibração, potencial) são exatamente os que desenham no backend interativo do
Matplotlib via `assets/mmplot.py`. `plot_fig` calcula o dpi da figura a
partir do `clientWidth` de `#plot-output` no instante do clique, para a
figura caber na coluna. A v1.2.0 mudou essa largura duas vezes:

1. Etapa 4: o card de resultado passou a ocupar a largura toda (antes, 3/4 de
   uma grade); dpi maior em quatro módulos. Condução MEF não mudou porque sua
   figura (800 px) já cabia e o dpi estava no teto de 100.
2. Etapa 5: a saída de exemplo antes do Run deixou a página da condução MDF
   mais alta que a janela; a barra de rolagem vertical apareceu e o
   `clientWidth` caiu 15 px. Diferença de 36 bytes no PNG, estável em três
   rodadas consecutivas (`etapa10-mmcommon`, mais duas rodadas de controle).

Os cinco módulos de GIF/PNG (onda, transiente 1D, transiente 2D, Navier-Stokes
nas duas configurações) desenham em tamanho fixo e os hashes batem. Era o
risco previsto na spec (seção 10): mesma aritmética, pixels diferentes.

## Rodadas intermediárias

| Rodada | Quando | Texto | Imagens |
|---|---|---|---|
| `etapa4` (layout novo, antes da saída de exemplo) | 17/10/2026 | 10/10 iguais à v1.1.0 | 5 diferem (dpi, item 1) |
| `etapa10-mmcommon` (guarda de i18n/progresso em `mmcommon.py`) | 17/10/2026 | 10/10 iguais à `etapa4` | 9/10 iguais à `etapa4`; condução MDF difere (item 2) |
| `v1.2.0-final` (limpeza de comentários, HTML limpo) | 17/10/2026 | 10/10 iguais à v1.1.0 | idem `etapa10-mmcommon` |

## Verificação visual

`scripts/shots.mjs` captura as 11 páginas em 1440, 820 e 390 px, PT e EN,
claro e escuro, e acusa rolagem horizontal; `scripts/probe.mjs` abre um
módulo, executa (Run, seletor Resultado | Problema, popup do código) e captura.
Capturas em `docs/v1.2.0/shots/` (fora do git): `antes/` (v1.1.0), uma pasta
por etapa e `probe/`. Rolagem horizontal a 390 px: três páginas na v1.1.0
(transiente 2D, potencial, Navier-Stokes), nenhuma na v1.2.0.

Conferência final pelo autor: abrir cada página, executar e olhar o
resultado nos dois temas (`node scripts/shots.mjs --label depois --themes
light,dark` gera as 132 capturas).
