# Aceite da v1.1.0: baseline e verificação numérica

Material de desenvolvimento desta versão. Acompanha a [especificação](spec.md)
(seção 2) e as [tarefas](tasks.md). Não entra no changelog nem no PR do LabMFA.

## Critério de aceite: métricas idênticas antes e depois

Os `simulation.py` recebem apenas edições não numéricas (strings, DOM,
cedência de controle ao browser, ids). Para garantir que nada mudou nos
resultados, cada módulo é executado com um conjunto fixo de parâmetros antes e
depois das mudanças, com `scripts/baseline.mjs` (Chrome headless), e as
métricas exibidas sob o gráfico têm de ser idênticas. Onde o módulo não exibe
métrica em texto (equação da onda, vibração, condução MDF), a comparação é
pelo hash SHA-256 da imagem gerada. Arquivos completos em
`docs/v1.1.0/baseline/`.

Conjunto de parâmetros: os defaults de cada página na `v1.0.0`, listados no
campo `params` de `docs/v1.1.0/baseline/v1.0.0-defaults.json`.

O hash da imagem só serve de critério até a primeira edição que toque no
gráfico (tradução de rótulos, troca de GIF por PNG). Antes disso entra um
gancho que exporta os resultados numéricos para `window.__minervamesh_result`
(spec, seção 2, passo 0); a partir daí a comparação é pelos números. O gancho é temporário e sai no fechamento da versão; para reinserir,
aplicar de novo o commit da etapa 1b (`git cherry-pick`) e rodar
`node scripts/baseline.mjs --label <rótulo>`.

### Antes (v1.0.0, 17/10/2026)

| Módulo | Execução (s) | Métricas exibidas | Imagem (sha256, 16 hex) |
|---|---|---|---|
| mdf-conducao | 1.7 | (gráfico em canvas; comparar pelo hash) | e09710f3ad8b5685<br>aaeb8c3558e1780d |
| mdf-onda | 4.9 | (só imagem) | fe2bcc08555a7ee3 |
| mef-conducao | 0.4 | Malha: 10 elementos lineares, h = 0.1000 m T_max (MEF): 1.120000 °C em x = 0.700 m T_max (analítica): 1.125000 °C em x = 0.750 m Erro máximo nodal: 8.88e-16 °C | b6790ae42f805931 |
| calor-transiente-1d | 7.6 | Tempo final: 0.200 s (100.0% de τ_dif = 0.20 s) T(L/2) MEF: 0.163656 °C T(L/2) analítica (regime permanente, k2/(k1+k2)): 0.166667 °C Desvio em x = L/2: 3.01e-03 °C | 498cf6d45b0deb8d |
| conveccao-difusao | 0.6 | Informações da Simulação Número de Péclet: 0.500 Estabilização SUPG: Sim Parâmetro τ: 0.032791 Velocidade: 0.050 m/s Tamanho elemento: 0.0200 m Erro máximo: 4.44e-16 °C | a30875179d03a3c2 |
| viga | 1.1 | Caso: Simplesmente Apoiada com carga uniforme, propriedades constantDeflexão máxima: 1.0417 mm (analítica 1.0417 mm, erro relativo 8.60e-13) em x = 1.000 m\|M\| máximo: 5.0083 kN·m (analítica 5.0000 kN·m, erro relativo 1.67e-03)\|V\| máximo: 9.5000 kN (analítica 10.0000 kN, erro relativo 5.00e-02) | 5a97054cc0504b39 |
| vibracao | 1.2 | (só imagem) | fc9480596bad8e77 |
| transcalor | 6.2 | Tempo final: 0.050 s (5.0% da escala difusiva τ_dif = 1.00 s)T(0,5; 0,5) MEF: 27.0799 °CT(0,5; 0,5) regime permanente (T_bot + (T_top − T_bot)·y): 65.8974 °CDesvio do regime permanente: 38.8175 °C (aumentar n_steps para convergir) | 7c3f213f15483e9a |
| potencial | 1.4 | Resultados Aerodinâmicos Sustentação (L): -0.0013 N/m Arrasto (D): 0.0158 N/m Circulação (Γ): -0.0013 m²/s : C_p na superfície do cilindro (teoria: C_p = 1 − 4 sen²θ)θ = 0°: C_p MEF = 0.892 \| teórico = 1.000 (Δ = -0.108)θ = 90°: C_p MEF = -2.926 \| teórico = -3.000 (Δ = +0.074)θ = 180°: C_p MEF = 0.976 \| teórico = 1.000 (Δ = - | b45f37b28e5bafc6 |
| navier-stokes | 127.5 | Passo: 500 / 500, Re = 100.0, malha 60×30ψ_min: +0.00000 em (0.000, 0.000)Simulação Concluída! | a0286f6ead411156 |

A coluna de métricas é um resumo truncado; a referência para comparação é o campo `text` (e `extra_text`) completo no JSON. Tempos de execução medidos em Chrome headless num MacBook (ver `date` no JSON); servem de ordem de grandeza, não de benchmark.

Observações da baseline:

- O default do Navier-Stokes na `v1.0.0` (canal com obstáculo, malha 60×30,
  500 passos) leva cerca de 2 minutos e não tem referência para comparar
  (ψ_min = 0). A `v1.1.0` troca o default para a cavidade.
- Com os valores do TCC (cavidade 41×41, Δt = 0,01, 2000 passos) a execução
  no Chrome headless levou 344 s e deu ψ_min = −0,10296 em (0,600; 0,725),
  contra ψ_min ≈ −0,1034 em (0,6172; 0,7344) de Ghia et al. (1982). Esse caso
  vira o preset "Caso de validação".
- Cada quadro renderizado pelo Matplotlib no Navier-Stokes custa cerca de
  0,9 s; o passo de tempo custa ~0,10 s em 31×31 e ~0,15 s em 41×41. A
  cadência de quadros pesa tanto quanto a malha na duração total.
- O contador do Navier-Stokes pode passar do total (ex.: "402 / 400") porque o
  loop avança `steps_per_frame` passos por quadro. É só exibição; a `v1.1.0`
  corrige o texto sem mudar o loop.

### Candidatos a default da cavidade (Chrome headless, 17/10/2026)

Todos com Re = 100, tampa u = 1, domínio [0, 1]². Referência Ghia et al.
(1982): ψ_min ≈ −0,1034 em (0,6172; 0,7344).

| Caso | Malha | Δt | Passos | t final | Quadros | Execução | ψ_min | Centro do vórtice |
|---|---|---|---|---|---|---|---|---|
| TCC (preset de validação) | 41×41 | 0,01 | 2000 | 20 | 40 | 344 s | −0,10296 | (0,600; 0,725) |
| A | 31×31 | 0,01 | 600 | 6 | 60 | 112 s | −0,10091 | (0,633; 0,733) |
| B | 41×41 | 0,01 | 400 | 4 | 67 | 122 s | −0,09858 | (0,625; 0,750) |
| **C (escolhido pelo autor em 17/10/2026)** | 31×31 | 0,02 | 300 | 6 | 20 | 45 s | −0,10090 | (0,633; 0,733) |
| D | 31×31 | 0,01 | 400 | 4 | 20 | 53 s | −0,09804 | (0,633; 0,767) |
| E | 21×21 | 0,02 | 400 | 8 | 20 | 33 s | −0,10140 | (0,600; 0,750) |

O caso C chega a 2,4 % do ψ_min de referência em 45 s, com o centro do
vórtice na célula certa. O E é mais rápido mas a malha 21×21 é grosseira
demais para o campo ficar legível. Com `steps_per_frame = 10` em vez de 15, o
C passa a atualizar a cada ~1,8 s e dura ~55 s. Arquivos:
`docs/v1.1.0/baseline/ns-cavity-*.json`.

### Depois (v1.1.0)

Pendente; preenchido ao final da etapa 10.

## Gancho de exportação validado (17/10/2026)

Rodada `v1.0.0-hook` (código da v1.0.0 mais o gancho `_export_result`, nada
além disso): hashes de imagem e textos exibidos idênticos aos de
`v1.0.0-defaults` nos 10 módulos, e o resultado numérico exportado em todos.
Os hashes abaixo são a **referência numérica** para a comparação "depois".

| Módulo | Imagem igual à v1.0.0 | Texto igual | Chaves exportadas | sha256/16 do resultado |
|---|---|---|---|---|
| mdf-conducao | sim | sim | x, T, T_analytical, nx, alpha, Q | `f4bec9c0ec3bf719` |
| mdf-onda | sim | sim | x, u, nt, dt, mode | `f9ee818ac6205a34` |
| mef-conducao | sim | sim | x, T, T_ana_mesh, e_max, T_max_mef, x_max_mef | `37b2c108a97899e2` |
| calor-transiente-1d | sim | sim | x_nodes, T, nt, dt, T_meio_mef, erro_meio | `60e236980bf88b4a` |
| conveccao-difusao | sim | sim | x, T, T_ana_mesh, erro_max, Pe_local, tau, use_supg | `da38aa9b91d42295` |
| viga | sim | sim | x, w, M, V, w_max_mef, x_wmax, M_abs_max, V_abs_max | `0cb54a229a4859de` |
| vibracao | sim | sim | x, frequencies, freq_analytical, eigenvectors | `c85c2f02e695cf19` |
| transcalor | sim | sim | X, Y, u, n_steps, dt, T_centro_mef, erro_centro | `a21b6afac4adabb1` |
| potencial | sim | sim | psi, cp_sorted, angles_sorted, Lift_dim, Drag_dim, Gamma_dim | `231d13de637773a3` |
| navier-stokes | sim | sim | psi, omega, current_step, total_steps, stopped | `908c0a7730520e05` |

Arquivo: `docs/v1.1.0/baseline/v1.0.0-hook.json` (contém os arrays completos).

## Verificações intermediárias

Cada etapa que toca em `.py` ou em HTML termina com uma rodada completa do
`scripts/baseline.mjs` comparada à referência `v1.0.0-hook`. Os JSONs
intermediários não são guardados; só o veredito.

| Etapa | Data | Imagens | Texto exibido | `__minervamesh_result` |
|---|---|---|---|---|
| 2 — camada compartilhada, ids novos, 10 páginas convertidas | 17/10/2026 | idênticas (10/10) | idêntico (10/10) | idêntico (10/10) |
| 2b — gráfico interativo (6 estáticos), layout largo, tabela da vibração em HTML | 17/10/2026 | mudaram nos 6 interativos (esperado); idênticas nos 4 demais | mudou onde a figura virou canvas (título/toolbar entram no texto) | idêntico (10/10) |

## Armadilha: cache do browser

Na primeira rodada com o gancho o resultado veio vazio porque o perfil
persistente do Chrome servia o `simulation.py` antigo do cache (o
`http.server` do Python não manda `Cache-Control`). O `scripts/baseline.mjs`
passou a servir os arquivos do repositório com `no-store`. Vale o mesmo aviso
para teste manual: depois de editar um `.py`, recarregar a página ignorando o
cache (Shift + recarregar), senão o PyScript roda a versão anterior.

## Fontes verificadas

Pendente; lista de onde cada fórmula ou afirmação fora do TCC foi conferida
(etapa 6 da spec).
