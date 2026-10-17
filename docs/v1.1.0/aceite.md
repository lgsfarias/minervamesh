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
`scripts/baseline.mjs --lang pt` comparada à referência `v1.0.0-hook` com
`scripts/compare-baseline.mjs` (desconta o prefixo título/toolbar do gráfico
interativo; a partir da etapa 3 o default do site é `en`, por isso o `--lang`).
Os JSONs intermediários não são guardados; só o veredito.

| Etapa | Data | Imagens | Texto exibido | `__minervamesh_result` |
|---|---|---|---|---|
| 2 — camada compartilhada, ids novos, 10 páginas convertidas | 17/10/2026 | idênticas (10/10) | idêntico (10/10) | idêntico (10/10) |
| 2b — gráfico interativo (6 estáticos), layout largo, tabela da vibração em HTML | 17/10/2026 | mudaram nos 6 interativos (esperado); idênticas nos 4 demais | mudou onde a figura virou canvas (título/toolbar entram no texto) | idêntico (10/10) |
| 3 — i18n dos HTML (data-i18n, JSON por módulo, menu bilíngue); nenhum `.py` tocado | 17/10/2026 | idênticas às da 2b (10/10) | idêntico fora o prefixo título/toolbar dos 6 interativos e a tabela da vibração em HTML (mudanças da 2b); Python ainda escreve em PT | idêntico (10/10) |
| 4 — i18n dos `.py`: literais de gráfico e de validação viram `i18n_t("py.…")`, PT byte a byte igual aos literais anteriores; baseline em `--lang pt` | 17/10/2026 | idênticas às da 3 (10/10) | idêntico (10/10) fora a tabela da vibração em HTML (2b), conferido com `scripts/compare-baseline.mjs` | idêntico (10/10) |
| 5 — progresso: `progress("start"/"step"/"encoding"/"done")` nos 10 `.py`, `await asyncio.sleep(0)` na cadência dos quadros nos 3 módulos de GIF, `min(passo, total)` no texto do Navier-Stokes, condução MDF em `@when` assíncrono, invólucro `*_guarded` com `try/finally` nos 8 handlers que não tinham (reabilita o botão se a simulação lançar exceção; testado com a vibração e 0 elementos) | 17/10/2026 | idênticas às da 4 (10/10) | idêntico (10/10) fora a tabela da vibração em HTML (2b) | idêntico (10/10) |
| 5b — modo de falha: `mmProgress.fail()` no `app.js` (reabilita o botão, esconde painel e linha de status, sem "Tempo total") e `except Exception` nos 10 invólucros escrevendo `py.error` em `#plot-output` (os 8 que só tinham `try/finally` e os 2 de escoamento, que já mostravam a mensagem). Caminho de falha testado em Chrome headless: vibração com 0 elementos, onda com `nx` vazio e potencial com `nx = 0`; nos três o botão volta a "Executar", sem painel, sem linha de status, sem esmaecimento, mensagem na área do resultado; rodar de novo com valores válidos termina com "Tempo total" | 17/10/2026 | idênticas às da 5 (10/10) | idêntico (10/10) fora a tabela da vibração em HTML (2b) | idêntico (10/10) |
| 6 — descrições: MathJax e `<details>` nas 10 páginas, textos novos, `section.method`; Navier-Stokes com default cavidade só no `index.html`. Nenhum `.py` tocado. Navier-Stokes rodado duas vezes: com os parâmetros antigos do canal via `--only navier-stokes --set …` (comparável à referência) e com o default novo (cavidade 31×31, Δt 0,02, 300 passos: ψ_min = −0,10090 em (0,633; 0,733), 54,8 s, igual ao candidato C) | 17/10/2026 | idênticas às da 5 (10/10, Navier-Stokes com os parâmetros antigos) | idêntico (10/10) fora a tabela da vibração em HTML (2b) | idêntico (10/10) |
| 6b — pendência da etapa 4 (decisão do autor em 17/10/2026): `prop_type` cru vira `i18n_t("option.prop_type." + prop_type).lower()` na viga e na vibração, e o `geo_type.capitalize()` do título do potencial vira `i18n_t("option.geo_type." + geo_type)`; rodada só desses 3 módulos (`--only`) | 17/10/2026 | potencial muda (título da figura traduzido; já diferia desde a 2b); viga e vibração idênticas às da 6 | viga e vibração: "propriedades constant" → "propriedades constante" (mudança pretendida); potencial idêntico | idêntico (3/3) |
| 6c — CDN centralizado: Prism sob demanda e MathJax injetados pelo `app.js`; `<head>` das 10 páginas só com PyScript, Tailwind e `app.js`. Nenhum `.py` tocado; Navier-Stokes de novo nas duas configurações | 17/10/2026 | idênticas às da 6b (10/10) | idêntico (10/10) fora a tabela da vibração (2b) e "propriedades constante" na viga e na vibração (6b) | idêntico (10/10) |

## Etapa 5: yield ao browser e tempo do transiente 2D (17/10/2026)

Teste do `asyncio.sleep(0)` no PyScript 2025.8.1 (Pyodide 0.28.1), thread
principal, Chrome headless: página descartável com um loop de 30 iterações de
100 ms de trabalho NumPy, escrevendo o índice no DOM e cedendo o controle ao
final de cada iteração. Medidos os quadros pintados (`requestAnimationFrame`)
e os valores intermediários lidos de fora durante o loop:

| Yield | Tempo do loop | Quadros pintados | Valores intermediários vistos |
|---|---|---|---|
| nenhum (bloqueante) | 3,02 s | 4 | 1 |
| `await asyncio.sleep(0)` | 3,02 s | 35 | 31 |
| `await asyncio.sleep(0.01)` | 3,35 s | 82 | 30 |

`sleep(0)` repinta e não custa tempo; `sleep(0.01)` custa 11 ms por yield.
Adotado `sleep(0)` nos três módulos de GIF (o Navier-Stokes mantém o `0.01`
que já tinha, por não ser uma edição necessária).

Tempo do transiente 2D (defaults, Chrome headless, 50 passos e 50 yields):
6,4 s antes da etapa, 6,7 a 7,0 s depois em quatro rodadas. Diferença dentro
da variação entre rodadas (a referência `v1.0.0-hook` marcou 6,4 s e a
`v1.0.0-defaults` 6,2 s).

### Tamanho da carga inicial (17/10/2026, sem cache)

Bytes baixados até `py:ready`, medidos com Puppeteer (CDP `Network`), por
conjunto de pacotes do `config.toml`:

| Pacotes | Na rede | Descomprimido | Maiores itens |
|---|---|---|---|
| numpy, matplotlib (condução MDF e MEF, viga, convecção) | 18,2 MB | 25,7 MB | matplotlib 6,6; numpy 3,1; pyodide.asm.wasm 2,7; stdlib 2,4 |
| numpy, matplotlib, imageio (onda, transiente 1D) | 18,5 MB | 26,0 MB | idem |
| numpy, matplotlib, scipy (vibração, potencial, Navier-Stokes) | 33,7 MB | 42,1 MB | scipy 13,2; libopenblas 1,9 |
| numpy, matplotlib, scipy, plotly, imageio (transiente 2D) | 43,8 MB | 51,4 MB | plotly 9,7 (vem do PyPI, sai na etapa 9) |

Decisão do autor em 17/10/2026: o texto sobre o download não entra na interface
(poluía); a carga mostra só "Preparando o ambiente de simulação..." com
cronômetro, discreto, junto ao botão Executar.

## Armadilha: cache do browser

Na primeira rodada com o gancho o resultado veio vazio porque o perfil
persistente do Chrome servia o `simulation.py` antigo do cache (o
`http.server` do Python não manda `Cache-Control`). O `scripts/baseline.mjs`
passou a servir os arquivos do repositório com `no-store`. Vale o mesmo aviso
para teste manual: depois de editar um `.py`, recarregar a página ignorando o
cache (Shift + recarregar), senão o PyScript roda a versão anterior.

## Fontes verificadas (etapa 6, 17/10/2026)

Cada condição de contorno e inicial do bloco "Sobre o problema" foi conferida
contra o `simulation.py` do módulo (linha das imposições de Dirichlet, penalidade
ou vetor inicial). O que está no `resultados.tex` do TCC usa o TCC como fonte;
o restante foi conferido nas fontes abaixo. Fórmulas analíticas elementares
foram ainda verificadas por substituição direta na EDO e nas condições de
contorno.

| Módulo | Afirmação | Fonte |
|---|---|---|
| condução MDF | solução parabólica de $-\alpha T'' = Q$ com $T(0)=0$, $T(L)=1$: $T = -\frac{Q}{2\alpha}x^2 + (\frac{1}{L} + \frac{QL}{2\alpha})x$ | forma geral $T = -\frac{e_{gen}}{2k}x^2 + C_1 x + C_2$ em F. Arınç, *ME-212 Thermo-Fluids Engineering II*, METU, 2013, "Heat Conduction, Steady 1D", Example 2 (<https://blog.metu.edu.tr/farinc/files/2025/02/Heat-Conduction_Steady-1D_8-Sept-2023.pdf>); constantes por substituição direta |
| condução MDF | diferenças centradas de 2ª ordem para $T''$ | LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007, ISBN 978-0-898716-29-0 (referência geral; sem número de equação conferido) |
| onda | leapfrog: diferenças centradas em $x$ e $t$; condição de Courant $c\,\Delta t/\Delta x \le 1$ | A. B. Adib, *Study Notes on Numerical Solutions of the Wave Equation with the Finite Difference Method*, arXiv:physics/0009068 (2000), eq. 2.13 (condição de Courant) e seção 2.5 (leapfrog); LeVeque 2007 como referência geral |
| onda | senoide: $u = \sin(\pi x/L)\cos(\pi c t/L)$ (separação de variáveis, extremidades fixas) | J. Wei, *MATH316 Lecture 23*, CUHK, 2015 (<https://www.math.cuhk.edu.hk/~wei/MATH316-102-Lecture23.pdf>): autofunções $\sin(n\pi x/L)$ com $\cos(n\pi c t/L)$ |
| onda | pulso: d'Alembert com extensão ímpar $2L$-periódica, $u_t(x,0)=0$ | R. Herman, *d'Alembert's Solution for g(x) = 0*, UNCW (<https://people.uncw.edu/hermanr/pde1/dAlembert/Ex3/index.html>): extensão ímpar em $x=0$ e $x=L$, $u = \tfrac12[F(x+ct)+F(x-ct)]$; mesma conclusão na Lecture 23 da CUHK |
| viga | balanço com carga uniforme: $w = \frac{q}{24EI}(6L^2x^2 - 4Lx^3 + x^4)$, $w_{\max} = qL^4/8EI$, $M_{\max} = qL^2/2$, $V_{\max} = qL$ | Budynas e Nisbett, *Shigley's Mechanical Engineering Design*, Tabela A-9, caso 3 (cópia em <http://www.faculty.fairfield.edu/wdornfeld/ME311/ShigleyAppendix.pdf>): $y = \frac{wx^2}{24EI}(4lx - x^2 - 6l^2)$, $y_{\max} = -\frac{wl^4}{8EI}$, $M_1 = wl^2/2$, $R_1 = wl$ |
| viga | biengastada com carga uniforme: $w = \frac{q x^2 (L-x)^2}{24EI}$, $w_{\max} = qL^4/384EI$, $M_{\max} = qL^2/12$, $V_{\max} = qL/2$ | Shigley, Tabela A-9, caso 16: $y = -\frac{wx^2}{24EI}(l-x)^2$, $y_{\max} = -\frac{wl^4}{384EI}$, $M_1 = M_2 = wl^2/12$, $V = \frac{w}{2}(l - 2x)$; também MIT 2.080J *Structural Mechanics*, Lecture 5, eq. 5.46a (<https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/3533c046dafcc488f1432e92d05ec210_MIT2_080JF13_Lecture5.pdf>) |
| viga | simplesmente apoiada com carga uniforme | TCC, `resultados.tex` 375–386 (e Shigley A-9, caso 7) |
| vibração | livre-livre $f_n = (n-1)c/2L$; engastada-engastada $f_n = nc/2L$; engastada-livre $f_n = (2n-1)c/4L$, $c = \sqrt{E/\rho}$ | L. San Andrés, *MEEN 617 HD#14 Vibrations of Continuous Systems*, TAMU, 2008/2012 (<https://rotorlab.tamu.edu/me617/HD14%20Vibrations%20of%20bars%20and%20beams.pdf>): Example 1 (fixed-free) $\omega_k = \frac{(2k-1)\pi}{2L}\sqrt{E/\rho}$; Example 2 (free-free) $\omega_k = \frac{k\pi}{L}\sqrt{E/\rho}$, $k = 0, 1, \ldots$; Example 3 (fixed-fixed) $\omega_k = \frac{k\pi}{L}\sqrt{E/\rho}$, $k = 1, 2, \ldots$. Livre-livre também no TCC (`resultados.tex` 458–465) |
| Navier-Stokes | Ghia, Ghia e Shin (1982), *J. Comput. Phys.* 48(3), 387–411, $\psi_{\min} \approx -0{,}1034$ em $(0{,}6172;\ 0{,}7344)$ | DOI 10.1016/0021-9991(82)90058-4 conferido no Crossref (<https://api.crossref.org/works/10.1016/0021-9991(82)90058-4>); valores do TCC, `resultados.tex` 739 |
| demais (condução MEF, transiente 1D, convecção-difusão/SUPG, transiente 2D, potencial/$C_p$ do cilindro, Thom, formulação $\psi$-$\omega$) | equações, CCs e referências | TCC, `resultados.tex` 61–143, 149–237, 243–333, 505–588, 594–689, 695–779 |

Divergências encontradas entre o código e o TCC ou os textos antigos (o bloco
de descrição segue o código):

- **Potencial, saída do canal**: o TCC (`resultados.tex` 610) diz "entrada e
  saída: ψ linear em y". No código, `outlet_nodes` é calculado mas não entra em
  `dirichlet_psi`; na saída vale a condição natural $\partial\psi/\partial x = 0$.
  A descrição diz isso. Corrigir no repositório do TCC.
- **Vibração, engastada-apoiada**: aplica as mesmas penalidades da
  engastada-engastada (`u(0) = u(L) = 0`) e não tem referência analítica no
  código. A descrição deixa explícito que, no modelo axial, os dois casos
  coincidem. Opção redundante do seletor; decisão de manter ou remover é do
  autor.
- **Vibração, nota de resultado**: o texto antigo dizia "método de penalidade e
  malha refinada"; o código não refina malha. Nota corrigida nas duas línguas.
- **Transiente 2D**: o TCC escreve $c_p$; o input da página é $c_v$ e o código
  multiplica `rho * cv`. A descrição usa $c_v$. Domínio $[0,1]^2$ e malha
  $40 \times 40$ são fixos no código (não são inputs).
- **Transiente 1D**: $T(0,t) = 0$ e $T(L,t) = 1$ são fixos no código, não
  inputs; a descrição diz "fixo no código".
- **Viga**: as funções `analytical_solution_*` com $q = 1$ são código morto; a
  referência exibida vem do bloco de métricas (carga uniforme × apoiada, balanço,
  biengastada). A descrição cobre só esses três casos.
- **Condução MDF**: $L = 1$, $T(0) = 0$ e $T(L) = 1$ fixos no código; o input
  `nx` é o total de nós (o solver usa `nx − 2` internos).
