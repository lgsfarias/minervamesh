# MinervaMesh 🌐

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plataforma educacional para simulações numéricas de transferência de calor, mecânica dos fluidos e análise estrutural, executada **integralmente no navegador** com PyScript. Interface bilíngue (inglês por padrão, português no seletor).

> **English summary.** MinervaMesh is an educational platform for numerical simulation (finite difference and finite element methods) in heat transfer, fluid mechanics and structural analysis. Everything runs in the browser through PyScript/Pyodide, with no backend. The interface is bilingual (English by default, Portuguese available). Each module states the governing equation, domain, boundary conditions, numerical method and reference solution, and compares the numerical result with an analytical solution or a benchmark case. Developed as an undergraduate thesis project in Mechanical Engineering at Escola Politécnica, UFRJ. MIT license.

---

## 🚀 Funcionalidades

### Método das Diferenças Finitas (MDF)
- Condução Permanente 1D com Geração Interna
- Equação da Onda 1D

### Método dos Elementos Finitos (MEF)

| Simulação | Tipo | Destaques |
|---|---|---|
| Condução Permanente 1D | Térmico | Poisson 1D, solução analítica |
| Condução Transiente 1D | Térmico | k(x) variável, método θ |
| Convecção-Difusão 1D | Transporte | Estabilização SUPG |
| Viga 1D (Euler-Bernoulli) | Estrutural | Flexão e deformação |
| Vibração 1D | Estrutural | Problema de autovalor |
| Transferência de Calor 2D | Térmico | Malha triangular P1 |
| Escoamento Potencial 2D | Fluidos | Laplace 2D |
| Navier-Stokes 2D | Fluidos | Vorticidade-Corrente, Thom BC |

### Ferramentas Extras
- Gerador de Malha Triangular (utilitário de apoio, fora do menu principal; acessível em `simulations/extra/gerador-malha/`)

---

## 🏗️ Estrutura do Projeto

```
minervamesh/
├── index.html                              # Menu principal
├── favicon.ico
├── LICENSE
├── assets/                                 # Camada compartilhada: app.js (i18n, cabeçalho, progresso), i18n/common.*.json, mmplot.py, CSS, logo
└── simulations/                            # Simulações organizadas por método
    ├── mdf/                               # Método das Diferenças Finitas
    │   ├── conducao-permanente-barra1d-geracao/
    │   └── equacao-onda/
    ├── mef/                               # Método dos Elementos Finitos
    │   ├── conducao-permanente-barra1d-mef/
    │   ├── calor-transiente-1d/
    │   ├── conveccao-difusao-1d/
    │   ├── viga-1d-mef/
    │   ├── vibracao-1d-mef/
    │   ├── transcalor/
    │   └── escoamento-fluido-2d/
    │       ├── common.py                  # Matrizes MEF compartilhadas
    │       ├── geometry.py                # Geração de geometrias
    │       ├── escoamento-potencial/
    │       └── navier-stokes/
    └── extra/
        └── gerador-malha/
```

Cada simulação segue o padrão:
```
simulacao/
├── index.html         # Interface (descrição do problema, parâmetros, resultado)
├── simulation.py      # Lógica numérica em Python
├── config.toml        # Dependências PyScript (numpy, scipy, etc.)
└── i18n.pt.json / i18n.en.json   # Textos da página nos dois idiomas
```

---

## ⚡ Arquitetura Client-Side

Toda a computação numérica acontece no navegador através do PyScript/Pyodide:

```
HTML → PyScript runtime → Pyodide (CPython em WebAssembly) → NumPy/SciPy/Matplotlib
```

- **Sem backend** — apenas um servidor HTTP simples para servir arquivos estáticos
- **Custo de infraestrutura ≈ $0** — compatível com GitHub Pages, Netlify, etc.
- **Computação distribuída** — cada usuário executa os cálculos em sua própria máquina
- **Portabilidade** — funciona em qualquer navegador moderno (Chrome, Firefox, Safari, Edge)

---

## 🔧 Instalação e Execução

```bash
# 1. Clone o repositório
git clone https://github.com/lgsfarias/minervamesh.git
cd minervamesh

# 2. Inicie um servidor HTTP local
python3 -m http.server 8000

# 3. Acesse no navegador
# http://localhost:8000/index.html
```

O CSS do Tailwind já vem compilado em `assets/tailwind.css`. Só é preciso recompilar se alterar classes nas páginas:

```bash
npm install
npm run build:css
```

---

## 🛠️ Tecnologias

| Camada | Tecnologia | Papel |
|---|---|---|
| Runtime | PyScript 2025.8.1 / Pyodide | Execução de Python no navegador via WebAssembly |
| Cálculo | NumPy, SciPy | Álgebra linear, solvers esparsos |
| Visualização | Matplotlib | Gráficos e campos 2D |
| Interface | TailwindCSS 3 (CSS compilado pelo CLI, versionado em `assets/`) | Estilização responsiva |
| Equações | MathJax 3 | Renderização das equações na descrição de cada problema |
| Código | Prism.js | Syntax highlighting do código-fonte |

---

## 📖 Como Usar

1. Escolha o idioma (PT | EN) e uma simulação no menu principal
2. Leia o bloco **Sobre o problema** (equação, domínio, condições de contorno, método, solução de referência)
3. Configure os parâmetros físicos e numéricos
4. Clique em **Executar** e acompanhe o progresso; o resultado aparece com as métricas de validação
5. Use **Ver código-fonte** para inspecionar a implementação Python

---

## 🎓 Trabalho de Conclusão de Curso

Esta plataforma foi desenvolvida como Projeto de Graduação em Engenharia Mecânica (Escola Politécnica, UFRJ), sob orientação do Prof. Gustavo Rabello dos Anjos, e aprovada em setembro de 2026. O relatório em LaTeX, com a fundamentação teórica e a validação de cada módulo contra soluções analíticas e casos de referência, está em repositório próprio: [lgsfarias/tcc-minervamesh](https://github.com/lgsfarias/tcc-minervamesh).

A tag [`v1.0-tcc`](https://github.com/lgsfarias/minervamesh/releases/tag/v1.0-tcc) marca o estado da plataforma usado para gerar as figuras do relatório.

---

## 👨‍💻 Autor

**Luiz Gustavo Santos Farias**
- 🎓 Engenharia Mecânica — Escola Politécnica, UFRJ
- [GitHub](https://github.com/lgsfarias) · [LinkedIn](https://www.linkedin.com/in/lgsfarias/) · [Email](mailto:lgsfarias@outlook.com)

## 📄 Licença

MIT — veja [LICENSE](LICENSE).