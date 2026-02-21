# MinervaMesh 🌐

Plataforma educacional para simulações numéricas de transferência de calor, mecânica dos fluidos e análise estrutural, executada **integralmente no navegador** com PyScript.

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
- Gerador de Malha Triangular

---

## 🏗️ Estrutura do Projeto

```
minervamesh/
├── index.html                              # Menu principal
├── favicon.ico
├── assets/                                 # Logos e imagens globais
│   ├── logo_poli_ufrj.png
│   └── conducao_esquema.png
├── simulations/                            # Simulações organizadas
│   ├── mdf/                               # Método das Diferenças Finitas
│   │   ├── conducao-permanente-barra1d-geracao/
│   │   └── equacao-onda/
│   ├── mef/                               # Método dos Elementos Finitos
│   │   ├── conducao-permanente-barra1d-mef/
│   │   ├── calor-transiente-1d/
│   │   ├── conveccao-difusao-1d/
│   │   ├── viga-1d-mef/
│   │   ├── vibracao-1d-mef/
│   │   ├── transcalor/
│   │   └── escoamento-fluido-2d/
│   │       ├── common.py                  # Matrizes MEF compartilhadas
│   │       ├── geometry.py                # Geração de geometrias
│   │       ├── escoamento-potencial/
│   │       └── navier-stokes/
│   └── extra/
│       └── gerador-malha/
├── examples/                               # Exemplos e referências
├── refs/                                   # Material de referência
├── tcc_minervamesh/                        # Documento LaTeX do TCC
└── docs/                                   # Artefatos de controle do projeto
```

Cada simulação segue o padrão:
```
simulacao/
├── index.html         # Interface (parâmetros + resultado)
├── simulation.py      # Lógica numérica em Python
└── config.toml        # Dependências PyScript (numpy, scipy, etc.)
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

---

## 🛠️ Tecnologias

| Camada | Tecnologia | Papel |
|---|---|---|
| Runtime | PyScript 2025.8.1 / Pyodide | Execução de Python no navegador via WebAssembly |
| Cálculo | NumPy, SciPy | Álgebra linear, solvers esparsos |
| Visualização | Matplotlib | Gráficos e campos 2D |
| Interface | TailwindCSS | Estilização responsiva |
| Código | Prism.js | Syntax highlighting do código-fonte |

---

## 📖 Como Usar

1. Escolha uma simulação no menu principal
2. Configure os parâmetros físicos e numéricos
3. Clique em **Rodar** e visualize os resultados
4. Use **Ver Código** para inspecionar a implementação Python
5. Compare com soluções analíticas (quando disponível)

---

## 👨‍💻 Autor

**Luiz Gustavo Santos Farias**
- 🎓 Engenharia Mecânica — Escola Politécnica, UFRJ
- [GitHub](https://github.com/lgsfarias) · [LinkedIn](https://www.linkedin.com/in/lgsfarias/) · [Email](mailto:lgsfarias@outlook.com)

## 📄 Licença

MIT — veja [LICENSE](LICENSE).