# MinervaMesh 🌐

MinervaMesh é uma plataforma educacional interativa para simulações numéricas em transferência de calor, mecânica dos fluidos e análise estrutural. O projeto oferece uma interface moderna e intuitiva para experimentos computacionais utilizando métodos numéricos avançados.

---

## 🚀 **Funcionalidades**

### **Método das Diferenças Finitas (MDF)**
- **Condução Permanente 1D com Geração Interna** - Solução da equação de Poisson com fonte
- **Equação da Onda 1D** - Simulação de propagação de ondas com diferentes condições de contorno

### **Método dos Elementos Finitos (MEF)**
- **Condução Permanente 1D** - Solução por elementos finitos lineares
- **Condução Transiente 1D** - Análise temporal com condutividade variável
- **Convecção-Difusão 1D** - Estabilização SUPG para alta convecção
- **Vibração 1D** - Problema de autovalor para frequências naturais
- **Viga 1D** - Análise de flexão e deformação
- **Simulação Transiente 2D** - Transferência de calor bidimensional

### **Ferramentas Extras**
- **Gerador de Malha** - Criação de malhas estruturadas

---

## 🏗️ **Estrutura do Projeto**

```
📁 mesh/
├── 📄 index.html                    # Menu principal
├── 📁 assets/                       # Assets centralizados
├── 📁 simulations/                  # Simulações organizadas
│   ├── 📁 mdf/                     # Método das Diferenças Finitas
│   ├── 📁 mef/                     # Método dos Elementos Finitos
│   └── 📁 extra/                   # Ferramentas extras
├── 📁 shared/                       # Código compartilhado
│   ├── 📁 python/                  # Utilitários Python
│   └── 📁 config/                  # Configurações globais
├── 📁 src/                         # Componentes modulares
│   ├── 📁 components/              # Templates HTML
│   ├── 📁 styles/                  # CSS centralizado
│   └── 📁 scripts/                 # JavaScript comum
└── 📁 examples/                    # Exemplos e referências
```

---

## 🔧 **Instalação e Execução**

### **1️⃣ Clone o repositório:**
```bash
git clone https://github.com/lgsfarias/minervamesh.git
cd minervamesh
```

### **2️⃣ Inicie um servidor HTTP:**
Para que o PyScript funcione corretamente, é necessário um servidor HTTP local:

```bash
# Python 3.x
python3 -m http.server 8000

# Ou usando Node.js
npx http-server -p 8000

# Ou usando PHP
php -S localhost:8000
```

### **3️⃣ Acesse no navegador:**
```
http://localhost:8000/index.html
```

---

## 🛠️ **Tecnologias Utilizadas**

### **Frontend**
- **PyScript** — Execução de Python no navegador
- **TailwindCSS** — Framework CSS para interface moderna
- **Prism.js** — Syntax highlighting para código
- **Plotly** — Visualizações interativas (CFD)

### **Computação Client-Side**
- **NumPy** — Cálculos numéricos e álgebra linear (executados no navegador)
- **SciPy** — Algoritmos científicos avançados (executados no navegador)
- **Matplotlib** — Geração de gráficos e visualizações (executados no navegador)
- **Meshio** — Manipulação de arquivos de malha (executados no navegador)

### **Métodos Numéricos**
- **Método das Diferenças Finitas** — Discretização espacial
- **Método dos Elementos Finitos** — Formulação variacional
- **Estabilização SUPG** — Controle de oscilações numéricas
- **Integração de Gauss** — Quadratura numérica

---

## ⚡ **Arquitetura Client-Side**

MinervaMesh utiliza uma arquitetura única onde **toda a computação numérica acontece diretamente no navegador** através do PyScript. Isso significa:

- ✅ **Sem necessidade de servidor backend** - Apenas um servidor HTTP simples para servir arquivos
- ✅ **Computação distribuída** - Cada usuário executa os cálculos em sua própria máquina
- ✅ **Privacidade total** - Dados não saem do dispositivo do usuário
- ✅ **Escalabilidade natural** - Não há limitação de servidor central
- ✅ **Execução offline** - Funciona sem conexão com internet após carregamento inicial

### **Como Funciona**
1. O navegador carrega o PyScript runtime
2. Os arquivos Python são executados diretamente no cliente
3. Bibliotecas científicas (NumPy, SciPy, Matplotlib) rodam no navegador
4. Resultados são renderizados instantaneamente na interface

---

## 📚 **Simulações Disponíveis**

### **Transferência de Calor**
| Simulação | Método | Descrição |
|-----------|--------|-----------|
| Condução Permanente 1D | MDF/MEF | Solução com/sem geração interna |
| Condução Transiente 1D | MEF | Análise temporal com k(x) variável |
| Convecção-Difusão 1D | MEF | Estabilização SUPG para alta convecção |
| Transiente 2D | MEF | Transferência de calor bidimensional |

### **Mecânica Estrutural**
| Simulação | Método | Descrição |
|-----------|--------|-----------|
| Vibração 1D | MEF | Problema de autovalor para frequências naturais |
| Viga 1D | MEF | Análise de flexão e deformação |

### **Mecânica dos Fluidos**
| Simulação | Método | Descrição |
|-----------|--------|-----------|
| Equação da Onda 1D | MDF | Propagação de ondas |
| Simulação CFD | MEF | Dinâmica dos fluidos computacional |

---

## 🎯 **Características Técnicas**

### **Interface Moderna**
- ✅ Design responsivo e intuitivo
- ✅ Componentes modulares reutilizáveis
- ✅ Loading states e feedback visual
- ✅ Syntax highlighting para código fonte

### **Arquitetura Modular**
- ✅ Separação por método numérico (MDF/MEF)
- ✅ Código compartilhado centralizado
- ✅ Configurações padronizadas
- ✅ Estrutura escalável para novas simulações

### **Precisão Numérica**
- ✅ Comparação com soluções analíticas
- ✅ Análise de convergência
- ✅ Validação de resultados
- ✅ Documentação técnica detalhada

---

## 📖 **Como Usar**

1. **Escolha uma simulação** no menu principal
2. **Configure os parâmetros** físicos e numéricos
3. **Execute a simulação** e visualize os resultados
4. **Compare com soluções analíticas** quando disponível
5. **Analise a convergência** variando o número de elementos

### **Dicas de Uso**
- Use números de elementos maiores para maior precisão
- Verifique o número de Péclet em problemas convectivos
- Compare diferentes condições de contorno
- Analise a estabilidade numérica dos resultados

---

## 🤝 **Contribuição**

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-simulacao`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova simulação'`)
4. Push para a branch (`git push origin feature/nova-simulacao`)
5. Abra um Pull Request

### **Áreas de Contribuição**
- Novas simulações e métodos numéricos
- Melhorias na interface e UX
- Otimizações de performance
- Documentação e exemplos
- Testes e validação

---

## 📄 **Licença**

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 **Autor**

**Luiz Gustavo Santos Farias**
- 🎓 Estudante de Engenharia Mecânica - UFRJ
- 🔬 Pesquisador em Métodos Numéricos
- 💻 Desenvolvedor Full-Stack

**Contato:**
- [GitHub](https://github.com/lgsfarias)
- [LinkedIn](https://www.linkedin.com/in/lgsfarias/)
- [Email](mailto:lgsfarias@outlook.com)

---

## 🙏 **Agradecimentos**

- **Escola Politécnica da UFRJ** - Suporte acadêmico
- **Comunidade PyScript** - Framework para Python no navegador
- **Contribuidores** - Feedback e melhorias
- **Professores e Colegas** - Inspiração e conhecimento

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela! ⭐**

</div>