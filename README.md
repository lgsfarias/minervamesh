# MinervaMesh 🌐

MinervaMesh é um simulador educacional interativo para estudos de transferência de calor e mecânica dos fluidos, utilizando o Método dos Elementos Finitos (MEF). O objetivo do projeto é proporcionar uma interface moderna e intuitiva para experimentos computacionais.

---

## 🚀 **Funcionalidades**

- Simulação de condução de calor em 1D com geração interna.
- Solução numérica e comparação com solução analítica.
- Interface gráfica interativa utilizando **PyScript** e **TailwindCSS**.

---

## 🔧 **Instalação e Execução**

1️⃣ **Clone o repositório:**

```bash
git clone https://github.com/lgsfarias/minervamesh.git
cd minervamesh
```

2️⃣ **Inicie um servidor HTTP simples:**
Para que o PyScript consiga acessar os arquivos locais, é necessário um servidor HTTP. Você pode iniciar um diretamente pelo Python:

```bash
# Python 3.x
python3 -m http.server 8000
```

3️⃣ **Acesse no navegador:**
Abra no navegador:

```
http://localhost:8000/index.html
```

---

## 🛠️ **Tecnologias Utilizadas**

- **PyScript** — Execução de Python no navegador.
- **TailwindCSS** — Estilização moderna e responsiva.
- **Matplotlib** — Geração de gráficos.
- **NumPy** — Cálculos numéricos.

---

## 📌 **Próximos Passos**

- [ ] Adicionar simulação em 2D e regime transiente.
- [ ] Modelagem de escoamento de fluidos em canais.
- [ ] Integração com Gmsh para geração de malhas complexas.

---

## 👨‍💻 **Autor**

- **Luiz Gustavo Farias**
  [GitHub](https://github.com/lgsfarias) | [LinkedIn](https://www.linkedin.com/in/lgsfarias/)

---
