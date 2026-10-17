"""
MinervaMesh — gráfico interativo compartilhado pelos módulos estáticos.

O backend do Matplotlib que o Pyodide traz desenha a figura num <canvas> dentro
de #plot-output (app.js aponta document.pyodideMplTarget para lá), com toolbar
de zoom, pan, coordenadas do cursor e exportação. Estas funções cuidam só da
apresentação; nenhuma delas participa do cálculo.

Uso em simulation.py (o config.toml do módulo busca este arquivo via [[fetch]]):

    from mmplot import plot_begin, plot_fig, plot_show

    plot_begin()
    fig, ax = plt.subplots(**plot_fig(8, 4))
    ...
    plot_show("Título da figura")
"""
import matplotlib.pyplot as plt
from js import document

DPI_MAX = 100  # dpi padrão do Matplotlib; o backend multiplica pelo devicePixelRatio


def plot_fig(width_in, height_in):
    """kwargs para plt.figure()/plt.subplots(): figsize original e dpi ajustado ao container.

    O backend não é responsivo: o canvas nasce com figsize × dpi pixels. Em vez de
    encolher o figsize (o que deixaria títulos e rótulos, medidos em pontos,
    grandes demais e sobrepostos), mantém-se o figsize com que o gráfico foi
    desenhado e reduz-se o dpi, de modo que a figura inteira escala como uma
    imagem, fontes incluídas.
    """
    try:
        w_px = document.getElementById("plot-output").clientWidth - 8
    except Exception:
        w_px = 0
    dpi = DPI_MAX if w_px < 300 else min(DPI_MAX, w_px / width_in)
    return {"figsize": (width_in, height_in), "dpi": dpi}


def plot_begin():
    """Fecha a figura anterior (o backend remove o <div> dela) e limpa a área."""
    plt.close("all")
    document.getElementById("plot-output").innerHTML = ""


def plot_show(title):
    """Mostra a figura atual com o título do módulo no lugar de 'Figure 1'.

    O backend (webagg) manda o rótulo da figura ao conectar o canvas, usando
    Figure.get_label(); se estiver vazio, cai em "Figure N". Por isso o título
    vai no label da figura, além do título da janela.
    """
    fig = plt.gcf()
    fig.set_label(title)
    fig.canvas.manager.set_window_title(title)
    plt.show()


def plot_append_html(html):
    """Anexa um bloco HTML (métricas, validação) abaixo da figura, sem apagá-la."""
    document.getElementById("plot-output").insertAdjacentHTML("beforeend", html)
