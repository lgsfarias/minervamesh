"""
MinervaMesh — gráfico interativo compartilhado pelos módulos estáticos.

O backend do Matplotlib que o Pyodide traz desenha a figura num <canvas> dentro
de #plot-output (app.js aponta document.pyodideMplTarget para lá), com toolbar
de zoom, pan, coordenadas do cursor e exportação. Estas funções cuidam só da
apresentação; nenhuma delas participa do cálculo.

Uso em simulation.py (o config.toml do módulo busca este arquivo via [[fetch]]):

    from mmplot import plot_begin, plot_size, plot_show

    plot_begin()
    fig, ax = plt.subplots(figsize=plot_size(8, 4))
    ...
    plot_show("Título da figura")
"""
import matplotlib.pyplot as plt
from js import document

DPI = 100  # dpi padrão do Matplotlib; o backend multiplica pelo devicePixelRatio


def plot_size(default_w, default_h):
    """figsize em polegadas ajustado à largura da área de resultado.

    O backend não é responsivo: o canvas nasce com figsize × dpi pixels. Por isso
    a largura é lida do container e a altura segue a proporção pedida.
    """
    try:
        w_px = document.getElementById("plot-output").clientWidth - 8
    except Exception:
        w_px = 0
    if w_px < 300:
        return (default_w, default_h)
    w_in = w_px / DPI
    return (w_in, w_in * default_h / default_w)


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
