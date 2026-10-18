"""
MinervaMesh — ponte entre o simulation.py e a página (assets/app.js).

Compartilhado pelos módulos via [[fetch]] no config.toml, como o mmplot.py.
Uso em simulation.py:
    from mmcommon import i18n_t, progress

i18n_t(chave): texto traduzido (rótulos de gráfico e mensagens), de
    i18n.<lang>.json via window.i18n. Sem o app.js, devolve a própria chave.
progress(método, *args): window.mmProgress.start/step/encoding/done/fail,
    o indicador de progresso da página. Sem o app.js, não faz nada.
Nenhuma destas funções participa do cálculo.
"""
import js

_i18n = getattr(js, "i18n", None)
_progress = getattr(js, "mmProgress", None)


def i18n_t(key):
    return str(_i18n.t(key)) if _i18n else key


def progress(method, *args):
    if _progress:
        getattr(_progress, method)(*args)
