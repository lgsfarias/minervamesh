"""Pós-processamento das miniaturas (chamado por scripts/thumbs.mjs).

Uso: python3 thumbs_crop.py <entrada.png|gif> <saida_full.png|-> <saida_thumb.webp> x y w h

- GIF: usa o último quadro.
- saida_full: figura inteira, 1000 px de largura, fundo branco (saída de exemplo
  antes do Run), PNG quantizado a 256 cores. "-" pula essa saída (execuções com
  parâmetros alternativos).
- saida_thumb: região (x, y, w, h), em frações da figura, normalmente só a área
  dos eixos de um subplot, encaixada em 16:9 e redimensionada para 384 × 216 px
  (card do menu; o tile tem ~120 px em CSS, então 384 é 1,6× em telas Retina).
  A região nunca é cortada: a que for mais larga que 16:9 ganha faixas brancas
  em cima e embaixo, a mais alta ganha faixas nas laterais. WebP com perda,
  qualidade 80 (v1.2.2): nos campos 2D fica 4× menor que o PNG, sem diferença
  visível nesse tamanho.
"""
import sys
from PIL import Image, ImageSequence

src, out_full, out_thumb, x, y, w, h = sys.argv[1:8]
x, y, w, h = (float(v) for v in (x, y, w, h))
TW, TH = 384, 216

im = Image.open(src)
if getattr(im, "n_frames", 1) > 1:
    frames = list(ImageSequence.Iterator(im))
    im = frames[-1].convert("RGBA")
else:
    im = im.convert("RGBA")

# Fundo branco (canvas do webagg é transparente fora dos eixos)
bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
bg.alpha_composite(im)
im = bg.convert("RGB")
W, H = im.size

def save_png(img, path):
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


msg = f"{W}x{H}"
if out_full != "-":
    full = im.resize((1000, round(H * 1000 / W)), Image.LANCZOS) if W != 1000 else im
    save_png(full, out_full)
    msg += f" -> full {full.size[0]}x{full.size[1]}"

# Região escolhida, encaixada em 16:9 com faixas brancas, depois 384 × 216
rx, ry, rw, rh = round(x * W), round(y * H), round(w * W), round(h * H)
region = im.crop((rx, ry, rx + rw, ry + rh))
cw, ch = region.size
s = min(TW / cw, TH / ch)
region = region.resize((max(1, round(cw * s)), max(1, round(ch * s))), Image.LANCZOS)
thumb = Image.new("RGB", (TW, TH), "white")
thumb.paste(region, ((TW - region.width) // 2, (TH - region.height) // 2))
thumb.save(out_thumb, "WEBP", quality=80, method=6)
print(f"{msg}, thumb {TW}x{TH} webp (regiao {rw}x{rh} -> {region.width}x{region.height})")
