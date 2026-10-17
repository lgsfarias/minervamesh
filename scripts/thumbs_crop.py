"""Pós-processamento das miniaturas (chamado por scripts/thumbs.mjs).

Uso: python3 -I thumbs_crop.py <entrada.png|gif> <saida_full.png> <saida_sm.png> x y w h

- GIF: usa o último quadro.
- saida_full: figura inteira, 1000 px de largura, fundo branco (saída de exemplo).
- saida_sm: região (x, y, w, h), em frações da figura, ajustada a 4:3 e
  redimensionada para 480 × 360 px (card do menu). Região mais larga que 4:3
  (um subplot baixo) ganha faixas brancas em cima e embaixo, para não perder o
  eixo x; região mais alta é cortada no centro.
- Os dois PNGs são quantizados a 256 cores (gráficos não perdem nada visível e
  o arquivo cai a uma fração).
"""
import sys
from PIL import Image, ImageSequence

src, out_full, out_sm, x, y, w, h = sys.argv[1:8]
x, y, w, h = (float(v) for v in (x, y, w, h))

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


full = im.resize((1000, round(H * 1000 / W)), Image.LANCZOS) if W != 1000 else im
save_png(full, out_full)

# Região escolhida, depois ajuste para 4:3
rx, ry, rw, rh = round(x * W), round(y * H), round(w * W), round(h * H)
region = im.crop((rx, ry, rx + rw, ry + rh))
cw, ch = region.size
target = 4 / 3
if cw / ch > target:
    nh = round(cw / target)
    padded = Image.new("RGB", (cw, nh), "white")
    padded.paste(region, (0, (nh - ch) // 2))
    region = padded
else:
    nh = round(cw / target)
    region = region.crop((0, (ch - nh) // 2, cw, (ch - nh) // 2 + nh))
save_png(region.resize((480, 360), Image.LANCZOS), out_sm)
print(f"{W}x{H} -> full {full.size[0]}x{full.size[1]}, sm 480x360 (regiao {rw}x{rh})")
