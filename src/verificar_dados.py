"""
Gera uma grade de exemplos (imagem + mascara de torax (azul) + mascara de
pulmao (vermelho) sobrepostas + fracao calculada) para conferencia visual
dos rotulos gerados por preparar_dados.py.

Uso:
    python verificar_dados.py
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from preparar_dados import mascara_corpo

RAIZ_PROJETO = Path(__file__).parent.parent
CSV_ROTULOS = RAIZ_PROJETO / "dados" / "rotulos_area_pulmonar.csv"
SAIDA_PNG = RAIZ_PROJETO / "dados" / "conferencia_visual.png"

N_AMOSTRAS = 6
TAMANHO_CELULA = 280

# sempre incluir esse caso, que tinha dado problema com o limiar antigo (Otsu)
CASO_FORCADO = "MCUCXR_0055_0.png"


def sobrepor_mascaras(caminho_imagem: Path, caminho_mascara: Path) -> Image.Image:
    imagem_pil = Image.open(caminho_imagem).convert("L")
    imagem_cinza = np.array(imagem_pil)
    corpo = mascara_corpo(imagem_cinza)

    mascara_pulmao_pil = Image.open(caminho_mascara).convert("L")

    base = imagem_pil.convert("RGB").resize((TAMANHO_CELULA, TAMANHO_CELULA))

    corpo_pil = Image.fromarray((corpo * 255).astype(np.uint8)).resize((TAMANHO_CELULA, TAMANHO_CELULA))
    pulmao_pil = mascara_pulmao_pil.resize((TAMANHO_CELULA, TAMANHO_CELULA))

    overlay_corpo = Image.new("RGB", base.size, (0, 100, 255))
    alpha_corpo = corpo_pil.point(lambda p: 60 if p > 127 else 0)
    resultado = base.copy()
    resultado.paste(overlay_corpo, (0, 0), alpha_corpo)

    overlay_pulmao = Image.new("RGB", base.size, (255, 0, 0))
    alpha_pulmao = pulmao_pil.point(lambda p: 110 if p > 127 else 0)
    resultado.paste(overlay_pulmao, (0, 0), alpha_pulmao)

    return resultado


def main() -> None:
    with open(CSV_ROTULOS, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    random.seed(42)
    amostras = random.sample(linhas, N_AMOSTRAS - 1)
    caso_forcado = next((l for l in linhas if CASO_FORCADO in l["caminho_imagem"]), None)
    if caso_forcado and caso_forcado not in amostras:
        amostras = [caso_forcado] + amostras[:-1]

    grade = Image.new("RGB", (TAMANHO_CELULA * 3, TAMANHO_CELULA * 2 + 60), (20, 20, 20))
    fonte = ImageFont.load_default()

    for i, linha in enumerate(amostras):
        caminho_imagem = RAIZ_PROJETO / linha["caminho_imagem"]
        caminho_mascara = Path(str(caminho_imagem).replace("/img/", "/mask/").replace("\\img\\", "\\mask\\"))

        celula = sobrepor_mascaras(caminho_imagem, caminho_mascara)

        x = (i % 3) * TAMANHO_CELULA
        y = (i // 3) * (TAMANHO_CELULA + 30)
        grade.paste(celula, (x, y))

        desenho = ImageDraw.Draw(grade)
        texto = f"{linha['origem']} | pulmao/torax={linha['fracao_pulmonar']}"
        desenho.text((x + 5, y + TAMANHO_CELULA + 5), texto, fill=(255, 255, 255), font=fonte)

    grade.save(SAIDA_PNG)
    print(f"Salvo em: {SAIDA_PNG}")


if __name__ == "__main__":
    main()
