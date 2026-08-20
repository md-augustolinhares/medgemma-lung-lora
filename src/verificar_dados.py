"""
Gera uma grade de exemplos (imagem + mascara sobreposta + fracao calculada)
para conferencia visual dos rotulos gerados por preparar_dados.py.

Uso:
    python verificar_dados.py
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ_PROJETO = Path(__file__).parent.parent
CSV_ROTULOS = RAIZ_PROJETO / "dados" / "rotulos_area_pulmonar.csv"
SAIDA_PNG = RAIZ_PROJETO / "dados" / "conferencia_visual.png"

N_AMOSTRAS = 6
TAMANHO_CELULA = 280


def sobrepor_mascara(caminho_imagem: Path, caminho_mascara: Path) -> Image.Image:
    imagem = Image.open(caminho_imagem).convert("RGB").resize((TAMANHO_CELULA, TAMANHO_CELULA))
    mascara = Image.open(caminho_mascara).convert("L").resize((TAMANHO_CELULA, TAMANHO_CELULA))

    overlay = Image.new("RGB", imagem.size, (255, 0, 0))
    mascara_alpha = mascara.point(lambda p: 90 if p > 127 else 0)

    resultado = imagem.copy()
    resultado.paste(overlay, (0, 0), mascara_alpha)
    return resultado


def main() -> None:
    with open(CSV_ROTULOS, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    random.seed(42)
    amostras = random.sample(linhas, N_AMOSTRAS)

    grade = Image.new("RGB", (TAMANHO_CELULA * 3, TAMANHO_CELULA * 2 + 60), (20, 20, 20))
    fonte = ImageFont.load_default()

    for i, linha in enumerate(amostras):
        caminho_imagem = RAIZ_PROJETO / linha["caminho_imagem"]
        caminho_mascara = Path(str(caminho_imagem).replace("/img/", "/mask/").replace("\\img\\", "\\mask\\"))

        celula = sobrepor_mascara(caminho_imagem, caminho_mascara)

        x = (i % 3) * TAMANHO_CELULA
        y = (i // 3) * (TAMANHO_CELULA + 30)
        grade.paste(celula, (x, y))

        desenho = ImageDraw.Draw(grade)
        texto = f"{linha['origem']} | fracao={linha['fracao_pulmonar']}"
        desenho.text((x + 5, y + TAMANHO_CELULA + 5), texto, fill=(255, 255, 255), font=fonte)

    grade.save(SAIDA_PNG)
    print(f"Salvo em: {SAIDA_PNG}")


if __name__ == "__main__":
    main()
