"""
Varre os datasets Montgomery e Shenzhen (imagem + mascara binaria de pulmao)
e gera uma tabela (CSV) com o rotulo de treino: a fracao da imagem ocupada
pelos pulmoes, calculada a partir da mascara.

Uso:
    python preparar_dados.py
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ_DADOS = Path(__file__).parent.parent / "dados" / "lung-segmentation"
SAIDA_CSV = Path(__file__).parent.parent / "dados" / "rotulos_area_pulmonar.csv"

DATASETS = {
    "Montgomery": RAIZ_DADOS / "Montgomery" / "Montgomery",
    "Shenzhen": RAIZ_DADOS / "Shenzhen" / "Shenzhen",
}

LIMIAR_BINARIZACAO = 127


def fracao_pulmonar(caminho_mascara: Path) -> float:
    mascara = np.array(Image.open(caminho_mascara).convert("L"))
    binaria = mascara > LIMIAR_BINARIZACAO
    return float(binaria.sum()) / binaria.size


def main() -> None:
    linhas = []

    for origem, pasta in DATASETS.items():
        pasta_img = pasta / "img"
        pasta_mask = pasta / "mask"

        for caminho_img in sorted(pasta_img.glob("*.png")):
            caminho_mask = pasta_mask / caminho_img.name
            if not caminho_mask.exists():
                print(f"Aviso: sem mascara para {caminho_img.name}, pulando.")
                continue

            fracao = fracao_pulmonar(caminho_mask)
            linhas.append(
                {
                    "origem": origem,
                    "caminho_imagem": str(caminho_img.relative_to(RAIZ_DADOS.parent.parent)),
                    "fracao_pulmonar": round(fracao, 4),
                }
            )

    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["origem", "caminho_imagem", "fracao_pulmonar"])
        escritor.writeheader()
        escritor.writerows(linhas)

    fracoes = np.array([linha["fracao_pulmonar"] for linha in linhas])
    print(f"\n{len(linhas)} pares imagem+mascara processados.")
    print(f"Salvo em: {SAIDA_CSV}")
    print(f"\nEstatisticas da fracao pulmonar (rotulo de treino):")
    print(f"  minimo:  {fracoes.min():.4f}")
    print(f"  maximo:  {fracoes.max():.4f}")
    print(f"  media:   {fracoes.mean():.4f}")
    print(f"  desvio:  {fracoes.std():.4f}")


if __name__ == "__main__":
    main()
