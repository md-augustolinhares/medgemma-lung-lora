"""
Varre os datasets Montgomery e Shenzhen (imagem + mascara binaria de pulmao)
e gera uma tabela (CSV) com o rotulo de treino: a fracao do TORAX (silhueta
do corpo do paciente) ocupada pelos pulmoes — nao a fracao da imagem inteira.

Isso torna o rotulo invariante ao enquadramento/zoom da imagem: cortar mais
ou menos fundo preto ao redor do corpo nao muda a razao pulmao/torax.

Uso:
    python preparar_dados.py
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

RAIZ_DADOS = Path(__file__).parent.parent / "dados" / "lung-segmentation"
SAIDA_CSV = Path(__file__).parent.parent / "dados" / "rotulos_area_pulmonar.csv"

DATASETS = {
    "Montgomery": RAIZ_DADOS / "Montgomery" / "Montgomery",
    "Shenzhen": RAIZ_DADOS / "Shenzhen" / "Shenzhen",
}

LIMIAR_BINARIZACAO_MASCARA = 127

# Limiar fixo e baixo, nao Otsu: o fundo (fora do paciente) e ~0 e ate o
# pulmao (a parte mais escura DENTRO do corpo) fica em ~80+. Um limiar
# "esperto" (Otsu) pode acabar separando pulmao-escuro de osso-claro em vez
# de fundo de corpo, especialmente em imagens com pouca borda preta -
# fragmentando o torax em varios pedacos desconectados pelo "rio" do pulmao.
LIMIAR_CORPO = 10


def mascara_corpo(imagem_cinza: np.ndarray) -> np.ndarray:
    """Segmenta a silhueta do corpo (torax) via limiar de intensidade fixo."""
    binaria = imagem_cinza > LIMIAR_CORPO

    # preenche buracos (ex: campos pulmonares escuros "vazando" pro fundo)
    binaria = ndimage.binary_fill_holes(binaria)

    # mantem so o maior componente conectado (remove texto/marcadores soltos)
    rotulado, n_componentes = ndimage.label(binaria)
    if n_componentes > 1:
        tamanhos = ndimage.sum(binaria, rotulado, range(1, n_componentes + 1))
        maior = np.argmax(tamanhos) + 1
        binaria = rotulado == maior

    return binaria


def fracao_pulmonar_no_torax(caminho_imagem: Path, caminho_mascara: Path) -> float:
    imagem = np.array(Image.open(caminho_imagem).convert("L"))
    mascara_pulmao = np.array(Image.open(caminho_mascara).convert("L")) > LIMIAR_BINARIZACAO_MASCARA

    corpo = mascara_corpo(imagem)

    area_torax = corpo.sum()
    area_pulmao = (mascara_pulmao & corpo).sum()  # garante que so conta pulmao dentro do corpo

    return float(area_pulmao) / float(area_torax)


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

            fracao = fracao_pulmonar_no_torax(caminho_img, caminho_mask)
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
    print(f"\nEstatisticas da fracao pulmao/torax (rotulo de treino):")
    print(f"  minimo:  {fracoes.min():.4f}")
    print(f"  maximo:  {fracoes.max():.4f}")
    print(f"  media:   {fracoes.mean():.4f}")
    print(f"  desvio:  {fracoes.std():.4f}")


if __name__ == "__main__":
    main()
