"""
Extrai o embedding visual (congelado) do MedGemma para cada imagem do dataset
de segmentacao pulmonar, e salva junto com o rotulo (fracao pulmonar).

Isso roda o encoder de visao UMA UNICA VEZ por imagem — o resultado nunca
muda entre epocas de treino, entao nao ha motivo pra recalcular.

Uso:
    python extrair_embeddings.py
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

MODEL_ID = "google/medgemma-4b-it"

RAIZ_PROJETO = Path(__file__).parent.parent
CSV_ROTULOS = RAIZ_PROJETO / "dados" / "rotulos_area_pulmonar.csv"
SAIDA_EMBEDDINGS = RAIZ_PROJETO / "dados" / "embeddings.npy"
SAIDA_ROTULOS = RAIZ_PROJETO / "dados" / "rotulos.npy"


def carregar_vision_tower():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processador = AutoProcessor.from_pretrained(MODEL_ID)
    modelo = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
    return modelo.model.vision_tower, processador


def embedding_de_uma_imagem(vision_tower, processador, caminho_imagem: Path) -> np.ndarray:
    imagem = Image.open(caminho_imagem).convert("RGB")
    entradas = processador.image_processor(images=imagem, return_tensors="pt")
    pixel_values = entradas["pixel_values"].to(vision_tower.device, dtype=vision_tower.dtype)

    with torch.no_grad():
        saida = vision_tower(pixel_values=pixel_values)
        # saida.last_hidden_state: (1, num_patches, 1152) -> media sobre os patches
        embedding = saida.last_hidden_state.mean(dim=1).squeeze(0)

    return embedding.float().cpu().numpy()


def main() -> None:
    with open(CSV_ROTULOS, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    print(f"Carregando vision_tower do {MODEL_ID}...")
    vision_tower, processador = carregar_vision_tower()

    embeddings = []
    rotulos = []

    for i, linha in enumerate(linhas):
        caminho_imagem = RAIZ_PROJETO / linha["caminho_imagem"]
        embedding = embedding_de_uma_imagem(vision_tower, processador, caminho_imagem)
        embeddings.append(embedding)
        rotulos.append(float(linha["fracao_pulmonar"]))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(linhas)} imagens processadas")

    embeddings = np.stack(embeddings)
    rotulos = np.array(rotulos, dtype=np.float32)

    np.save(SAIDA_EMBEDDINGS, embeddings)
    np.save(SAIDA_ROTULOS, rotulos)

    print(f"\nEmbeddings salvos: {SAIDA_EMBEDDINGS} (shape={embeddings.shape})")
    print(f"Rotulos salvos: {SAIDA_ROTULOS} (shape={rotulos.shape})")


if __name__ == "__main__":
    main()
