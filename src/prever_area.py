"""
Recebe uma radiografia de torax e preve a fracao da imagem ocupada pelos
pulmoes, usando o vision_tower congelado do MedGemma + o cabecote linear
treinado em treinar_cabecote.py.

Uso:
    python prever_area.py caminho/para/imagem.png
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

MODEL_ID = "google/medgemma-4b-it"
RAIZ_PROJETO = Path(__file__).parent.parent
PESOS_CABECOTE = RAIZ_PROJETO / "checkpoints" / "cabecote.pt"

_vision_tower = None
_processador = None
_cabecote = None


def carregar_tudo():
    global _vision_tower, _processador, _cabecote
    if _vision_tower is not None:
        return

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    _processador = AutoProcessor.from_pretrained(MODEL_ID)
    modelo = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
    _vision_tower = modelo.model.vision_tower

    _cabecote = nn.Linear(1152, 1)
    _cabecote.load_state_dict(torch.load(PESOS_CABECOTE, map_location="cpu"))
    _cabecote.eval()


def prever(caminho_imagem: str) -> float:
    """Retorna a fracao prevista (0.0 a 1.0) da imagem ocupada pelos pulmoes."""
    carregar_tudo()

    imagem = Image.open(caminho_imagem).convert("RGB")
    entradas = _processador.image_processor(images=imagem, return_tensors="pt")
    pixel_values = entradas["pixel_values"].to(_vision_tower.device, dtype=_vision_tower.dtype)

    with torch.no_grad():
        saida = _vision_tower(pixel_values=pixel_values)
        embedding = saida.last_hidden_state.mean(dim=1).squeeze(0).float().cpu()
        fracao = _cabecote(embedding).item()

    return fracao


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python prever_area.py caminho/para/imagem.png")
        sys.exit(1)

    fracao = prever(sys.argv[1])
    print(f"\nArea pulmonar prevista: {fracao * 100:.1f}% da imagem")
    print("(medida relativa - nao e uma area fisica em cm2, ver explicacao no chat)")


if __name__ == "__main__":
    main()
