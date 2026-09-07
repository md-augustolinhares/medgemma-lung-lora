"""
Preve a fracao pulmao/torax usando o vision_tower com o adapter LoRA
treinado (treinar_lora.py), em vez do cabecote linear puro sobre features
congeladas (prever_area.py).

Uso:
    python prever_area_lora.py caminho/para/imagem.png [outra_imagem.png ...]
"""

import sys
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from treinar_lora import CabecoteRegressao

MODEL_ID = "google/medgemma-4b-it"
RAIZ_PROJETO = Path(__file__).parent.parent
CAMINHO_LORA = RAIZ_PROJETO / "checkpoints" / "lora_vision"
CAMINHO_CABECOTE = RAIZ_PROJETO / "checkpoints" / "cabecote_lora.pt"

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
    modelo_base = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
    vision_tower_base = modelo_base.model.vision_tower
    del modelo_base.model.language_model
    del modelo_base.lm_head
    torch.cuda.empty_cache()

    _vision_tower = PeftModel.from_pretrained(vision_tower_base, str(CAMINHO_LORA))
    _vision_tower.eval()

    _cabecote = CabecoteRegressao()
    _cabecote.load_state_dict(torch.load(CAMINHO_CABECOTE, map_location="cpu"))
    _cabecote.eval()


def prever(caminho_imagem: str) -> float:
    carregar_tudo()

    imagem = Image.open(caminho_imagem).convert("RGB")
    entradas = _processador.image_processor(images=imagem, return_tensors="pt")
    device = next(_vision_tower.parameters()).device
    dtype = next(_vision_tower.parameters()).dtype
    pixel_values = entradas["pixel_values"].to(device, dtype=dtype)

    with torch.no_grad():
        saida = _vision_tower(pixel_values=pixel_values)
        embedding = saida.last_hidden_state.mean(dim=1).squeeze(0).float().cpu()
        fracao = _cabecote(embedding.unsqueeze(0)).item()

    return fracao


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python prever_area_lora.py imagem1.png [imagem2.png ...]")
        sys.exit(1)

    for caminho in sys.argv[1:]:
        fracao = prever(caminho)
        print(f"{caminho}: {fracao * 100:.1f}% (pulmao/torax)")


if __name__ == "__main__":
    main()
