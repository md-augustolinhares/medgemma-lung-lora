"""
Analisa uma radiografia de torax usando o checkpoint oficial do MedGemma
(google/medgemma-4b-it) direto do Hugging Face, em 4-bit via bitsandbytes.

Uso:
    python analisar_rx.py caminho/para/imagem.png
"""

import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "google/medgemma-4b-it"

PROMPT_SISTEMA = (
    "Voce e um assistente de apoio educacional para leitura de radiografias de "
    "torax. Avalie CADA item da checklist abaixo individualmente, na ordem, "
    "antes de qualquer conclusao. Para cada item responda: Presente / Ausente / "
    "Incerto, com uma justificativa curta baseada no que e visualmente "
    "observavel na imagem. Nao conclua 'normal' em um item sem justificar "
    "explicitamente o que foi observado nele — 'nao avaliado' nunca deve virar "
    "'ausente' por omissao.\n\n"
    "Checklist obrigatoria:\n"
    "1. Qualidade tecnica (posicionamento, penetracao, rotacao, inspiracao)\n"
    "2. Traqueia e mediastino\n"
    "3. Silhueta cardiaca / indice cardiotoracico\n"
    "4. Hilos pulmonares\n"
    "5. Campos pulmonares - atelectasia\n"
    "6. Campos pulmonares - consolidacao / infiltrado\n"
    "7. Campos pulmonares - massa / nodulo\n"
    "8. Campos pulmonares - enfisema / hiperinsuflacao\n"
    "9. Campos pulmonares - fibrose / padrao intersticial\n"
    "10. Pneumotorax\n"
    "11. Seios costofrenicos / derrame pleural (avalie cada lado separadamente)\n"
    "12. Edema pulmonar / congestao vascular\n"
    "13. Ossos (fraturas, lesoes liticas ou blasticas)\n"
    "14. Partes moles, corpos estranhos e dispositivos (tubos, cateteres, "
    "marca-passo)\n\n"
    "Depois da checklist completa, escreva:\n"
    "- Resumo dos achados positivos (ou 'sem achados positivos' se todos os "
    "itens relevantes vieram Ausente apos avaliacao explicita)\n"
    "- Hipoteses diagnosticas quando houver achado relevante\n"
    "- Nivel de confianca geral (baixo / medio / alto) e um lembrete de "
    "correlacao clinica\n\n"
    "Este resultado e apenas para fins de estudo/pesquisa: nao e um dispositivo "
    "medico, nao substitui avaliacao de um radiologista ou medico responsavel, e "
    "nao deve ser usado para decisao clinica real."
)

_modelo = None
_processador = None


def carregar_modelo():
    global _modelo, _processador
    if _modelo is not None:
        return _modelo, _processador

    print(f"Carregando {MODEL_ID} em 4-bit (isso baixa ~8GB na primeira vez)...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    _processador = AutoProcessor.from_pretrained(MODEL_ID)
    _modelo = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    return _modelo, _processador


def analisar(caminho_imagem: str) -> str:
    modelo, processador = carregar_modelo()
    imagem = Image.open(caminho_imagem).convert("RGB")

    mensagens = [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SISTEMA}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analise esta radiografia de torax."},
                {"type": "image", "image": imagem},
            ],
        },
    ]

    entradas = processador.apply_chat_template(
        mensagens,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(modelo.device)

    tamanho_entrada = entradas["input_ids"].shape[-1]

    with torch.inference_mode():
        saida = modelo.generate(**entradas, max_new_tokens=1500, do_sample=False)
        saida = saida[0][tamanho_entrada:]

    return processador.decode(saida, skip_special_tokens=True)


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python analisar_rx.py caminho/para/imagem.png")
        sys.exit(1)

    caminho_imagem = sys.argv[1]
    laudo = analisar(caminho_imagem)
    print("\n" + laudo)


if __name__ == "__main__":
    main()
