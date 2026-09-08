"""
Interface web para prever a fracao pulmao/torax numa radiografia de torax,
usando o vision_tower do MedGemma com o adapter LoRA treinado.

Uso:
    python app_area.py

Abre em http://127.0.0.1:7861
"""

import gradio as gr

from prever_area_lora import carregar_tudo, prever

AVISO = (
    "Ferramenta de estudo/pesquisa. O numero e a fracao do TORAX do "
    "paciente ocupada pelos pulmoes, uma medida relativa (nao e area "
    "fisica em cm2: as imagens de treino nao tem calibracao de escala real)."
)


def prever_upload(caminho_imagem: str | None) -> str:
    if caminho_imagem is None:
        return "Envie uma imagem de radiografia de torax primeiro."
    try:
        fracao = prever(caminho_imagem)
        return f"{fracao * 100:.1f}% do torax ocupado pelos pulmoes"
    except Exception as erro:
        return f"Erro ao processar a imagem: {erro}"


with gr.Blocks(title="Area Pulmonar (MedGemma + LoRA)") as demo:
    gr.Markdown("# Previsao de area pulmonar (pulmao / torax)")
    gr.Markdown(AVISO)

    with gr.Row():
        entrada = gr.Image(type="filepath", label="Radiografia de torax")
        saida = gr.Textbox(label="Previsao")

    botao = gr.Button("Calcular", variant="primary")
    botao.click(fn=prever_upload, inputs=entrada, outputs=saida)

if __name__ == "__main__":
    print("Carregando o modelo (pode levar um tempo na primeira vez)...")
    carregar_tudo()
    demo.launch(server_name="127.0.0.1", server_port=7861)
