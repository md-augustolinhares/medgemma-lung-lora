"""
Interface web para prever a fracao de area pulmonar visivel numa radiografia
de torax, usando o cabecote treinado sobre embeddings congelados do MedGemma.

Uso:
    python app_area.py

Abre em http://127.0.0.1:7861
"""

import gradio as gr

from prever_area import carregar_tudo, prever

AVISO = (
    "⚠️ Ferramenta de estudo/pesquisa. O numero e uma medida RELATIVA "
    "(% da imagem ocupada pelo pulmao), nao uma area fisica em cm² — "
    "as imagens de treino nao tem calibracao de escala real."
)


def prever_upload(caminho_imagem: str | None) -> str:
    if caminho_imagem is None:
        return "Envie uma imagem de radiografia de torax primeiro."
    try:
        fracao = prever(caminho_imagem)
        return f"{fracao * 100:.1f}% da imagem ocupada pelos pulmoes"
    except Exception as erro:
        return f"Erro ao processar a imagem: {erro}"


with gr.Blocks(title="Area Pulmonar (MedGemma + cabecote linear)") as demo:
    gr.Markdown("# Previsao de area pulmonar (relativa)")
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
