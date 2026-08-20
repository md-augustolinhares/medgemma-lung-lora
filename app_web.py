"""
Interface web local para analisar radiografias de torax com o checkpoint
oficial do MedGemma (google/medgemma-4b-it), rodando via transformers.

Uso:
    python app_web.py

Abre em http://127.0.0.1:7860
"""

import gradio as gr

from analisar_rx import analisar, carregar_modelo

AVISO = (
    "⚠️ Ferramenta de estudo/pesquisa. Nao e um dispositivo medico e nao "
    "substitui avaliacao de um radiologista ou medico responsavel."
)


def analisar_upload(caminho_imagem: str | None) -> str:
    if caminho_imagem is None:
        return "Envie uma imagem de radiografia de torax primeiro."
    try:
        return analisar(caminho_imagem)
    except Exception as erro:  # mostra o erro na propria interface
        return f"Erro ao analisar a imagem: {erro}"


with gr.Blocks(title="Analise de RX de Torax (MedGemma oficial)") as demo:
    gr.Markdown("# Analise de radiografia de torax")
    gr.Markdown(AVISO)

    with gr.Row():
        entrada = gr.Image(type="filepath", label="Radiografia de torax")
        saida = gr.Textbox(label="Laudo gerado", lines=20)

    botao = gr.Button("Analisar", variant="primary")
    botao.click(fn=analisar_upload, inputs=entrada, outputs=saida)

if __name__ == "__main__":
    print("Carregando o modelo (pode levar um tempo na primeira vez)...")
    carregar_modelo()
    demo.launch(server_name="127.0.0.1", server_port=7860)
