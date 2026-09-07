"""
Fine-tuning real (LoRA) do vision_tower do MedGemma, treinado junto com um
cabecote de regressao, para prever a fracao pulmao/torax.

Diferente do linear probe (treinar_cabecote.py), aqui o vision_tower deixa
de estar 100% congelado: pequenas matrizes LoRA dentro das camadas de
atencao (q_proj, k_proj, v_proj, out_proj) tambem sao treinadas, junto
com o cabecote. Por isso a imagem passa pelo vision_tower A CADA epoca
(nao dá mais pra usar embeddings pre-calculados e congelados).

Uso:
    python treinar_lora.py
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

MODEL_ID = "google/medgemma-4b-it"
RAIZ_PROJETO = Path(__file__).parent.parent
CSV_ROTULOS = RAIZ_PROJETO / "dados" / "rotulos_area_pulmonar.csv"
SAIDA_LORA = RAIZ_PROJETO / "checkpoints" / "lora_vision"
SAIDA_CABECOTE = RAIZ_PROJETO / "checkpoints" / "cabecote_lora.pt"

FRACAO_VALIDACAO = 0.2
SEMENTE = 42
EPOCAS = 8
TAMANHO_LOTE = 4  # quantas imagens acumular antes de cada passo do otimizador
TAXA_APRENDIZADO = 1e-4

# rank (posto) do LoRA: dimensao das matrizes pequenas A e B. Quanto maior,
# mais expressivo o ajuste, mas mais parametros treinaveis. 8 e um valor
# tipico de partida para tarefas simples.
LORA_RANK = 8
# alpha escala a contribuicao do ajuste LoRA (ajuste_efetivo = (alpha/rank) * B*A).
# alpha=2*rank e uma convencao comum de partida.
LORA_ALPHA = 16
# dropout dentro do caminho do LoRA, para reduzir overfitting - relevante
# aqui porque nosso dataset (704 imagens) e pequeno.
LORA_DROPOUT = 0.05

# O vision_tower tem 27 camadas de atencao. O gradient checkpointing (que
# reduziria a memoria de ativacao guardada para o backward) NAO e realmente
# implementado nesta versao do SiglipVisionModel apesar de declarar suporte
# - entao, em vez disso, aplicamos o LoRA so nas ULTIMAS camadas: assim,
# so as ativacoes dessas camadas precisam ficar guardadas para o backward,
# nao das 27. Isso tambem segue uma pratica padrao de transfer learning:
# camadas iniciais captam padroes visuais genericos (bordas, texturas),
# as ultimas captam conceitos mais especificos - geralmente sao as ultimas
# que vale a pena ajustar para uma tarefa nova.
N_CAMADAS_TOTAL = 27
N_ULTIMAS_CAMADAS_LORA = 6
MODULOS_ALVO = [
    f"encoder.layers.{i}.self_attn.{proj}"
    for i in range(N_CAMADAS_TOTAL - N_ULTIMAS_CAMADAS_LORA, N_CAMADAS_TOTAL)
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]
]


class CabecoteRegressao(nn.Module):
    def __init__(self, dim_entrada: int = 1152):
        super().__init__()
        self.linear = nn.Linear(dim_entrada, 1)

    def forward(self, x):
        return self.linear(x)


def carregar_vision_tower_lora():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processador = AutoProcessor.from_pretrained(MODEL_ID)
    modelo_base = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
    vision_tower = modelo_base.model.vision_tower

    # O language_model tem 2,28 bilhoes de parametros (10,8x o vision_tower,
    # que tem 211 milhoes) e nunca e usado aqui - so precisamos da parte de
    # visao. Carregar o modelo completo traz ele pra VRAM de qualquer jeito;
    # descartamos logo em seguida pra liberar ~1,25GB que, sem isso, deixava
    # o treino na borda do limite da placa (e o Windows, nesse limite, pode
    # cair para memoria compartilhada com a RAM - dezenas de vezes mais
    # lenta - o que explica o teste anterior ter ficado preso por horas).
    del modelo_base.model.language_model
    del modelo_base.lm_head
    torch.cuda.empty_cache()

    # Prepara o modelo 4-bit para treino: ajusta os hooks necessarios para
    # gradiente fluir corretamente atraves de camadas congeladas +
    # quantizadas. NAO habilitamos gradient checkpointing aqui - essa
    # versao do SiglipVisionModel declara suporte mas nao implementa de
    # verdade (confirmado inspecionando o codigo-fonte), entao ligar a
    # flag so daria falsa sensacao de seguranca. Em vez disso, controlamos
    # a memoria de ativacao aplicando o LoRA so nas ultimas camadas
    # (ver MODULOS_ALVO) - unico dos dois metodos que funcionou aqui.
    vision_tower = prepare_model_for_kbit_training(
        vision_tower, use_gradient_checkpointing=False
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=MODULOS_ALVO,
        bias="none",
    )
    vision_tower_lora = get_peft_model(vision_tower, lora_config)
    vision_tower_lora.print_trainable_parameters()

    return vision_tower_lora, processador


def embedding_de_imagem(vision_tower, processador, caminho_imagem: Path, device, dtype):
    imagem = Image.open(caminho_imagem).convert("RGB")
    entradas = processador.image_processor(images=imagem, return_tensors="pt")
    pixel_values = entradas["pixel_values"].to(device, dtype=dtype)

    saida = vision_tower(pixel_values=pixel_values)
    return saida.last_hidden_state.mean(dim=1).squeeze(0)  # (1152,)


def avaliar(vision_tower, cabecote, processador, linhas, device, dtype):
    vision_tower.eval()
    cabecote.eval()
    erros = []
    with torch.no_grad():
        for linha in linhas:
            caminho = RAIZ_PROJETO / linha["caminho_imagem"]
            embedding = embedding_de_imagem(vision_tower, processador, caminho, device, dtype)
            pred = cabecote(embedding.float().unsqueeze(0)).item()
            real = float(linha["fracao_pulmonar"])
            erros.append(abs(pred - real))
    return float(np.mean(erros))


def main() -> None:
    with open(CSV_ROTULOS, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    rng = np.random.default_rng(SEMENTE)
    indices = rng.permutation(len(linhas))
    corte = int(len(indices) * (1 - FRACAO_VALIDACAO))
    linhas_treino = [linhas[i] for i in indices[:corte]]
    linhas_val = [linhas[i] for i in indices[corte:]]

    print(f"Treino: {len(linhas_treino)} | Validacao: {len(linhas_val)}")

    vision_tower, processador = carregar_vision_tower_lora()
    device = next(vision_tower.parameters()).device
    dtype = next(vision_tower.parameters()).dtype

    cabecote = CabecoteRegressao().to(device=device, dtype=torch.float32)

    parametros_treinaveis = [p for p in vision_tower.parameters() if p.requires_grad]
    parametros_treinaveis += list(cabecote.parameters())
    otimizador = torch.optim.Adam(parametros_treinaveis, lr=TAXA_APRENDIZADO)
    perda_fn = nn.MSELoss()

    ordem = list(range(len(linhas_treino)))

    for epoca in range(EPOCAS):
        vision_tower.train()
        cabecote.train()
        random_epoca = np.random.default_rng(SEMENTE + epoca)
        random_epoca.shuffle(ordem)

        perda_acumulada = 0.0
        otimizador.zero_grad()
        inicio_epoca = time.time()

        for passo, idx in enumerate(ordem):
            linha = linhas_treino[idx]
            caminho = RAIZ_PROJETO / linha["caminho_imagem"]
            embedding = embedding_de_imagem(vision_tower, processador, caminho, device, dtype)
            pred = cabecote(embedding.float().unsqueeze(0))
            real = torch.tensor([[float(linha["fracao_pulmonar"])]], device=device)

            perda = perda_fn(pred, real) / TAMANHO_LOTE
            perda.backward()
            perda_acumulada += perda.item() * TAMANHO_LOTE

            if (passo + 1) % TAMANHO_LOTE == 0 or (passo + 1) == len(ordem):
                otimizador.step()
                otimizador.zero_grad()

            if (passo + 1) % 20 == 0:
                decorrido = time.time() - inicio_epoca
                seg_por_imagem = decorrido / (passo + 1)
                restante = seg_por_imagem * (len(ordem) - passo - 1)
                print(
                    f"  [{passo + 1}/{len(ordem)}] {seg_por_imagem:.2f}s/imagem "
                    f"| restante nesta epoca: ~{restante / 60:.1f} min",
                    flush=True,
                )

        perda_media_treino = perda_acumulada / len(ordem)
        mae_val = avaliar(vision_tower, cabecote, processador, linhas_val, device, dtype)
        print(f"epoca {epoca + 1}/{EPOCAS} | perda treino: {perda_media_treino:.5f} | MAE validacao: {mae_val:.4f}")

    SAIDA_LORA.mkdir(parents=True, exist_ok=True)
    vision_tower.save_pretrained(str(SAIDA_LORA))
    torch.save(cabecote.state_dict(), SAIDA_CABECOTE)
    print(f"\nAdapter LoRA salvo em: {SAIDA_LORA}")
    print(f"Cabecote salvo em: {SAIDA_CABECOTE}")


if __name__ == "__main__":
    main()
