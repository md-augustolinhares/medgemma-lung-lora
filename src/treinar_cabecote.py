"""
Treina um cabecote de regressao linear simples (1152 -> 1) sobre os
embeddings visuais congelados do MedGemma, para prever a fracao pulmonar.

Uso:
    python treinar_cabecote.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ_PROJETO = Path(__file__).parent.parent
EMBEDDINGS = RAIZ_PROJETO / "dados" / "embeddings.npy"
ROTULOS = RAIZ_PROJETO / "dados" / "rotulos.npy"
SAIDA_MODELO = RAIZ_PROJETO / "checkpoints" / "cabecote.pt"
SAIDA_GRAFICO = RAIZ_PROJETO / "dados" / "previsto_vs_real.png"

FRACAO_VALIDACAO = 0.2
SEMENTE = 42
EPOCAS = 300
TAXA_APRENDIZADO = 1e-3


def main() -> None:
    embeddings = np.load(EMBEDDINGS)
    rotulos = np.load(ROTULOS)

    rng = np.random.default_rng(SEMENTE)
    indices = rng.permutation(len(embeddings))
    corte = int(len(indices) * (1 - FRACAO_VALIDACAO))
    idx_treino, idx_val = indices[:corte], indices[corte:]

    print(f"Treino: {len(idx_treino)} imagens | Validacao: {len(idx_val)} imagens")

    X_treino = torch.tensor(embeddings[idx_treino], dtype=torch.float32)
    y_treino = torch.tensor(rotulos[idx_treino], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(embeddings[idx_val], dtype=torch.float32)
    y_val = torch.tensor(rotulos[idx_val], dtype=torch.float32).unsqueeze(1)

    modelo = nn.Linear(1152, 1)
    perda_fn = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)

    for epoca in range(EPOCAS):
        modelo.train()
        otimizador.zero_grad()
        pred_treino = modelo(X_treino)
        perda_treino = perda_fn(pred_treino, y_treino)
        perda_treino.backward()
        otimizador.step()

        if (epoca + 1) % 50 == 0:
            modelo.eval()
            with torch.no_grad():
                perda_val = perda_fn(modelo(X_val), y_val)
            print(
                f"epoca {epoca + 1}/{EPOCAS} | perda treino: {perda_treino.item():.5f} "
                f"| perda validacao: {perda_val.item():.5f}"
            )

    modelo.eval()
    with torch.no_grad():
        pred_val = modelo(X_val)
        erro_absoluto_medio = (pred_val - y_val).abs().mean().item()

    print(f"\nErro absoluto medio na validacao: {erro_absoluto_medio:.4f}")
    print("(ou seja, em media, o chute do modelo erra por essa fracao da imagem)")

    torch.save(modelo.state_dict(), SAIDA_MODELO)
    print(f"Cabecote treinado salvo em: {SAIDA_MODELO}")

    plt.figure(figsize=(5, 5))
    plt.scatter(y_val.numpy(), pred_val.numpy(), alpha=0.6)
    limite = [0, max(y_val.max().item(), pred_val.max().item()) * 1.1]
    plt.plot(limite, limite, "r--", label="previsao perfeita")
    plt.xlabel("Fracao pulmonar real (da mascara)")
    plt.ylabel("Fracao pulmonar prevista (pelo modelo)")
    plt.title("Validacao: previsto vs real")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAIDA_GRAFICO)
    print(f"Grafico salvo em: {SAIDA_GRAFICO}")


if __name__ == "__main__":
    main()
