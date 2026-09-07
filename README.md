# MedGemma RX — Análise de Radiografia de Tórax com IA

Projeto de estudo prático sobre fine-tuning de modelos de IA médica, usando o
[MedGemma](https://huggingface.co/google/medgemma-4b-it) do Google (variante
médica do Gemma 3) para analisar radiografias de tórax — geração de laudo por
checklist e, principalmente, um experimento completo de **fine-tuning com
LoRA** para prever a proporção pulmão/tórax visível na imagem.

Este projeto foi feito com o objetivo de **aprender o processo** de
fine-tuning de modelos de IA aplicados à medicina — não é uma ferramenta
clínica pronta, e os resultados devem ser lidos como isso: um exercício de
aprendizado, documentado com os erros e correções reais que aconteceram no
caminho.

## O que tem aqui

Duas frentes:

1. **Laudo por checklist** (`src/analisar_rx.py` + `src/app_web.py`) — usa o
   MedGemma direto, em modo texto, para gerar uma leitura estruturada da
   radiografia (14 itens: qualidade técnica, mediastino, campos pulmonares,
   seios costofrênicos, etc.)
2. **Previsão de área pulmonar** (o núcleo do projeto) — um pipeline completo
   de fine-tuning que ensina o modelo a prever **que fração do tórax do
   paciente é ocupada pelos pulmões**, a partir só da imagem.

## Como funciona a previsão de área pulmonar

### 1. Dado de treino

Usamos o dataset público [Montgomery + Shenzhen Chest X-ray Segmentation](https://data.mendeley.com/datasets/8gf9vpkhgy/2)
(CC BY 4.0, 704 imagens com máscara de pulmão já anotada por especialistas).
Como o dataset só tem máscara de *pulmão*, calculamos a máscara do *tórax*
(silhueta do corpo do paciente) por conta própria, via limiarização de
intensidade — o ar fora do paciente não atenua o raio-X e fica bem mais
escuro que qualquer tecido do corpo, então dá pra separar corpo de fundo só
pelo brilho do pixel.

O rótulo de treino é: `área do pulmão ÷ área do tórax` — não
`área do pulmão ÷ imagem inteira`. Essa escolha foi deliberada (ver seção
de decisões abaixo): torna a métrica invariante ao quanto a foto foi
enquadrada/cortada ao redor do corpo.

### 2. Extração de embeddings + linear probe

Primeiro experimento: usamos o `vision_tower` (encoder de imagem) do
MedGemma **congelado** — sem treinar nada nele — e treinamos só uma camada
linear simples em cima da média dos embeddings de imagem, prevendo a fração
pulmão/tórax. Resultado: erro médio de ~2,4 pontos percentuais na validação.
Bom, mas com limitações reais em imagens fora do dataset de treino (ver
seção de resultados).

### 3. Fine-tuning com LoRA

Para melhorar a precisão em casos fora da distribuição de treino, aplicamos
**LoRA** (Low-Rank Adaptation) nas últimas 6 camadas de atenção do
`vision_tower` (de 27 no total), treinado junto com o cabeçote de
regressão — dessa vez de ponta a ponta, não mais sobre embeddings
congelados. Resultado final: **1,98% de erro médio na validação**, e melhora
mensurável de generalização em imagens externas (detalhes abaixo).

## Resultados

### Curva de treino do LoRA (8 épocas)

| Época | Perda treino | MAE validação |
|---|---|---|
| 1 | 0,18295 | 4,59% |
| 2 | 0,00296 | 3,36% |
| 4 | 0,00148 | 2,62% |
| 6 | 0,00105 | 2,19% |
| 8 | 0,00083 | **1,98%** |

### Teste em 4 imagens reais de fora do dataset de treino

Antes e depois do fine-tuning com LoRA, nas mesmas 4 imagens (nunca vistas
em treino nem validação):

| Imagem | Cabeçote linear (congelado) | Com LoRA |
|---|---|---|
| RX em expiração | 27,3% | 24,2% |
| RX em inspiração (mesmo paciente) | 27,5% | 26,1% |
| RX pós-cirurgia cardíaca | 39,4% (outlier) | 30,6% |
| RX em decúbito ventral | 27,7% | 29,5% |

Dois achados importantes desse teste:

- **Sensibilidade a inspiração/expiração**: o par inspiração/expiração do
  mesmo paciente deveria mostrar mais área pulmonar na inspiração (pulmão
  mais cheio de ar). O cabeçote linear captava isso muito fracamente
  (diferença de só 0,2 pontos); com LoRA, a diferença subiu para 1,9 pontos,
  na direção clinicamente correta.
- **Generalização em caso atípico**: a radiografia pós-cirúrgica (com fios
  de esternotomia, achado que não aparece no dataset de treino) tinha um
  valor bem fora da curva com o cabeçote linear; com LoRA, ficou muito mais
  próximo dos demais casos.

## Decisões de design (e os erros no caminho)

Esse projeto teve vários obstáculos reais, documentados aqui de propósito —
são a parte mais instrutiva do processo:

- **Por que pulmão/tórax, não pulmão/imagem**: a primeira versão calculava a
  fração da *imagem inteira* ocupada pelo pulmão. Testando com uma imagem
  mais cortada (menos fundo preto ao redor do corpo), o valor disparou pro
  teto do que o modelo via no treino — o enquadramento da foto, não a
  fisiologia, estava dominando o número. Redefinir o rótulo como
  pulmão/tórax resolveu isso.
- **O bug do limiar de Otsu**: a primeira tentativa de segmentar o tórax
  usava limiarização automática (Otsu). Em algumas imagens, o algoritmo
  encontrou o limiar entre "pulmão escuro" e "osso claro" em vez de entre
  "fundo" e "corpo" — fragmentando o tórax em pedaços desconectados e dando
  um rótulo quase zero. Trocado por um limiar fixo e baixo, já que o fundo
  real é ~0 e até o pulmão (a parte mais escura do corpo) já fica bem acima
  disso.
- **VRAM e o "travamento" de 3 horas**: a primeira tentativa de treino com
  LoRA ficou presa por horas sem terminar. Causa raiz, em duas partes: (1)
  o pipeline carregava o modelo MedGemma **completo**, incluindo o
  language_model (2,28 bilhões de parâmetros — 10,8x o tamanho do
  vision_tower), mesmo sem usá-lo; (2) isso deixava a VRAM da GPU no limite,
  e o Windows, nesse ponto, passa a usar memória compartilhada com a RAM
  (dezenas de vezes mais lenta) sem avisar. Corrigido descartando o
  language_model logo após carregar, e aplicando o LoRA só nas últimas
  camadas do encoder (o `gradient_checkpointing`, testado como alternativa,
  não é realmente implementado nessa versão do encoder do MedGemma apesar
  de declarar suporte).

## Limitações importantes

- **Não é dispositivo médico.** O MedGemma é distribuído pelo Google como
  ferramenta de pesquisa/educação, sem validação clínica regulatória. Nada
  aqui deve embasar decisão clínica real.
- **A métrica é relativa, não uma medida física.** "% pulmão/tórax" não é
  volume em litros nem área em cm² — as imagens de treino não têm
  calibração de escala física (distância de aquisição), então não dá pra
  converter pixel em medida real. Uma radiografia é uma projeção 2D; volume
  pulmonar é uma grandeza 3D — no melhor caso, esse número é um proxy visual
  aproximado, não uma medida de função pulmonar.
- **Dataset de treino pequeno e específico**: 704 imagens de dois programas
  de triagem de tuberculose (EUA e China) — não é representativo de
  qualquer serviço de radiologia, aparelho ou população.
- **Viés herdado do gabarito**: a máscara de pulmão original já sub-marca o
  seio costofrênico (região de baixo contraste, difícil de anotar) — o
  modelo herda esse viés.

## Estrutura do projeto

```
Medgemma-RX/
├── requirements.txt
├── checkpoints/
│   ├── cabecote.pt              cabecote linear (metrica antiga, pulmao/imagem)
│   ├── cabecote_lora.pt         cabecote treinado junto com o LoRA
│   └── lora_vision/             adapter LoRA (vision_tower)
└── src/
    ├── analisar_rx.py           laudo por checklist (MedGemma em modo texto)
    ├── app_web.py                interface web do laudo (Gradio)
    ├── preparar_dados.py         calcula o rotulo pulmao/torax a partir das mascaras
    ├── verificar_dados.py        grade visual de conferencia dos rotulos
    ├── extrair_embeddings.py     extrai embeddings congelados do vision_tower
    ├── treinar_cabecote.py       treina o cabecote linear (linear probe)
    ├── treinar_lora.py           fine-tuning real: LoRA + cabecote, ponta a ponta
    ├── prever_area.py            inferencia com o cabecote linear puro
    ├── prever_area_lora.py       inferencia com o adapter LoRA
    └── app_area.py                interface web da previsao de area (Gradio)
```

`dados/` (imagens, dataset baixado, embeddings) fica só local — nunca sobe
pro repositório (é dado potencialmente sensível e/ou grande demais pro git).

## Como rodar

Requer Python 3.14, GPU NVIDIA com CUDA, e uma conta no Hugging Face com
acesso liberado ao [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
(modelo com licença restrita — precisa aceitar os termos na página antes de
baixar).

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\hf.exe auth login   # cole seu token do Hugging Face

# laudo por checklist
.venv\Scripts\python.exe src\app_web.py

# previsao de area pulmonar (precisa preparar o dado antes, ver preparar_dados.py)
.venv\Scripts\python.exe src\app_area.py
```

## Créditos e licenças

- **MedGemma**: Google, [Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/medgemma). Uso sujeito aos termos do Google.
- **Dataset de segmentação pulmonar**: Danilov, V., Proutski, A., Kirpich, A., Litmanovich, D., Gankin, Y. *Chest X-ray dataset for lung segmentation*. Mendeley Data, V2, 2022. [doi: 10.17632/8gf9vpkhgy.2](https://doi.org/10.17632/8gf9vpkhgy.2) (CC BY 4.0). Combina os datasets Montgomery e Shenzhen, originalmente publicados pela U.S. National Library of Medicine (Jaeger et al., 2014).
