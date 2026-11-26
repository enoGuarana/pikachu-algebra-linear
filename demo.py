"""Demo interativa do seminário.  Rodar com:  streamlit run demo.py"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src"))

from comum import LADO, ajustar_svd, carregar, formato, vetorizar  # noqa: E402
from etapa4_cnn import CNN, acuracia_cnn, tensor, transformar, treinar_cnn  # noqa: E402

PESOS = RAIZ / "modelos" / "cnn.pt"


@st.cache_resource(show_spinner="Preparando os modelos (a primeira vez leva alguns minutos)...")
def preparar():
    dados = carregar()
    (tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))

    X = vetorizar(tr)
    media, _, Vt = ajustar_svd(X)
    linear = LogisticRegression(C=0.01, max_iter=5000).fit((X - media) @ Vt[:50].T, y_tr)
    acc_linear = linear.score((vetorizar(te) - media) @ Vt[:50].T, y_te)

    if PESOS.exists():
        cnn = CNN()
        cnn.load_state_dict(torch.load(PESOS, weights_only=True))
    else:
        cnn = treinar_cnn(tr, y_tr, va, y_va, com_aumento=True, semente=0)
        PESOS.parent.mkdir(exist_ok=True)
        torch.save(cnn.state_dict(), PESOS)
    cnn.eval()
    return media, Vt, linear, cnn, te, y_te, acc_linear, acuracia_cnn(cnn, te, y_te)


def probabilidade_cnn(cnn, img):
    with torch.no_grad():
        return float(cnn(tensor(img[None].copy())).softmax(1)[0, 1])


st.set_page_config(page_title="Isso é um Pikachu?", layout="wide")
st.title("Isso é um Pikachu?")
st.caption("Álgebra Linear por trás do reconhecimento de imagens")

media, Vt, linear, cnn, teste, y_teste, acc_linear, acc_cnn = preparar()

with st.sidebar:
    st.header("1. Escolha a imagem")
    arquivo = st.file_uploader("Envie uma foto", type=["jpg", "jpeg", "png", "webp"])
    indice = st.number_input("ou use uma imagem de teste", 0, len(teste) - 1, int(y_teste.argmax()),
                             disabled=arquivo is not None)
    st.header("2. Quantos componentes?")
    k = st.slider("k (reconstrução)", 1, len(Vt), 50)
    st.header("3. Teste de estresse")
    dx = st.slider("deslocar na horizontal (pixels)", -16, 16, 0)
    dy = st.slider("deslocar na vertical (pixels)", -16, 16, 0)
    giro = st.slider("girar (graus)", -45, 45, 0)
    zoom = st.slider("zoom", 1.0, 1.6, 1.0, 0.05)

if arquivo is not None:
    foto = ImageOps.exif_transpose(Image.open(arquivo)).convert("RGB")
    original = np.asarray(foto.resize((LADO, LADO), Image.LANCZOS))
else:
    original = teste[indice]

mexeu = dx or dy or giro or zoom != 1.0
img = transformar(original, dx=dx, dy=dy, giro=giro, zoom=zoom) if mexeu else original

x = vetorizar(img[None])[0]
pesos = Vt @ (x - media)
reconstrucao = np.clip(formato(media + pesos[:k] @ Vt[:k]), 0, 1)

st.subheader("A foto como soma de moldes")
a, b, c = st.columns(3)
a.image(original, caption="original 64 x 64", width=256)
b.image(img, caption="entrada dos modelos", width=256)
c.image(reconstrucao, caption=f"reconstruída com k = {k}", width=256)
st.write(f"De **{x.size:,}** números para **{k}**.".replace(",", "."))

st.subheader("O que cada modelo responde")
p_linear = float(linear.predict_proba(pesos[:50][None])[0, 1])
p_cnn = probabilidade_cnn(cnn, img)
for coluna, nome, p, acc in zip(st.columns(2), ["PCA (50) + Regressão Logística", "CNN"],
                                [p_linear, p_cnn], [acc_linear, acc_cnn]):
    coluna.metric(nome, "Pikachu" if p >= 0.5 else "Não é Pikachu")
    coluna.progress(p, text=f"probabilidade de Pikachu: {100 * p:.0f}%")
    coluna.caption(f"Acurácia no teste, sem perturbação: {100 * acc:.1f}%")

st.caption("O k da barra lateral muda só a reconstrução; o classificador linear usa sempre 50. "
           "Os modelos só conhecem duas respostas e podem errar com confiança.")

with st.expander("Ver os primeiros Eigen-Pokémons"):
    for i, coluna in enumerate(st.columns(6)):
        v = formato(Vt[i])
        coluna.image((v - v.min()) / (v.max() - v.min()), caption=f"componente {i + 1}", width=128)
