"""Funções compartilhadas por todas as etapas."""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
ZIP = RAIZ / "data" / "pikachu.zip"
FIGURAS = RAIZ / "resultados" / "figuras"
TABELAS = RAIZ / "resultados" / "tabelas"
URL = "https://www.kaggle.com/api/v1/datasets/download/hal0samuel/pikachu-classification-dataset"
LADO = 64
DIVISOES = ("train", "validation", "test")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def baixar():
    if not ZIP.exists():
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        print("Baixando o dataset (~70 MB)...")
        urllib.request.urlretrieve(URL, ZIP)


def _assinatura(img):
    """Hash perceptual de 64 bits: imagens quase iguais geram bits quase iguais."""
    a = np.asarray(img.convert("L").resize((9, 8)), dtype=float)
    return (a[:, 1:] > a[:, :-1]).ravel()


def carregar():
    """Retorna {divisão: (imagens uint8 N x 64 x 64 x 3, rótulos)}; 1 = Pikachu.

    Remove de validação/teste as imagens quase idênticas a alguma do treino
    (e do teste, as quase idênticas à validação), para não inflar a acurácia.
    """
    baixar()
    dados = {d: ([], [], []) for d in DIVISOES}
    with zipfile.ZipFile(ZIP) as z:
        for nome in sorted(z.namelist()):
            if not nome.endswith(".jpg"):
                continue
            _, divisao, classe, _ = nome.split("/")
            img = Image.open(io.BytesIO(z.read(nome))).convert("RGB")
            imgs, rotulos, hashes = dados[divisao]
            imgs.append(np.asarray(img.resize((LADO, LADO), Image.LANCZOS)))
            rotulos.append(int(classe == "pikachu"))
            hashes.append(_assinatura(img))

    saida, vistos = {}, np.empty((0, 64), dtype=bool)
    for divisao in DIVISOES:
        imgs, rotulos, hashes = (np.array(v) for v in dados[divisao])
        if len(vistos):
            repetida = np.array([((vistos != h).sum(1) <= 3).any() for h in hashes])
        else:
            repetida = np.zeros(len(imgs), dtype=bool)
        saida[divisao] = (imgs[~repetida], rotulos[~repetida])
        vistos = np.vstack([vistos, hashes])
    return saida


def vetorizar(imgs, cor=True):
    """Achata N imagens em uma matriz N x d com valores entre 0 e 1."""
    x = imgs.astype(np.float32) / 255.0
    if not cor:
        x = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return x.reshape(len(x), -1)


def ajustar_svd(X):
    """Média e SVD da matriz centralizada: X - média = U S Vt."""
    media = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - media, full_matrices=False)
    return media, S, Vt


def formato(v, cor=True):
    return v.reshape((LADO, LADO, 3) if cor else (LADO, LADO))


def para_exibir(v, cor=True):
    """Reescala um vetor qualquer (ex.: autovetor) para o intervalo 0..1."""
    v = formato(v, cor)
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def salvar_figura(nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURAS / nome, dpi=150)
    plt.close()


def salvar_tabela(nome, conteudo):
    TABELAS.mkdir(parents=True, exist_ok=True)
    with open(TABELAS / nome, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)
