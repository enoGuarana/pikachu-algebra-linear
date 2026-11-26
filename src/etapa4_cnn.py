"""Etapa 4 (Enomoto): o limite da linearidade e a comparação com uma CNN.

Quatro modelos recebem as mesmas imagens coloridas 64 x 64:
PCA + Regressão Logística e CNN pequena, cada um treinado sem e com
aumento de dados. Depois o teste é deslocado, girado e ampliado.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch import nn

from comum import ajustar_svd, carregar, salvar_figura, salvar_tabela, vetorizar

SEMENTES = [0, 1, 2]
EPOCAS = 40
PERTURBACOES = {
    "deslocamento (pixels)": ("desloc", [0, 4, 8, 12, 16]),
    "rotação (graus)": ("giro", [0, 10, 20, 30, 45]),
    "zoom (fator)": ("zoom", [1.0, 1.15, 1.3, 1.45, 1.6]),
}


def transformar(img, dx=0, dy=0, giro=0.0, zoom=1.0):
    """Gira, amplia e desloca; as bordas são espelhadas para não surgir moldura preta."""
    lado = img.shape[0]
    m = lado // 2
    grande = Image.fromarray(np.pad(img, ((m, m), (m, m), (0, 0)), mode="reflect"))
    grande = grande.rotate(giro, resample=Image.BILINEAR)
    meio = grande.width / 2
    metade = lado / (2 * zoom)
    caixa = (meio - metade - dx, meio - metade - dy, meio + metade - dx, meio + metade - dy)
    return np.asarray(grande.resize((lado, lado), Image.BILINEAR, box=caixa))


def perturbar(imgs, tipo, valor):
    if tipo == "desloc":  # metade desloca para um canto, metade para o oposto
        return np.array([
            transformar(im, dx=valor * (1 if i % 2 else -1), dy=valor * (1 if i % 4 < 2 else -1))
            for i, im in enumerate(imgs)
        ])
    if tipo == "giro":
        return np.array([transformar(im, giro=valor * (1 if i % 2 else -1)) for i, im in enumerate(imgs)])
    return np.array([transformar(im, zoom=valor) for im in imgs])


def aumentar(imgs, rng):
    """Cópias aleatórias: deslocamento, rotação, zoom e espelhamento."""
    saida = []
    for im in imgs:
        if rng.random() < 0.5:
            im = im[:, ::-1]
        saida.append(transformar(
            np.ascontiguousarray(im),
            dx=rng.uniform(-12, 12), dy=rng.uniform(-12, 12),
            giro=rng.uniform(-30, 30), zoom=rng.uniform(0.85, 1.4),
        ))
    return np.array(saida)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        camadas, entrada = [], 3
        for saida in (16, 32, 64, 128):
            camadas += [nn.Conv2d(entrada, saida, 3, padding=1), nn.BatchNorm2d(saida), nn.ReLU(), nn.MaxPool2d(2)]
            entrada = saida
        self.filtros = nn.Sequential(*camadas)
        self.cabeca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.cabeca(self.filtros(x))


def tensor(imgs):
    return torch.from_numpy(imgs.astype(np.float32) / 255).permute(0, 3, 1, 2)


def acuracia_cnn(modelo, imgs, y):
    modelo.eval()
    with torch.no_grad():
        return float((modelo(tensor(imgs)).argmax(1).numpy() == y).mean())


def treinar_cnn(tr, y_tr, va, y_va, com_aumento, semente):
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = CNN()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-2)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, EPOCAS)
    alvo = torch.from_numpy(y_tr)
    melhor, pesos = -1.0, None
    for _ in range(EPOCAS):
        modelo.train()
        x = tensor(aumentar(tr, rng) if com_aumento else tr)
        for lote in torch.from_numpy(rng.permutation(len(y_tr))).split(32):
            otimizador.zero_grad()
            nn.functional.cross_entropy(modelo(x[lote]), alvo[lote]).backward()
            otimizador.step()
        agenda.step()
        acc = acuracia_cnn(modelo, va, y_va)
        if acc > melhor:
            melhor, pesos = acc, copy.deepcopy(modelo.state_dict())
    modelo.load_state_dict(pesos)
    return modelo


def treinar_pca(tr, y_tr, va, y_va):
    media, _, Vt = ajustar_svd(vetorizar(tr))
    candidatos = []
    for k in [20, 50, 100, 200]:
        projetar = lambda a, k=k: (vetorizar(a) - media) @ Vt[:k].T
        P_tr, P_va = projetar(tr), projetar(va)
        for C in [0.001, 0.01, 0.1, 1]:
            m = LogisticRegression(C=C, max_iter=5000).fit(P_tr, y_tr)
            candidatos.append((m.score(P_va, y_va), m, projetar))
    _, m, projetar = max(candidatos, key=lambda c: c[0])
    return lambda imgs, y: float(m.score(projetar(imgs), y))


def main():
    dados = carregar()
    (tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))
    testes = {
        (tipo, v): perturbar(te, tipo, v) for tipo, valores in PERTURBACOES.values() for v in valores
    }

    rng = np.random.default_rng(0)
    tr_aum = np.concatenate([tr] + [aumentar(tr, rng) for _ in range(5)])
    y_aum = np.tile(y_tr, 6)

    modelos = {
        "PCA + Logística": [treinar_pca(tr, y_tr, va, y_va)],
        "PCA + Logística (com aumento)": [treinar_pca(tr_aum, y_aum, va, y_va)],
    }
    for nome, com_aumento in [("CNN", False), ("CNN (com aumento)", True)]:
        modelos[nome] = []
        for s in SEMENTES:
            print(f"Treinando {nome}, semente {s}...", flush=True)
            cnn = treinar_cnn(tr, y_tr, va, y_va, com_aumento, s)
            modelos[nome].append(lambda imgs, y, cnn=cnn: acuracia_cnn(cnn, imgs, y))

    resultados = {}
    for nome, avaliadores in modelos.items():
        original = [f(te, y_te) for f in avaliadores]
        resultados[nome] = {
            "teste_original": float(np.mean(original)),
            "min_max_entre_sementes": [min(original), max(original)],
        }
        for tipo, valores in PERTURBACOES.values():
            resultados[nome][tipo] = {
                str(v): float(np.mean([f(testes[tipo, v], y_te) for f in avaliadores])) for v in valores
            }
        print(nome, resultados[nome])
    salvar_tabela("etapa4.json", resultados)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for a, (titulo, (tipo, valores)) in zip(ax, PERTURBACOES.items()):
        for nome, estilo in zip(modelos, ["o-", "o--", "s-", "s--"]):
            a.plot(valores, [100 * resultados[nome][tipo][str(v)] for v in valores], estilo, label=nome)
        a.set_xlabel(titulo)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("acurácia no teste (%)")
    ax[0].legend(fontsize=8)
    salvar_figura("4_robustez.png")

    i = int(y_te.argmax())
    exemplos = [("original", te[i]), ("deslocada 12 px", transformar(te[i], dx=12, dy=12)),
                ("girada 30°", transformar(te[i], giro=30)), ("zoom 1,45x", transformar(te[i], zoom=1.45))]
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    for a, (titulo, img) in zip(ax, exemplos):
        a.imshow(img)
        a.set_title(titulo)
        a.axis("off")
    salvar_figura("4_perturbacoes.png")


if __name__ == "__main__":
    main()
