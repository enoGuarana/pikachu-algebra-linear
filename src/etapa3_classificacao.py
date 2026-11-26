"""Etapa 3 (Luca): redução de dimensionalidade e classificação linear.

Os hiperparâmetros (k, C, classificador) são escolhidos na VALIDAÇÃO.
O teste só é usado no final.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import LinearSVC

from comum import ajustar_svd, carregar, salvar_figura, salvar_tabela, vetorizar

KS = [2, 5, 10, 20, 50, 100, 200, 400]
CS = [0.001, 0.01, 0.1, 1, 10]
MODELOS = {
    "Regressão Logística": lambda C: LogisticRegression(C=C, max_iter=5000),
    "SVM linear": lambda C: LinearSVC(C=C, max_iter=50000),
}

dados = carregar()
(tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))


def melhor_modelo(A_tr, A_va, fabrica):
    candidatos = [fabrica(C).fit(A_tr, y_tr) for C in CS]
    return max(candidatos, key=lambda m: m.score(A_va, y_va))


resultados, curvas = {}, {}
for versao, cor in [("cinza", False), ("colorido", True)]:
    X_tr, X_va, X_te = (vetorizar(a, cor) for a in (tr, va, te))
    media, _, Vt = ajustar_svd(X_tr)
    for nome, fabrica in MODELOS.items():
        pontos = []
        for k in KS:
            P_tr, P_va, P_te = ((A - media) @ Vt[:k].T for A in (X_tr, X_va, X_te))
            m = melhor_modelo(P_tr, P_va, fabrica)
            pontos.append((m.score(P_va, y_va), k, m, P_te))
        curvas[f"{nome}, {versao}"] = [p[0] for p in pontos]
        acc_va, k, m, P_te = max(pontos, key=lambda p: p[0])
        resultados[f"{nome}, {versao}"] = {
            "k": k, "C": m.C, "validacao": float(acc_va), "teste": float(m.score(P_te, y_te)),
        }
        if nome == "Regressão Logística" and cor:
            matriz = confusion_matrix(y_te, m.predict(P_te))
    if cor:
        P = (X_tr - media) @ Vt[:2].T

# Linha de base: só 3 números por foto (média de vermelho, verde e azul)
C_tr, C_va, C_te = (a.reshape(len(a), -1, 3).mean(axis=1) / 255 for a in (tr, va, te))
m = max((LogisticRegression(C=C, max_iter=5000).fit(C_tr, y_tr) for C in [0.1, 1, 10, 100, 1000]),
        key=lambda m: m.score(C_va, y_va))
resultados["Linha de base: cor média (3 números)"] = {
    "validacao": float(m.score(C_va, y_va)), "teste": float(m.score(C_te, y_te)),
}

for nome, r in resultados.items():
    print(f"{nome:42s} {r}")
salvar_tabela("etapa3.json", resultados)

plt.figure(figsize=(8, 4.5))
for nome, accs in curvas.items():
    plt.plot(KS, 100 * np.array(accs), marker="o", label=nome)
plt.xscale("log")
plt.xticks(KS, KS)
plt.xlabel("número de componentes k")
plt.ylabel("acurácia na validação (%)")
plt.grid(alpha=0.3)
plt.legend()
salvar_figura("3_acuracia_por_k.png")

plt.figure(figsize=(6, 5))
for classe, nome, c in [(0, "não Pikachu", "tab:blue"), (1, "Pikachu", "goldenrod")]:
    plt.scatter(*P[y_tr == classe].T, s=10, alpha=0.7, label=nome, color=c)
plt.xlabel("1º componente")
plt.ylabel("2º componente")
plt.legend()
salvar_figura("3_dispersao_2d.png")

ConfusionMatrixDisplay(matriz, display_labels=["não Pikachu", "Pikachu"]).plot(colorbar=False)
plt.xlabel("previsto")
plt.ylabel("verdadeiro")
salvar_figura("3_matriz_confusao.png")
