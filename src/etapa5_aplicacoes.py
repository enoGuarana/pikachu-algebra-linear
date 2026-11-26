"""Etapa 5 (Luigi): aplicações reais de SVD/PCA."""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from comum import ajustar_svd, carregar, formato, salvar_figura, salvar_tabela, vetorizar

dados = carregar()
(tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))
X_tr, X_va, X_te = (vetorizar(a) for a in (tr, va, te))
media, _, Vt = ajustar_svd(X_tr)


def reconstruir(X, media, base):
    return media + (X - media) @ base.T @ base


resumo = {}

# 1) Compressão: cada foto vira k pesos (a base e a média são guardadas uma vez só)
n, d = X_tr.shape
resumo["compressao"] = {
    f"k={k}": {
        "fator_para_o_treino": round(n * d / (n * k + k * d + d), 1),
        "erro_medio_por_pixel_no_teste": round(float(np.sqrt(((X_te - reconstruir(X_te, media, Vt[:k])) ** 2).mean())), 4),
    }
    for k in [20, 50, 200]
}

# 2) Remoção de ruído: projetar nos k componentes descarta parte do ruído
rng = np.random.default_rng(0)
ruidoso = X_te + rng.normal(0, 0.2, X_te.shape).astype(np.float32)
erros = {k: float(((reconstruir(X_va + rng.normal(0, 0.2, X_va.shape).astype(np.float32), media, Vt[:k]) - X_va) ** 2).mean())
         for k in [20, 50, 100, 200, 400]}
k_ruido = min(erros, key=erros.get)
limpo = reconstruir(ruidoso, media, Vt[:k_ruido])
resumo["ruido"] = {
    "k": k_ruido,
    "erro_quadratico_com_ruido": float(((ruidoso - X_te) ** 2).mean()),
    "erro_quadratico_apos_projecao": float(((limpo - X_te) ** 2).mean()),
}
i = int(y_te.argmax())
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
for a, (v, titulo) in zip(ax, [(X_te[i], "original"), (ruidoso[i], "com ruído"), (limpo[i], f"projetada, k = {k_ruido}")]):
    a.imshow(np.clip(formato(v), 0, 1))
    a.set_title(titulo)
    a.axis("off")
salvar_figura("5_ruido.png")

# 3) Detecção de anomalias: o PCA conhece SÓ Pikachus; erro alto = "estranho"
media_p, _, Vt_p = ajustar_svd(X_tr[y_tr == 1])


def erro(X, k):
    return np.linalg.norm(X - reconstruir(X, media_p, Vt_p[:k]), axis=1)


k_anom = max([5, 10, 20, 50, 100, 200], key=lambda k: roc_auc_score(1 - y_va, erro(X_va, k)))
e_va, e_te = erro(X_va, k_anom), erro(X_te, k_anom)
limiares = np.unique(e_va)
limiar = max(limiares, key=lambda t: ((e_va <= t) == y_va).mean())
resumo["anomalia"] = {
    "k": k_anom,
    "auc_no_teste": float(roc_auc_score(1 - y_te, e_te)),
    "acuracia_no_teste": float(((e_te <= limiar) == y_te).mean()),
}
plt.figure(figsize=(7, 4))
bins = np.linspace(e_te.min(), e_te.max(), 30)
plt.hist(e_te[y_te == 1], bins, alpha=0.7, label="Pikachu", color="goldenrod")
plt.hist(e_te[y_te == 0], bins, alpha=0.6, label="não Pikachu", color="tab:blue")
plt.axvline(limiar, color="black", ls="--", label="limiar (validação)")
plt.xlabel("erro de reconstrução")
plt.ylabel("número de imagens")
plt.legend()
salvar_figura("5_anomalia.png")

# 4) Custo: classificar com poucos números é barato, mesmo sem GPU
pesos = np.random.default_rng(0).normal(size=50).astype(np.float32)
x = X_te[:1]
inicio = time.perf_counter()
for _ in range(2000):
    ((x - media) @ Vt[:50].T) @ pesos
resumo["inferencia_pca_k50_ms_por_imagem"] = round((time.perf_counter() - inicio) / 2000 * 1000, 3)

# 5) Recomendação: SVD numa matriz pequena de notas (exemplo INVENTADO, só didático)
filmes = ["Ação 1", "Ação 2", "Romance 1", "Romance 2"]
notas = np.array([[5, 4, 1, 1], [4, 5, 1, 2], [1, 1, 5, 4], [2, 1, 4, 5], [5, 0, 1, 1]], dtype=float)
conhecida = notas > 0
cheia = np.where(conhecida, notas, notas[conhecida].mean())
for _ in range(100):  # SVD de posto 2, preenchendo repetidamente a nota que falta
    U, S, Wt = np.linalg.svd(cheia, full_matrices=False)
    cheia = np.where(conhecida, notas, (U[:, :2] * S[:2]) @ Wt[:2])
resumo["recomendacao_exemplo_didatico"] = {"usuario": 5, "filme": filmes[1], "nota_prevista": round(float(cheia[4, 1]), 2)}

print(resumo)
salvar_tabela("etapa5.json", resumo)
