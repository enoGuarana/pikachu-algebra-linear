"""Etapa 2 (Isaac): SVD, Eigen-Pokémons e reconstrução."""
import matplotlib.pyplot as plt
import numpy as np

from comum import ajustar_svd, carregar, formato, para_exibir, salvar_figura, salvar_tabela, vetorizar

dados = carregar()
imgs, y = dados["train"]
X = vetorizar(imgs)
media, S, Vt = ajustar_svd(X)
variancia = np.cumsum(S**2) / np.sum(S**2)

fig, ax = plt.subplots(3, 6, figsize=(12, 6.5))
for a, i in zip(ax.ravel(), list(range(15)) + [99, 299, 599]):
    a.imshow(para_exibir(Vt[i]))
    a.set_title(f"componente {i + 1}")
    a.axis("off")
salvar_figura("2_eigen_pokemons.png")

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(S) + 1), 100 * variancia)
plt.xlabel("número de componentes k")
plt.ylabel("variância acumulada (%)")
plt.grid(alpha=0.3)
salvar_figura("2_variancia.png")

# Reconstrução de um Pikachu do TESTE (nunca visto pelo SVD)
teste, y_teste = dados["test"]
originais = [("treino", imgs[y == 1][0]), ("teste", teste[y_teste == 1][0])]
ks = [5, 20, 50, 200, len(S)]
fig, ax = plt.subplots(2, len(ks) + 1, figsize=(13, 4.8))
for linha, (nome, img) in enumerate(originais):
    x = vetorizar(img[None])[0]
    ax[linha, 0].imshow(img)
    ax[linha, 0].set_title(f"original ({nome})")
    for col, k in enumerate(ks, start=1):
        pesos = Vt[:k] @ (x - media)  # projeção: k números
        rec = media + pesos @ Vt[:k]  # soma ponderada dos moldes
        ax[linha, col].imshow(np.clip(formato(rec), 0, 1))
        ax[linha, col].set_title(f"k = {k}")
for a in ax.ravel():
    a.axis("off")
salvar_figura("2_reconstrucao.png")

resumo = {f"variancia_com_k={k}": round(float(variancia[k - 1]), 4) for k in [1, 5, 20, 50, 100, 200]}
resumo["k_para_90_por_cento"] = int(np.searchsorted(variancia, 0.90) + 1)
resumo["k_para_95_por_cento"] = int(np.searchsorted(variancia, 0.95) + 1)
resumo["erro_ortonormalidade_VtV_menos_I"] = float(np.abs(Vt[:50] @ Vt[:50].T - np.eye(50)).max())
print(resumo)
salvar_tabela("etapa2.json", resumo)
