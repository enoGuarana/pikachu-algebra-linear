"""Etapa 6: a Álgebra Linear do projeto, verificada numericamente."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from comum import ajustar_svd, carregar, formato, salvar_figura, salvar_tabela, vetorizar

dados = carregar()
(tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))
# float64 para que as igualdades possam ser conferidas com precisão
X_tr, X_va, X_te = (vetorizar(a).astype(np.float64) for a in (tr, va, te))
media, S, Vt = ajustar_svd(X_tr)
Xc = X_tr - media
k = 50
Vk = Vt[:k].T  # 12288 x 50, colunas ortonormais
r = {}

# 1) Autovetor pelo método da potência: v <- Cv / ||Cv||, com C = XcT Xc
v = np.random.default_rng(0).normal(size=Xc.shape[1])
v /= np.linalg.norm(v)
convergencia = []
for _ in range(30):
    v = Xc.T @ (Xc @ v)  # sem montar a matriz 12288 x 12288
    v /= np.linalg.norm(v)
    convergencia.append(1 - abs(v @ Vt[0]))
r["potencia_cosseno_com_svd"] = float(abs(v @ Vt[0]))
r["potencia_autovalor_sobre_sigma1_ao_quadrado"] = float(np.linalg.norm(Xc @ v) ** 2 / S[0] ** 2)

# 2) Autovalores da matriz de Gram (774 x 774) são os quadrados dos valores singulares
autovalores = np.linalg.eigvalsh(Xc @ Xc.T)[::-1]
r["gram_maior_erro_relativo_50_primeiros"] = float(np.max(np.abs(autovalores[:k] / S[:k] ** 2 - 1)))

# 3) Projeção ortogonal e Pitágoras: ||x||² = ||projeção||² + ||resíduo||²
x = X_te[int(y_te.argmax())] - media
proj = Vk @ (Vk.T @ x)
res = x - proj
r["pitagoras"] = {
    "norma2_x": float(x @ x), "norma2_projecao": float(proj @ proj), "norma2_residuo": float(res @ res),
    "produto_interno_projecao_residuo": float(proj @ res),
}

# 4) Eckart-Young: erro² do posto k = soma dos sigma² descartados
ks = [1, 5, 10, 20, 50, 100, 200, 400]
medido = [np.linalg.norm(Xc - (Xc @ Vt[:i].T) @ Vt[:i]) ** 2 for i in ks]
teoria = [np.sum(S[i:] ** 2) for i in ks]
r["eckart_young_maior_erro_relativo"] = float(np.max(np.abs(np.array(medido) / np.array(teoria) - 1)))
# comparação: uma base ortonormal ALEATÓRIA de 50 direções
Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(Xc.shape[1], k)))
r["erro2_k50_svd_vs_base_aleatoria"] = [
    float(np.sum(S[k:] ** 2)), float(np.linalg.norm(Xc - (Xc @ Q) @ Q.T) ** 2)]

# 5) O classificador visto como imagem: w·z + b = (Vk w)·(x - média) + b
clf = LogisticRegression(C=0.01, max_iter=5000).fit(Xc @ Vk, y_tr)
w_pixels = Vk @ clf.coef_[0]
direto = (X_te - media) @ w_pixels + clf.intercept_[0]
r["w_como_imagem_maior_diferenca"] = float(np.abs(direto - clf.decision_function((X_te - media) @ Vk)).max())

# 6) Classificador linear direto nos 12288 pixels, sem PCA (C escolhido na validação)
modelos = [LogisticRegression(C=C, max_iter=5000).fit(Xc, y_tr) for C in [1e-4, 1e-3, 1e-2, 1e-1]]
bruto = max(modelos, key=lambda m: m.score(X_va - media, y_va))
r["acuracia_teste"] = {
    "pca_50": float(clf.score((X_te - media) @ Vk, y_te)),
    "pixels_sem_pca": float(bruto.score(X_te - media, y_te)),
}

# 7) Redes neurais: sem ativação, duas camadas viram uma só matriz
rng = np.random.default_rng(0)
W1, W2, e = rng.normal(size=(8, 16)), rng.normal(size=(3, 8)), rng.normal(size=16)
relu = lambda a: np.maximum(a, 0)
r["camadas_sem_ativacao_diferenca"] = float(np.abs(W2 @ (W1 @ e) - (W2 @ W1) @ e).max())
r["camadas_com_relu_diferenca"] = float(np.abs(W2 @ relu(W1 @ e) - (W2 @ W1) @ e).max())

# 8) Convolução é multiplicação por uma matriz esparsa com pesos repetidos
img = X_te[int(y_te.argmax())].reshape(64, 64, 3).mean(axis=2)
nucleo = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)  # detector de bordas
pedaco = img[20:25, 20:25]
M = np.zeros((9, 25))
for i in range(3):
    for j in range(3):
        janela = np.zeros((5, 5))
        janela[i:i + 3, j:j + 3] = nucleo
        M[i * 3 + j] = janela.ravel()
janelas = np.lib.stride_tricks.sliding_window_view(pedaco, (3, 3))
r["convolucao_vs_matriz_diferenca"] = float(
    np.abs(M @ pedaco.ravel() - (janelas * nucleo).sum(axis=(2, 3)).ravel()).max())

for nome, valor in r.items():
    print(f"{nome}: {valor}")
salvar_tabela("etapa6.json", r)

plt.figure(figsize=(7, 4))
plt.semilogy(range(1, 31), np.maximum(convergencia, 1e-16), marker="o")
plt.xlabel("iteração")
plt.ylabel("1 - |cosseno com o autovetor do SVD|")
plt.title("Método da potência")
plt.grid(alpha=0.3)
salvar_figura("6_metodo_potencia.png")

plt.figure(figsize=(7, 4))
plt.plot(ks, teoria, label="soma dos σ² descartados (teoria)")
plt.plot(ks, medido, "o", label="erro medido")
plt.xlabel("k")
plt.ylabel("erro quadrático total")
plt.title("Teorema de Eckart–Young")
plt.legend()
plt.grid(alpha=0.3)
salvar_figura("6_eckart_young.png")

W = formato(w_pixels)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a, canal, nome in zip(ax, range(3), ["vermelho", "verde", "azul"]):
    limite = np.abs(W).max()
    mapa = a.imshow(W[:, :, canal], cmap="RdBu_r", vmin=-limite, vmax=limite)
    a.set_title(f"w no canal {nome}")
    a.axis("off")
fig.colorbar(mapa, ax=ax, label="vermelho: vota Pikachu / azul: vota contra", shrink=0.8)
FIG = __import__("comum").FIGURAS
plt.savefig(FIG / "6_w_como_imagem.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, (m, titulo) in zip(ax, [(pedaco, "pedaço 5 x 5"), (nucleo, "filtro 3 x 3"), (M, "a mesma convolução: matriz 9 x 25")]):
    a.imshow(m, cmap="gray")
    a.set_title(titulo)
    a.set_xticks([])
    a.set_yticks([])
salvar_figura("6_convolucao_matriz.png")
