"""Etapa 1 (Paulo): imagens como vetores e matrizes."""
import matplotlib.pyplot as plt

from comum import DIVISOES, carregar, formato, para_exibir, salvar_figura, salvar_tabela, vetorizar

dados = carregar()
imgs, y = dados["train"]
X = vetorizar(imgs)
X_cinza = vetorizar(imgs, cor=False)
media = X.mean(axis=0)
Xc = X - media

resumo = {
    "imagens_por_divisao": {d: len(dados[d][1]) for d in DIVISOES},
    "pikachu_por_divisao": {d: int(dados[d][1].sum()) for d in DIVISOES},
    "matriz_X_cinza": list(X_cinza.shape),
    "matriz_X_colorida": list(X.shape),
    "maior_media_de_coluna_apos_centralizar": float(abs(Xc.mean(axis=0)).max()),
}
print(resumo)
salvar_tabela("etapa1.json", resumo)

i = int(y.argmax())  # primeiro Pikachu do treino
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
ax[0].imshow(imgs[i])
ax[0].set_title("Foto 64 x 64")
ax[1].imshow(formato(X_cinza[i], False), cmap="gray")
ax[1].set_title("Escala de cinza")
ax[2].imshow(imgs[i][20:28, 28:36])
ax[2].set_title("Zoom em 8 x 8 pixels")
ax[3].plot(X_cinza[i], lw=0.4)
ax[3].set_title("A mesma foto: vetor de 4.096 números")
ax[3].set_xlabel("posição no vetor")
for a in ax[:3]:
    a.axis("off")
salvar_figura("1_foto_vira_vetor.png")

fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, (img, titulo) in zip(ax, [
    (formato(media), "Imagem média do treino"),
    (formato(X[y == 1].mean(axis=0)), "Pikachu médio"),
    (formato(X[y == 0].mean(axis=0)), "Não Pikachu médio"),
    (para_exibir(Xc[i]), "Foto menos a média"),
]):
    a.imshow(img)
    a.set_title(titulo)
    a.axis("off")
salvar_figura("1_imagem_media.png")
