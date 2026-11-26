"""Etapa 7: atalho do fundo preto, grade de erros e filtros da CNN."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from comum import ajustar_svd, carregar, salvar_figura, salvar_tabela, vetorizar
from etapa4_cnn import SEMENTES, tensor, treinar_cnn


def fundo_preto(img):
    """Verdadeiro se pelo menos 80% da moldura (2 px) é quase preta."""
    moldura = np.concatenate([img[:2].reshape(-1, 3), img[-2:].reshape(-1, 3),
                              img[:, :2].reshape(-1, 3), img[:, -2:].reshape(-1, 3)])
    return (moldura.max(axis=1) < 30).mean() >= 0.8


def treinar(tr, y_tr, va, y_va):
    """PCA(50) + logística e três CNNs sem aumento; devolve funções de previsão."""
    media, _, Vt = ajustar_svd(vetorizar(tr))
    clf = LogisticRegression(C=0.01, max_iter=5000).fit((vetorizar(tr) - media) @ Vt[:50].T, y_tr)
    pca = lambda a: clf.predict((vetorizar(a) - media) @ Vt[:50].T)
    cnns = [treinar_cnn(tr, y_tr, va, y_va, False, s) for s in SEMENTES]

    def prever(modelo, a):
        modelo.eval()
        with torch.no_grad():
            return modelo(tensor(a)).argmax(1).numpy()

    return pca, cnns, prever


def main():
    dados = carregar()
    (tr, y_tr), (va, y_va), (te, y_te) = (dados[d] for d in ("train", "validation", "test"))
    pretos = {nome: np.array([fundo_preto(i) for i in imgs]) for nome, imgs in
              [("treino", tr), ("validacao", va), ("teste", te)]}
    r = {"fundo_preto": {
        nome: {"pikachu": int((m & (y == 1)).sum()), "nao_pikachu": int((m & (y == 0)).sum())}
        for (nome, m), y in zip(pretos.items(), [y_tr, y_va, y_te])
    }}
    sem = ~pretos["teste"]
    r["imagens_no_teste_sem_fundo_preto"] = int(sem.sum())

    for rotulo, (m_tr, m_va) in {
        "treino_completo": (np.ones(len(tr), bool), np.ones(len(va), bool)),
        "treino_sem_fundo_preto": (~pretos["treino"], ~pretos["validacao"]),
    }.items():
        print("Treinando:", rotulo, flush=True)
        pca, cnns, prever = treinar(tr[m_tr], y_tr[m_tr], va[m_va], y_va[m_va])
        p_pca = pca(te)
        p_cnns = [prever(c, te) for c in cnns]
        r[rotulo] = {
            nome: {
                "pca": float((p_pca[mask] == y_te[mask]).mean()),
                "cnn_media_3_sementes": float(np.mean([(p[mask] == y_te[mask]).mean() for p in p_cnns])),
            }
            for nome, mask in [("teste_completo", np.ones(len(te), bool)), ("teste_sem_fundo_preto", sem)]
        }
        if rotulo == "treino_completo":
            erros_pca, erros_cnn, cnn = p_pca != y_te, p_cnns[0] != y_te, cnns[0]

    r["erros_no_teste"] = {
        "pca": int(erros_pca.sum()), "cnn_semente_0": int(erros_cnn.sum()),
        "ambos": int((erros_pca & erros_cnn).sum()),
        "indices_pca": np.where(erros_pca)[0].tolist(), "indices_cnn": np.where(erros_cnn)[0].tolist(),
    }
    print({k: v for k, v in r.items() if k != "erros_no_teste"})
    salvar_tabela("etapa7.json", r)

    # Grade: linha 1 só o PCA erra, linha 2 só a CNN erra, linha 3 os dois erram
    grupos = [("só o PCA errou", erros_pca & ~erros_cnn), ("só a CNN errou", erros_cnn & ~erros_pca),
              ("os dois erraram", erros_pca & erros_cnn)]
    fig, ax = plt.subplots(3, 8, figsize=(13, 5.6))
    for linha, (titulo, mask) in zip(ax, grupos):
        for a, i in zip(linha, list(np.where(mask)[0][:8]) + [None] * 8):
            a.axis("off")
            if i is not None:
                a.imshow(te[i])
                a.set_title("era Pikachu" if y_te[i] else "não era", fontsize=9)
        linha[0].text(-8, 32, titulo, rotation=90, va="center", ha="right", fontsize=11)
    salvar_figura("7_erros.png")

    # Filtros da 1ª camada e o que cada um responde a um Pikachu
    pesos = cnn.filtros[0].weight.detach().numpy()
    img = te[int(y_te.argmax())]
    cnn.eval()
    with torch.no_grad():
        mapas = cnn.filtros[:3](tensor(img[None]))[0].numpy()
    fig, ax = plt.subplots(2, 9, figsize=(14, 3.6))
    ax[0, 0].imshow(img)
    ax[0, 0].set_title("entrada", fontsize=10)
    ax[1, 0].axis("off")
    for j in range(8):
        f = pesos[j].transpose(1, 2, 0)
        ax[0, j + 1].imshow((f - f.min()) / (f.max() - f.min()), interpolation="nearest")
        ax[0, j + 1].set_title(f"filtro {j + 1}", fontsize=10)
        ax[1, j + 1].imshow(mapas[j], cmap="magma")
    for a in ax.ravel():
        a.axis("off")
    salvar_figura("7_filtros_cnn.png")


if __name__ == "__main__":
    main()
