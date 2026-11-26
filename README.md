# Isso é um Pikachu? Álgebra Linear por trás do reconhecimento de imagens

Projeto de seminário: **até onde dá para reconhecer um Pikachu usando só Álgebra Linear?**

## Resultados principais (243 imagens de teste, 64 x 64)

| Método | Acurácia |
|---|---|
| Linha de base: só a cor média (3 números) | 68,7% |
| PCA em cinza (k = 2) + Regressão Logística | 72,0% |
| PCA colorido (k = 50) + Regressão Logística | 81,1% |
| PCA colorido (k = 50) + SVM linear | 81,5% |
| CNN pequena (média de 3 sementes) | 90,7% |
| CNN pequena com aumento de dados (média de 3 sementes) | 89,6% |

Com 243 imagens, diferenças de poucos pontos percentuais não são conclusivas.

## Como rodar

Requer Python 3.13. Não precisa de GPU nem de conta no Kaggle.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python executar_tudo.py
```

No Linux/macOS: `python3.13 -m venv .venv && source .venv/bin/activate`.

O dataset é baixado automaticamente na primeira execução. Na máquina em que o projeto foi criado (CPU, 12 núcleos lógicos), tudo rodou em cerca de 7 min 30 s, a maior parte no treino da CNN.

## Demo interativa

```powershell
streamlit run demo.py
```

Envie uma foto, varie o número de componentes, desloque, gire ou amplie e compare PCA com CNN. Na primeira execução a CNN é treinada e salva em `modelos/`.

## Divisão do trabalho

| Etapa | Responsável | Guia | Código |
|---|---|---|---|
| 1. Imagens como vetores | Paulo | [guia](docs/1_paulo.md) | `src/etapa1_vetorizacao.py` |
| 2. SVD e Eigen-Pokémons | Isaac | [guia](docs/2_isaac.md) | `src/etapa2_svd.py` |
| 3. Classificação linear | Luca | [guia](docs/3_luca.md) | `src/etapa3_classificacao.py` |
| 4. Limite da linearidade e CNN | Enomoto | [guia](docs/4_enomoto.md) | `src/etapa4_cnn.py` |
| 5. Aplicações reais | Luigi | [guia](docs/5_luigi.md) | `src/etapa5_aplicacoes.py` |

**Destaque:** [a Álgebra Linear do projeto](docs/algebra_linear.md), verificada numericamente em `src/etapa6_algebra.py`.

Erros, teste do atalho do fundo preto e filtros da CNN: `src/etapa7_erros.py`.

Plano da apresentação: [docs/apresentacao.md](docs/apresentacao.md). Figuras e números ficam em `resultados/`.

## Dataset

[Pikachu Classification Dataset](https://www.kaggle.com/datasets/hal0samuel/pikachu-classification-dataset) (hal0 Samuel, Kaggle). Classificação binária: Pikachu ou não Pikachu.

- 1.290 imagens: 774 de treino, 258 de validação e 258 de teste, metade de cada classe.
- Retiramos da avaliação as imagens quase idênticas a outra de uma divisão anterior (hash perceptual). Restaram 241 de validação e 243 de teste.
- A licença consta como "Unknown" e as imagens vieram de busca na web. Use apenas para fins acadêmicos. O dataset não é versionado aqui, mas algumas figuras mostram imagens dele.

## Limitações

- Classes visualmente muito diferentes do Pikachu (fotos de cães, sprites em fundo preto) podem servir de atalho para os modelos.
- No teste de robustez, as bordas são preenchidas por espelhamento, o que às vezes duplica partes do desenho.
- Poucas imagens e uma única divisão de dados.
