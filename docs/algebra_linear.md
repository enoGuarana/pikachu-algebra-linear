# A Álgebra Linear do projeto

Código: `src/etapa6_algebra.py`. Números: `resultados/tabelas/etapa6.json`.

| # | Conceito | Fórmula | O que verificamos | Quem |
|---|---|---|---|---|
| 1 | Vetores e centralização | x ∈ ℝ¹²²⁸⁸, Xc = X − μ | Matriz de treino 774 x 12288 | Paulo |
| 2 | Autovetores: método da potência | v ← Cv / ‖Cv‖, C = XcᵀXc | Chega ao mesmo 1º vetor do SVD (cosseno ≈ 1) | Isaac |
| 3 | Valores singulares e autovalores | λᵢ(Xc Xcᵀ) = σᵢ² | Erro relativo da ordem de 10⁻¹⁵ nos 50 primeiros | Isaac |
| 4 | Eckart–Young | ‖Xc − Xk‖² = Σ σᵢ² (i > k) | Teoria e medida coincidem; com k = 50, erro² de 185 mil contra 1,04 milhão de uma base aleatória | Isaac |
| 5 | Projeção ortogonal | ‖x‖² = ‖Px‖² + ‖x − Px‖² | 926,3 = 495,3 + 431,0; produto interno ≈ 0 | Isaac e Luca |
| 6 | Classificador como hiperplano | w·z + b = (V_k w)·(x − μ) + b | As duas contas coincidem; w pode ser visto como imagem | Luca |
| 7 | PCA contra pixels crus | — | 81,1% com 50 números; 80,2% com 12.288 | Luca |
| 8 | Redes sem ativação | W₂(W₁x) = (W₂W₁)x | Igual sem ativação; com ReLU, deixa de ser | Enomoto |
| 9 | Convolução como matriz | y = Mx | Mesmo resultado com uma matriz 9 x 25 esparsa e de pesos repetidos | Enomoto |

## Como explicar cada figura

- `6_metodo_potencia.png`: multiplicar repetidamente por C faz o vetor se alinhar à direção dominante. É um jeito de entender o que o SVD calcula (o `numpy` usa outro algoritmo).
- `6_eckart_young.png`: os pontos medidos caem sobre a curva da teoria. Nenhuma outra aproximação de posto k dos dados de treino erra menos.
- `6_w_como_imagem.png`: no centro, w é positivo no vermelho e negativo no azul, o que favorece tons amarelos e alaranjados ali. Nas laterais e no topo, o azul é positivo. Os pesos multiplicam x − μ, a diferença em relação à média.
- `6_convolucao_matriz.png`: cada linha da matriz é o mesmo filtro em uma posição diferente.

## Precisão dos termos

- A "convolução" das CNNs é, a rigor, uma correlação cruzada (o filtro não é invertido).
- Com viés (bias), as camadas são **afins**; sem ativação, a composição continua sendo uma única transformação afim.
- O item 7 não mostra que o PCA melhora a acurácia: a diferença é pequena demais. Mostra que 50 números preservam o suficiente para este classificador.

## A frase central
> Uma rede neural alterna transformações lineares e funções não lineares. Sem as não linearidades, todas as camadas colapsam em uma só matriz.
