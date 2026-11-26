# Etapa 2: SVD/PCA e os Eigen-Pokémons (Isaac)

**Mensagem:** toda foto do treino é uma soma ponderada de "moldes", ordenados por importância.

## A matemática
- SVD: Xc = U Σ Vᵀ.
- As linhas de Vᵀ são ortonormais e são os autovetores de XcᵀXc: os Eigen-Pokémons.
- A variância no componente i é σᵢ²/(n−1).
- Pesos de uma foto: z = V_kᵀ (x − μ). Reconstrução: x̂ = μ + V_k z.
- Eckart–Young: truncar o SVD dá a melhor aproximação de posto k em erro quadrático.

## Resultados (versão colorida)
| k | variância acumulada |
|---|---|
| 1 | 46,3% |
| 5 | 60,8% |
| 20 | 74,3% |
| 50 | 82,3% |
| 200 | 93,4% |

São necessários 128 componentes para 90% e 254 para 95%.

## O que mostrar
- `2_eigen_pokemons.png`: os primeiros são manchas de cor, fundo e silhueta central. **Não há orelhas nítidas.** Os de número alto parecem ruído.
- `2_reconstrucao.png`: o melhor slide. A foto do treino volta perfeita com todos os componentes; a foto **do teste não**, porque não pertence ao espaço gerado pelas fotos de treino.
- `2_variancia.png`: retornos decrescentes.

## Cuidados
- Muita variância não significa utilidade para classificar: o PCA não usa os rótulos.
- Componentes finais têm pouca variância, o que não garante que sejam só ruído: detalhes finos também ficam ali.
- Os valores de autovetores podem ser negativos; as imagens foram reescaladas para exibição.

## Perguntas prováveis
- *SVD ou PCA?* PCA é o SVD aplicado aos dados centralizados.
- *Por que não calcular XᵀX?* Seria 12288 x 12288 e numericamente menos estável.
