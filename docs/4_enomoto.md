# Etapa 4: O limite da linearidade e a CNN (Enomoto)

**Mensagem:** o PCA compara pixels por posição; a CNN procura padrões em qualquer lugar da imagem.

## Experimento
Quatro modelos, mesmas imagens coloridas 64 x 64: PCA + Logística e uma CNN pequena (4 blocos convolucionais), cada um sem e com aumento de dados. Depois, o teste é deslocado, girado e ampliado.

## Resultados (acurácia no teste)
| Modelo | original | deslocado 16 px | girado 45° | zoom 1,6x |
|---|---|---|---|---|
| PCA | 81,1% | 52,3% | 74,9% | 79,4% |
| PCA com aumento | 81,9% | 51,9% | 79,0% | 80,2% |
| CNN | 90,7% | 85,0% | 86,3% | 83,5% |
| CNN com aumento | 89,6% | 86,0% | 88,1% | 90,0% |

Os valores da CNN são médias de 3 sementes.

## O que dizer
- **Deslocamento é o ponto fraco do PCA:** cai para perto do acaso.
- **Rotação e zoom afetaram o PCA bem menos do que esperávamos.** Hipótese (não testada): ele se apoia muito em cor e manchas grandes, que mudam pouco com giro ou ampliação no centro.
- A CNN é mais precisa e muito mais estável ao deslocamento, mas **não é imune**.
- Aumento de dados não melhorou a CNN no teste original, mas ajudou no zoom.
- Aumento de dados quase não ajudou o PCA no deslocamento.

## Conceito
- A convolução é **equivariante** à translação: se a entrada anda, o mapa de ativação anda junto.
- Pooling global torna a decisão aproximadamente **invariante**. Bordas e subamostragem quebram a exatidão.
- Rotação e zoom não vêm de graça: dependem dos dados.
- A CNN também usa Álgebra Linear (convolução é uma operação linear). A diferença está nas **não linearidades** e nos filtros **aprendidos a partir dos rótulos**.

## O que mostrar
`4_perturbacoes.png` e `4_robustez.png`.

## Limitações
Não testamos mudança de fundo, então não afirmamos nada sobre isso. As bordas espelhadas podem duplicar partes do desenho.
