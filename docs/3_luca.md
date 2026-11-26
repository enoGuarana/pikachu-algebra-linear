# Etapa 3: Redução de dimensionalidade e classificação linear (Luca)

**Mensagem:** com 50 números por foto, uma fronteira linear acerta cerca de 81%.

## A ideia
- Cada foto vira z = V_kᵀ (x − μ).
- Um classificador linear decide pelo sinal de w·z + b: um hiperplano.
- Regressão Logística transforma w·z + b em probabilidade. A SVM busca a maior margem.
- k, C e o classificador são escolhidos na **validação**; o teste é usado só no final.

## Resultados
| Modelo | k | validação | teste |
|---|---|---|---|
| Logística, cinza | 2 | 69,7% | 72,0% |
| SVM, cinza | 2 | 69,7% | 71,6% |
| Logística, colorido | 50 | 81,7% | 81,1% |
| SVM, colorido | 50 | 83,0% | 81,5% |
| Cor média (3 números) | — | 73,0% | 68,7% |

## O que dizer
- A cor vale cerca de 9 pontos no teste.
- Em cinza, mais componentes não ajudaram.
- Só a cor média já acerta bem mais que o acaso (50%): parte do problema é "tem muito amarelo?".
- Logística e SVM empatam.

## O que mostrar
- `3_acuracia_por_k.png`
- `3_dispersao_2d.png`: há tendência de separação, mas **muita sobreposição**. Não são clusters limpos. O grupo compacto à direita é de não Pikachus (vale conferir se são os sprites de fundo preto).
- `3_matriz_confusao.png`: 23 erros de cada tipo, com a Regressão Logística colorida.

## Perguntas prováveis
- *Por que k = 50?* Foi o melhor na validação entre os valores testados.
- *Isso precisa de GPU?* Não.
