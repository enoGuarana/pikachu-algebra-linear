# Plano da apresentação

**Título:** Isso é um Pikachu? Álgebra Linear por trás do reconhecimento de imagens

**Duração:** cerca de 50 minutos; perguntas à parte. A demo ao vivo (3 min) entra no slide 13. As referências ficam na tela durante as perguntas.

## Roteiro

| # | Slide | Quem | Tempo |
|---|---|---|---|
| 1 | Título e integrantes | Paulo | 1:00 |
| 2 | Pergunta central | Paulo | 1:00 |
| 3 | Votação: é Pikachu? | Paulo | 1:00 |
| 4 | O dataset | Paulo | 1:30 |
| 5 | Foto vira vetor | Paulo | 2:30 |
| 6 | A foto média | Paulo | 3:00 |
| 7 | SVD e Eckart–Young | Isaac | 4:00 |
| 8 | Eigen-Pokémons | Isaac | 3:00 |
| 9 | Já dá para reconhecer? | Isaac | 4:00 |
| 10 | De 12.288 para 50 números | Luca | 3:00 |
| 11 | Quantos componentes? | Luca | 2:00 |
| 12 | O que o classificador enxerga | Luca | 3:00 |
| 13 | Teste de estresse e demo | Enomoto | 3:00 |
| 14 | PCA contra CNN | Enomoto | 2:00 |
| 15 | Resultado da votação | Luca | 1:00 |
| 16 | Onde cada um errou | Enomoto | 1:00 |
| 17 | Fundo preto é atalho? | Enomoto | 1:00 |
| 18 | Redes neurais são matrizes | Enomoto | 1:00 |
| 19 | O que a CNN aprendeu | Enomoto | 1:00 |
| 20 | Exercício ao vivo: XOR | Enomoto | 1:30 |
| 21 | Resposta do XOR | Enomoto | 0:30 |
| 22 | Nosso projeto e o mercado | Luigi | 5:00 |
| 23 | Onde falhou | Luigi | 3:00 |
| 24 | Conclusão | Luigi | 1:00 |
| 25 | Referências | Luigi | 0:00 |

Total: 50 minutos.

## Conclusão sugerida
1. Álgebra Linear sozinha chega a cerca de 81% neste problema.
2. Boa parte vem da cor.
3. Deslocar a imagem quebra o método; a CNN resiste muito mais e acerta cerca de 90%.
4. SVD segue útil em compressão e ruído, mas não resolve tudo.

## Abertura
Mostrar a reconstrução com k = 5, 20, 50, 200 e pedir que levantem a mão quando reconhecerem o Pikachu.

## Perguntas difíceis
| Pergunta | Resposta curta |
|---|---|
| Escolheram k olhando o teste? | Não, pela validação. |
| O modelo reconhece o Pikachu ou só o amarelo? | Em parte o amarelo: só a cor média acerta 68,7%. |
| 90% contra 81% é significativo? | O teste tem 243 imagens; a diferença é razoável, mas uma única divisão não é prova definitiva. |
| Havia imagens repetidas? | Sim, quase duplicatas entre divisões; retiramos da avaliação. |
| A CNN foi pré-treinada? | Não, treinada do zero. |

## Antes do dia
- [ ] Cada um roda `python executar_tudo.py` ou ao menos o próprio script.
- [ ] Dois ensaios cronometrados.
- [ ] Figuras salvas nos slides (não depender de rodar ao vivo).
- [ ] Citar o dataset no último slide.
