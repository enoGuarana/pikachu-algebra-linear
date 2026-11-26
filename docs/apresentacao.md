# Plano da apresentação

**Título:** Isso é um Pikachu? Álgebra Linear por trás do reconhecimento de imagens

**Duração suposta:** 20 minutos + perguntas (ajustar conforme a regra da disciplina).

## Roteiro

| # | Slide | Quem | Tempo | Figura |
|---|---|---|---|---|
| 1 | Título e integrantes | Paulo | 0:30 | — |
| 2 | Pergunta: até onde vai a Álgebra Linear? | Paulo | 0:30 | exemplo das duas classes |
| 3 | O dataset e suas armadilhas | Paulo | 1:00 | grade de fotos |
| 4 | Foto vira vetor | Paulo | 1:00 | `1_foto_vira_vetor` |
| 5 | Imagem média e centralização | Paulo | 1:00 | `1_imagem_media` |
| 6 | SVD, método da potência e Eckart–Young | Isaac | 1:30 | `6_eckart_young` |
| 7 | Eigen-Pokémons | Isaac | 1:00 | `2_eigen_pokemons` |
| 8 | "Já reconhece?" reconstrução aos poucos | Isaac | 1:30 | `2_reconstrucao` |
| 9 | De 12.288 para 50 números | Luca | 1:00 | `3_dispersao_2d` |
| 10 | Quantos componentes? | Luca | 1:00 | `3_acuracia_por_k` |
| 11 | Resultados e o que o classificador "enxerga" | Luca | 1:30 | `6_w_como_imagem` |
| 12 | Teste de estresse | Enomoto | 1:00 | `4_perturbacoes` |
| 13 | PCA contra CNN | Enomoto | 1:30 | `4_robustez` |
| 14 | Redes neurais são matrizes: camadas e convolução | Enomoto | 1:30 | `6_convolucao_matriz` |
| 15 | Onde SVD ajuda | Luigi | 1:30 | `5_ruido` |
| 16 | Onde falhou para nós | Luigi | 1:00 | `5_anomalia` |
| 17 | Conclusão | Luigi | 1:00 | — |

Total: cerca de 19 minutos.

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
