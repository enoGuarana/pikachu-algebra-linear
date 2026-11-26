# Roteiro individual: Luigi

**Tema:** nosso projeto e aplicações reais
**Tempo:** 9 minutos (41:00 a 50:00), slides 22 a 25
**Código:** `src/etapa5_aplicacoes.py`
**Mensagem que a turma deve levar:** SVD/PCA continua útil quando o problema combina com a ferramenta.

## Minuto a minuto

| Tempo | Slide | Assunto |
|---|---|---|
| 41:00–46:00 | 22 | Nosso projeto (2 min) e onde o SVD ajuda (3 min) |
| 46:00–49:00 | 23 | Onde falhou |
| 49:00–50:00 | 24 | Conclusão |
| perguntas | 25 | Referências |

## Falas sugeridas

**Slide 22.** "Fizemos um experimento completo: fotos viram vetores, o SVD resume, um classificador decide e comparamos com uma CNN. O linear acertou 81% e a CNN cerca de 91%. A mesma ferramenta serve para outras coisas: comprimir, limpar ruído e recomendar."

**Slide 23.** "Tentamos detectar o que não é Pikachu pelo erro de reconstrução. Deu o mesmo que chutar. Mostramos porque saber onde não funciona faz parte do trabalho."

**Slide 24.** "Foto é vetor, dataset é matriz, SVD encontra direções, classificar é separar com um hiperplano, e uma rede neural alterna matrizes e funções não lineares. Obrigado."

## O que testamos

### 1. Compressão
Guardamos os pesos, a base e a média: nk + kd + d números no lugar de nd.

Com n = 774, d = 12.288 e k = 50: 9.510.912 / 665.388 ≈ 14,3.

| k | vezes menor | erro RMS por pixel no teste (escala 0 a 1) |
|---|---|---|
| 20 | 34,8 | 0,179 |
| 50 | 14,3 | 0,158 |
| 200 | 3,6 | 0,128 |

É compressão **com perda** visível.

### 2. Remoção de ruído
Somamos ruído gaussiano de desvio 0,2 e projetamos em k = 400 (escolhido na validação). Erro quadrático médio: 0,040 → 0,014. A imagem fica borrada.

Por quê: para ruído independente de média zero e variância τ², E‖Pε‖² = kτ², contra E‖ε‖² = dτ². Só sobra o ruído das k direções; o preço é perder o detalhe fora delas.

### 3. Detecção de anomalias
O PCA aprende só com Pikachus do treino. A nota é e(x) = ‖(I − P)(x − μ)‖.

Resultado: AUC 0,51 e acurácia de 53,1%, com k = 5. **Não funcionou.**

Explicação plausível, não testada: nossos Pikachus variam em pose, fundo e estilo. A técnica tende a ir melhor quando o "normal" é muito padronizado, como uma peça fotografada sempre na mesma posição.

### 4. Custo
Projetar em 50 componentes e multiplicar pelos pesos levou cerca de 0,1 ms por imagem na CPU de um computador, sem contar leitura e redimensionamento. Não testamos em hardware embarcado.

### 5. Recomendação (exemplo inventado)
Matriz de 5 usuários × 4 filmes com uma nota faltando. O SVD de posto 2, preenchendo repetidamente a lacuna, previu 4,96 para um filme parecido com o que o usuário gostou.

Formulação usual com notas faltando: minimizar Σ (Rᵢⱼ − pᵢᵀqⱼ)² nas notas conhecidas, mais uma penalidade sobre p e q.

## Exemplos no quadro (slide 22)

**Anomalia.** Com v₁ = (1, 1)/√2 do exemplo de Isaac: o ponto (2, −2) tem projeção zero e erro quadrático 8; o ponto (2, 1) tem erro 0,5.

**Recomendação.** Matriz de linhas (4, 2) e (2, ?). Se o posto é 1, ela é (2, 1)ᵀ(2, 1), logo ? = 1.

## Outras aplicações (citar sem afirmar que testamos)
- Redução de ruído em imagens médicas e astronômicas.
- Visualização de embeddings de LLMs em 2D ou 3D.

## Figuras
`5_ruido.png`, `5_anomalia.png`.

## Perguntas prováveis

**A Netflix usa SVD puro?** Fatoração de matrizes é uma base clássica; sistemas atuais combinam várias técnicas.

**A recomendação usa o dataset?** Não. É um exemplo à parte, com números inventados.

**A compressão é melhor que JPEG?** Não comparamos. O objetivo é ilustrar o conceito.

**Por que mostrar um resultado ruim?** Porque define o limite da técnica.

**Qual a principal lição?** Escolher a ferramenta pelo problema e medir antes de afirmar.

## Antes do dia
- [ ] Rodar `python src/etapa5_aplicacoes.py`.
- [ ] Fazer os dois exemplos do quadro sem consultar.
- [ ] Ensaiar a frase final de memória.
