# Roteiro individual: Luca

**Tema:** redução de dimensionalidade e classificação linear
**Tempo:** 9 minutos (21:00 a 29:00, mais 1 min no slide 15), slides 10 a 12 e 15
**Código:** `src/etapa3_classificacao.py` e `src/etapa6_algebra.py`
**Mensagem que a turma deve levar:** com 50 números por foto, uma fronteira linear acerta cerca de 81%.

## Minuto a minuto

| Tempo | Slide | Assunto |
|---|---|---|
| 21:00–24:00 | 10 | De 12.288 para 50 números e a ideia de hiperplano |
| 24:00–26:00 | 11 | Quantos componentes |
| 26:00–29:00 | 12 | Resultados e o que o classificador "enxerga" |
| 34:00–35:00 | 15 | Resultado da votação |

## Falas sugeridas

**Slide 10.** "Cada foto agora é um ponto num espaço de 50 dimensões. Em 2 dimensões já se vê uma tendência de separação, mas com muita mistura."

**Slide 11.** "Testamos vários valores de k e escolhemos pela validação, nunca pelo teste. Com cores, 50 foi o melhor. Em cinza, aumentar k não ajudou."

**Slide 12.** "Acertamos 81%. Só a cor média, 3 números, já dá 69%. E como w pode ser desenhado, vemos que no centro ele favorece tons amarelados."

## Resultado da votação (slide 15)

| Imagem | Rótulo | PCA | CNN |
|---|---|---|---|
| A (blocos) | não Pikachu | errou | acertou |
| B (mistura com Homer) | não Pikachu | acertou | errou |
| C (dormindo) | Pikachu | errou | errou |

"Os rótulos de A e B são discutíveis. Vocês concordam?" A CNN aqui é a sem aumento de dados, um treino.

## A matemática, passo a passo

### 1. Projeção
z = V₅₀ᵀ(x − μ)

### 2. Hiperplano
O classificador calcula s = wᵀz + b. Se s > 0, responde Pikachu. A fronteira é o conjunto s = 0, e w é perpendicular a ela. A distância com sinal de um ponto à fronteira é s/‖w‖.

### 3. Regressão Logística
p = 1/(1 + e⁻ˢ), entre 0 e 1.

Minimiza ½‖w‖² − C Σ [yᵢ ln pᵢ + (1 − yᵢ) ln(1 − pᵢ)], com y em {0, 1}.

Gradiente em w: w + C Σ (pᵢ − yᵢ) zᵢ.

### 4. SVM linear
Com y em {−1, +1}, minimiza ½‖w‖² + C Σ max(0, 1 − yᵢ(wᵀzᵢ + b))². É a perda padrão do `LinearSVC`. A margem tem largura 2/‖w‖.

### 5. O papel de C
C pequeno: penaliza mais o tamanho de w. C grande: prioriza acertar o treino, com risco de sobreajuste.

### 6. w como imagem
wᵀz = wᵀV_kᵀ(x − μ) = (V_kw)ᵀ(x − μ). O vetor V_kw tem 12.288 entradas e pode ser desenhado. Conferimos numericamente que as duas contas coincidem.

## Exemplo no quadro (slide 10)

Valores inventados para w e b: z = 2,12 (do exemplo de Isaac), w = 2, b = −1.

1. s = 2 × 2,12 − 1 = 3,24.
2. p = 1/(1 + e^(−3,24)) ≈ 0,96.
3. Fronteira: 2z − 1 = 0, ou seja, z = 0,5.
4. Distância até a fronteira: 3,24/2 = 1,62.

## Resultados

| Modelo | k | validação | teste |
|---|---|---|---|
| Logística, cinza | 2 | 69,7% | 72,0% |
| SVM, cinza | 2 | 69,7% | 71,6% |
| Logística, colorido | 50 | 81,7% | 81,1% |
| SVM, colorido | 50 | 83,0% | 81,5% |
| Cor média (3 números) | — | 73,0% | 68,7% |
| Logística nos 12.288 pixels | — | — | 80,2% |

### Matriz de confusão (Logística colorida, k = 50)

| | previsto não | previsto Pikachu |
|---|---|---|
| **é não Pikachu** | 94 | 23 |
| **é Pikachu** | 23 | 103 |

Acurácia: (94 + 103)/243 = 81,1%. Dos Pikachus, achou 103 de 126 (81,7%).

## Figuras
`3_dispersao_2d.png`, `3_acuracia_por_k.png`, `3_matriz_confusao.png`, `6_w_como_imagem.png`.

Sobre `6_w_como_imagem.png`: no centro, w é positivo no vermelho e negativo no azul. Os pesos multiplicam x − μ, a diferença em relação à média.

## Cuidados
- Não diga que há "clusters limpos": há muita sobreposição.
- Não diga que o PCA melhora a acurácia: 81,1% contra 80,2% é diferença pequena demais.

## Perguntas prováveis

**Por que k = 50?** Foi o melhor na validação entre 2, 5, 10, 20, 50, 100, 200 e 400.

**81% é confiável?** Com 243 imagens, a incerteza é de aproximadamente 5 pontos para cima ou para baixo.

**Logística ou SVM?** Empataram.

**Para que usar PCA se os pixels dão quase o mesmo?** Para trabalhar com 50 números em vez de 12.288.

**Precisa de GPU?** Não.

## Passagem para Enomoto
"81% é bom. Mas e se o Pikachu mudar de lugar?"

## Antes do dia
- [ ] Rodar `python src/etapa3_classificacao.py`.
- [ ] Fazer o exemplo do quadro sem consultar.
- [ ] Ensaiar com cronômetro.
