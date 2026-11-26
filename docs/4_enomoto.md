# Roteiro individual: Enomoto

**Tema:** o limite da linearidade e a CNN
**Tempo:** 11 minutos (29:00 a 41:00), slides 13, 14 e 16 a 21
**Código:** `src/etapa4_cnn.py`, `src/etapa6_algebra.py` e `demo.py`
**Mensagem que a turma deve levar:** o PCA compara posições fixas; a CNN procura padrões em várias posições.

## Minuto a minuto

| Tempo | Slide | Assunto |
|---|---|---|
| 29:00–32:00 | 13 | Teste de estresse com demo ao vivo |
| 32:00–34:00 | 14 | PCA contra CNN |
| 34:00–35:00 | 15 | Votação (Luca apresenta) |
| 35:00–36:00 | 16 | Onde cada um errou |
| 36:00–37:00 | 17 | Fundo preto é atalho? |
| 37:00–38:00 | 18 | Redes neurais são matrizes |
| 38:00–39:00 | 19 | O que a CNN aprendeu |
| 39:00–40:30 | 20 | Turma resolve o XOR no quadro |
| 40:30–41:00 | 21 | Resposta |

## Falas sugeridas

**Slide 13.** "Vamos mexer na foto: deslocar, girar e ampliar, e ver o que cada modelo responde." Rode a demo.

**Slide 14.** "Deslocando 16 pixels, o PCA cai para perto do acaso. A CNN perde bem menos. Rotação e zoom afetaram o PCA menos do que esperávamos."

**Slide 18.** "Uma rede neural é uma sequência de matrizes com funções não lineares entre elas. Sem essas funções, tudo vira uma matriz só."

## Demo ao vivo

```powershell
cd C:\Users\Enomoto\pikachu-algebra-linear
.venv\Scripts\streamlit run demo.py
```

1. Deixe aberta **antes** da apresentação.
2. Mostre uma foto original e as duas respostas.
3. Deslize o deslocamento horizontal e o vertical.
4. Gire e amplie.
5. Peça à turma um número de imagem de teste.

Escolha as fotos nos ensaios. O PCA não erra em todas, e a demo usa um único treino da CNN (88,5% no teste). Grave a tela como plano B.

## O experimento

- Mesmas imagens coloridas 64 × 64 para os dois métodos, cada um sem e com aumento de dados.
- CNN: 4 blocos (convolução 3 × 3, normalização, ReLU, pooling), com 16, 32, 64 e 128 canais; depois média global, dropout e camada linear.
- Treino do zero: 40 épocas, lotes de 32, otimizador AdamW; melhor época pela validação; média de 3 sementes.
- Aumento: espelhamento, deslocamento até 12 px, giro até 30°, zoom de 0,85 a 1,4.

| Modelo | original | deslocado 16 px | girado 45° | zoom 1,6x |
|---|---|---|---|---|
| PCA | 81,1% | 52,3% | 74,9% | 79,4% |
| PCA com aumento | 81,9% | 51,9% | 79,0% | 80,2% |
| CNN | 90,7% | 85,0% | 86,3% | 83,5% |
| CNN com aumento | 89,6% | 86,0% | 88,1% | 90,0% |

## A matemática, passo a passo

### 1. Camada
h = φ(Wx + b), em que φ é não linear. A ReLU é φ(a) = max(0, a).

### 2. Sem não linearidade
W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂). Continua sendo uma transformação afim.

### 3. Convolução
S(i, j) = Σₘ Σₙ I(i+m, j+n) K(m, n). A rigor é uma correlação cruzada. É linear, e a matriz equivalente é esparsa e repete pesos.

### 4. Parâmetros
Primeira camada: 16 filtros × (3 × 3 × 3) + 16 vieses = 448. Uma camada densa de 12.288 entradas para 16 saídas teria 196.624.

### 5. Equivariância e invariância
Se T desloca a imagem, então conv(Tx) = T conv(x), desconsiderando as bordas. O pooling torna a resposta aproximadamente invariante.

### 6. Aprendizado
A regra da cadeia leva o erro até os pesos. Em uma camada densa, ∂L/∂W = δxᵀ, com δ = ∂L/∂(Wx + b).

## Exemplos no quadro (slides 18 e 20)

### XOR
f(x) = wᵀ max(0, Wx + c), com W de linhas (1, 1) e (1, 1), c = (0, −1) e w = (1, −2).

| x | Wx + c | ReLU | saída |
|---|---|---|---|
| (0, 0) | (0, −1) | (0, 0) | 0 |
| (0, 1) | (1, 0) | (1, 0) | 1 |
| (1, 0) | (1, 0) | (1, 0) | 1 |
| (1, 1) | (2, 1) | (2, 1) | 0 |

### Convolução
Filtro (−1, 1). No sinal (0, 0, 1, 1) a resposta é (0, 1, 0). No sinal (0, 1, 1, 1) é (1, 0, 0): a borda andou e a resposta andou junto.

## Figuras
`4_perturbacoes.png`, `4_robustez.png`, `6_convolucao_matriz.png`.

## Erros, atalho e filtros (slides 16, 17 e 19)

**Erros:** o PCA confunde objetos amarelos; os dois erram Pikachus em poses incomuns ou fundo escuro.

**Fundo preto:** retirando essas imagens do teste, o PCA vai de 81,1% para 81,6% e a CNN de 90,7% para 91,4%. Retreinando sem elas, 83,4% e 91,7%. Não há sinal de acurácia inflada por esse atalho; outros não foram testados.

**Filtros:** são 3 × 3 coloridos; alguns realçam bordas, outros regiões de cor.

## Cuidados
- A CNN **não é imune** a deslocamento, rotação ou zoom.
- Não testamos troca de fundo.
- A explicação para o PCA resistir a rotação e zoom (apoio em cor e manchas grandes) é uma hipótese não testada.
- As bordas espelhadas podem duplicar partes do desenho.

## Perguntas prováveis

**A CNN não usa Álgebra Linear?** Usa. A diferença são as não linearidades e os filtros aprendidos com os rótulos.

**Por que não usar rede pré-treinada?** O foco é entender o método; treinamos do zero.

**A comparação é justa?** Mesmas imagens e mesmas divisões. A CNN tem mais capacidade e mais ajustes, e isso deve ser dito.

**Aumento de dados resolve para o PCA?** No nosso teste, quase não ajudou no deslocamento.

## Passagem para Luigi
"Se a CNN vence, para que ainda serve o SVD?"

## Antes do dia
- [ ] Testar a demo no computador da apresentação.
- [ ] Escolher as fotos e gravar o plano B.
- [ ] Fazer XOR e convolução sem consultar.
