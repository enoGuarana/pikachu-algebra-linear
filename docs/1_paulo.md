# Roteiro individual: Paulo

**Tema:** imagens como vetores e matrizes
**Tempo:** 10 minutos (0:00 a 10:00), slides 1 a 6
**Código:** `src/etapa1_vetorizacao.py`
**Mensagem que a turma deve levar:** para o computador, uma foto é um vetor e o dataset é uma matriz.

## Minuto a minuto

| Tempo | Slide | Assunto |
|---|---|---|
| 0:00–1:00 | 1 | Título, grupo e divisão das falas |
| 1:00–2:00 | 2 | A pergunta central |
| 2:00–3:00 | 3 | Votação: é Pikachu? |
| 3:00–4:30 | 4 | Dataset e armadilhas |
| 4:30–7:00 | 5 | Foto vira vetor |
| 7:00–10:00 | 6 | Imagem média e centralização |

**Votação (slide 3):** peça que levantem a mão para A, B e C e anote os números no quadro. Não revele: Luca mostra as respostas no slide 15.

## Falas sugeridas

**Slide 1.** "Somos Paulo, Isaac, Luca, Enomoto e Luigi. Vamos mostrar como a Álgebra Linear ajuda um computador a reconhecer um Pikachu, e onde ela deixa de ser suficiente."

**Slide 2.** Mostre a reconstrução (`2_reconstrucao.png`) aos poucos. "Levantem a mão quando reconhecerem." Depois: "A pergunta do trabalho é: até onde dá para reconhecer um Pikachu usando Álgebra Linear?"

**Slide 4.** "Usamos um dataset do Kaggle com 1.290 imagens, metade Pikachu e metade não. Há bananas, flores amarelas, cães, outros Pokémon e até fantasias de Pikachu. Encontramos imagens quase repetidas entre treino e teste e as retiramos da avaliação, senão o resultado ficaria inflado."

| Divisão | Imagens | Pikachu | Para que serve |
|---|---|---|---|
| Treino | 774 | 387 | Aprender |
| Validação | 241 | 126 | Escolher ajustes |
| Teste | 243 | 126 | Medir o resultado |

**Slide 5.** "Cada foto foi reduzida para 64 × 64. Cada pixel tem vermelho, verde e azul. São 64 × 64 × 3 = 12.288 números. Se colocarmos em fila, temos um vetor. Empilhando as 774 fotos de treino, temos uma matriz 774 × 12.288."

**Slide 6.** "A foto média é um borrão. O Pikachu médio é amarelado e o não Pikachu médio é acinzentado. Subtraindo a média, sobra o que cada foto tem de diferente."

## A matemática, passo a passo

### 1. Imagem como vetor
Uma imagem colorida é um arranjo 64 × 64 × 3. Achatar é escolher uma ordem fixa para os valores:

x ∈ ℝ¹²²⁸⁸, com cada entrada entre 0 e 1 (valor do pixel dividido por 255).

Em cinza, cada pixel vira 0,299 R + 0,587 G + 0,114 B, e o vetor tem 4.096 entradas. Essa conta é um **produto interno** entre (R, G, B) e um vetor de pesos.

### 2. Matriz de dados
X tem uma foto por linha. A linha i é xᵢᵀ; a coluna j reúne o valor da posição j em todas as fotos.

### 3. Distância entre fotos
‖a − b‖² = ‖a‖² + ‖b‖² − 2 aᵀb

### 4. Média e centralização
μ = (1/n) Σ xᵢ e Xc = X − 1μᵀ, onde 1 é o vetor de uns.

Em forma de matriz: Xc = HX, com H = I − (1/n) 11ᵀ.

H é simétrica e H² = H, ou seja, é uma **projeção ortogonal**. Aplicar duas vezes não muda mais nada.

### 5. Posto
As linhas de Xc somam o vetor zero, logo são linearmente dependentes. O posto é no máximo n − 1 = 773. Mesmo com 12.288 colunas, as fotos de treino centralizadas ocupam um subespaço de no máximo 773 dimensões.

## Exemplo no quadro (cerca de 2 minutos, dentro do slide 6)

Duas "fotos" 2 × 2: A com linhas (8, 6) e (4, 2); B com linhas (2, 4) e (6, 8).

1. Vetorizar: a = (8, 6, 4, 2) e b = (2, 4, 6, 8).
2. Média: μ = (5, 5, 5, 5).
3. Centralizar: a − μ = (3, 1, −1, −3) e b − μ = (−3, −1, 1, 3).
4. Observar: as linhas somam zero e uma é o oposto da outra. O posto é 1 = n − 1.
5. Distância: ‖a − b‖² = 36 + 4 + 4 + 36 = 80.

## Figuras
- `1_foto_vira_vetor.png`: foto, versão cinza, zoom nos pixels e o vetor (em cinza, 4.096 números).
- `1_imagem_media.png`: média geral, Pikachu médio, não Pikachu médio e uma foto menos a média.

## Cuidados
- A média é calculada **só com o treino** e reaproveitada na validação e no teste.
- Redimensionar para um quadrado distorce algumas fotos.

## Perguntas prováveis

**Achatar não destrói a imagem?** Não se perde valor algum além da redução para 64 × 64, e dá para desfazer. O que o método ignora é quais posições são vizinhas.

**Por que 64 × 64?** Para o cálculo rodar em um computador comum.

**Por que dividir por 255?** Para os valores ficarem entre 0 e 1.

**Como acharam as repetidas?** Com um resumo de 64 bits de cada imagem (hash perceptual). Imagens com até 3 bits diferentes foram consideradas quase iguais.

**As classes são equilibradas?** No treino, exatamente. No teste, 126 contra 117: chutar sempre "Pikachu" daria 51,9%.

## Passagem para Isaac
"São 12.288 números por foto. Será que precisamos de todos?"

## Antes do dia
- [ ] Rodar `python src/etapa1_vetorizacao.py` e abrir as figuras.
- [ ] Fazer o exemplo do quadro sem consultar.
- [ ] Ensaiar com cronômetro.
