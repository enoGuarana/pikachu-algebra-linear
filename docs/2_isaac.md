# Roteiro individual: Isaac

**Tema:** SVD, PCA e Eigen-Pokémons
**Tempo:** 11 minutos (10:00 a 21:00), slides 7 a 9
**Código:** `src/etapa2_svd.py` e `src/etapa6_algebra.py`
**Mensagem que a turma deve levar:** cada foto pode ser escrita como a média mais uma soma ponderada de direções ordenadas por importância.

## Minuto a minuto

| Tempo | Slide | Assunto |
|---|---|---|
| 10:00–14:00 | 7 | SVD, método da potência e Eckart–Young, com exemplo no quadro |
| 14:00–17:00 | 8 | Eigen-Pokémons |
| 17:00–21:00 | 9 | Reconstrução |

## Falas sugeridas

**Slide 7.** "Queremos a direção em que as fotos mais variam, depois a segunda, perpendicular à primeira, e assim por diante. O SVD entrega todas de uma vez."

**Slide 8.** "Cada direção tem 12.288 números, então podemos desenhá-la. As primeiras são manchas de cor e silhueta; não aparecem orelhas nítidas, porque as fotos variam muito. As de número alto parecem ruído."

**Slide 9.** "Com 5 componentes, um borrão amarelo. Com 200, já dá para reconhecer. A foto de treino volta perfeita com todos os componentes. A de teste não, porque ela não foi usada para construir as direções."

## A matemática, passo a passo

### 1. O problema
Achar v com ‖v‖ = 1 que maximize a variância das projeções: vᵀCv, com C = XcᵀXc/(n−1).

### 2. De onde vêm os autovetores
Lagrangiana: L = vᵀCv − λ(vᵀv − 1). Derivando e igualando a zero: 2Cv − 2λv = 0, logo **Cv = λv**.

A variância nessa direção é vᵀCv = λ. Então a melhor direção é o autovetor de maior autovalor.

### 3. SVD
Xc = UΣVᵀ, com U e V de colunas ortonormais e Σ diagonal, σ₁ ≥ σ₂ ≥ … ≥ 0.

XcᵀXc = VΣUᵀUΣVᵀ = VΣ²Vᵀ

Logo as colunas de V são os autovetores de C e λᵢ = σᵢ²/(n−1).

### 4. Projeção e reconstrução
- Pesos: z = V_kᵀ(x − μ), um vetor com k números.
- Reconstrução: x̂ = μ + V_k z.
- P = V_kV_kᵀ é simétrica e P² = P.
- O resíduo é perpendicular à projeção, então vale Pitágoras: ‖x − μ‖² = ‖z‖² + ‖x − x̂‖².

### 5. Eckart–Young
Entre todas as matrizes de posto até k, a que fica mais próxima de Xc é a do SVD truncado, e:

‖Xc − X_k‖²_F = σ²ₖ₊₁ + σ²ₖ₊₂ + …

No projeto, a medida e a fórmula coincidem. Com k = 50, o erro quadrático total é cerca de 185 mil; com 50 direções ortonormais aleatórias, cerca de 1,04 milhão.

### 6. Método da potência
Comece com v qualquer e repita v ← Cv/‖Cv‖. Escrevendo v na base de autovetores, cada passo multiplica a componente i por λᵢ. A de λ₁ cresce mais que as outras, desde que λ₁ > λ₂ e v inicial não seja ortogonal a v₁. A razão (λ₂/λ₁) controla a velocidade.

### 7. Atalho de cálculo
XcXcᵀ é 774 × 774, bem menor que 12.288 × 12.288, e tem os mesmos autovalores não nulos. Se XcXcᵀu = σ²u com σ > 0, então v = Xcᵀu/σ.

## Exemplo no quadro (slide 7)

Quatro pontos já centralizados: (2, 1), (1, 2), (−2, −1), (−1, −2).

1. XᵀX tem linhas (10, 8) e (8, 10).
2. det(XᵀX − λI) = (10 − λ)² − 64 = 0, logo λ = 18 ou 2.
3. Autovetores: v₁ = (1, 1)/√2 e v₂ = (1, −1)/√2. São perpendiculares.
4. O primeiro guarda 18/20 = 90%.
5. Para (2, 1): z = 3/√2 ≈ 2,12 e a reconstrução é (1,5; 1,5).
6. Pitágoras: 5 = 4,5 + 0,5.
7. Eckart–Young: 4 pontos × 0,5 = 2 = σ₂².

Aqui os autovalores são de XᵀX; os de C seriam divididos por n − 1 = 3, sem mudar as proporções.

## Resultados nas fotos coloridas

| k | variância acumulada |
|---|---|
| 1 | 46,3% |
| 5 | 60,8% |
| 20 | 74,3% |
| 50 | 82,3% |
| 100 | 88,1% |
| 200 | 93,4% |

São necessários 128 componentes para 90% e 254 para 95%.

## Figuras
`6_metodo_potencia.png`, `6_eckart_young.png`, `2_eigen_pokemons.png`, `2_reconstrucao.png`, `2_variancia.png`.

## Cuidados
- Variância alta não significa utilidade para classificar: o PCA não usa os rótulos.
- Pouca variância não garante que seja só ruído.
- Autovetores têm valores negativos; as imagens foram reescaladas para exibição.
- O sinal de um autovetor é arbitrário: v e −v servem igualmente.

## Perguntas prováveis

**SVD e PCA são a mesma coisa?** O PCA pode ser calculado pelo SVD dos dados centralizados.

**O numpy usa o método da potência?** Não. Mostramos o método para entender o que está sendo calculado.

**Por que não montar XᵀX?** Seria uma matriz 12.288 × 12.288, e formar o produto perde precisão numérica.

**Por que a foto de teste não volta perfeita?** Porque tem uma parte fora do subespaço gerado pelas fotos de treino.

**Qual a ligação com redes neurais?** Um autoencoder linear com erro quadrático aprende o mesmo subespaço do PCA (livro *Deep Learning*, seção 14.1).

## Passagem para Luca
"Resumimos cada foto em poucos números. Dá para classificar com esse resumo?"

## Antes do dia
- [ ] Rodar `python src/etapa6_algebra.py`.
- [ ] Resolver o exemplo do quadro sem consultar.
- [ ] Ensaiar com cronômetro.
