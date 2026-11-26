# Etapa 1: Imagens como vetores e matrizes (Paulo)

**Mensagem:** para o computador, uma foto é uma lista de números, e o dataset inteiro é uma matriz.

## A ideia
- Uma foto 64 x 64 em cinza tem 4.096 pixels, então vira um vetor em ℝ⁴⁰⁹⁶.
- Colorida, há 3 valores por pixel (vermelho, verde, azul): ℝ¹²²⁸⁸.
- Empilhando as 774 fotos de treino: X é 774 x 4096 (cinza) ou 774 x 12288 (cor).
- Centralização: μ = (1/n) Σ xᵢ e Xc = X − μ. A nuvem de pontos passa a ter centro na origem.

## Por que centralizar
Sem subtrair a média, a primeira direção encontrada pelo SVD tende a apontar para a própria média, e não para a direção em que as fotos mais variam.

## O que mostrar
- `resultados/figuras/1_foto_vira_vetor.png`: a foto, o zoom nos pixels e o vetor.
- `resultados/figuras/1_imagem_media.png`: o Pikachu médio é um borrão **amarelo**; o não Pikachu médio é acinzentado. Isso antecipa a importância da cor.

## Cuidado
A média é calculada **só com o treino** e reaproveitada na validação e no teste.

## Perguntas prováveis
- *Achatar não perde a noção de vizinhança?* Os valores continuam todos lá, mas o método não sabe quais posições eram vizinhas. A etapa 4 mostra a consequência.
- *Por que 64 x 64?* Mantém o cálculo leve. As fotos foram redimensionadas para um quadrado, o que distorce as proporções de algumas.
