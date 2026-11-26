# Etapa 5: Aplicações reais no mercado (Luigi)

**Mensagem:** SVD/PCA continua útil, desde que o problema combine com a ferramenta.

## O que testamos
| Aplicação | Resultado no nosso dataset |
|---|---|
| Compressão | k = 50 reduz o armazenamento do treino 14,3 vezes, com perda visível (erro RMS 0,158 no teste, numa escala de 0 a 1) |
| Remoção de ruído | Erro quadrático caiu de 0,040 para 0,014 (k = 400), mas a imagem fica borrada |
| Detecção de anomalias | **Não funcionou:** AUC 0,51, igual a chutar |
| Sistemas leves | Projeção com k = 50 mais produto escalar: da ordem de 0,1 ms por imagem em CPU de computador (sem contar leitura e redimensionamento; não testado em hardware embarcado) |
| Recomendação | Exemplo inventado de 5 usuários x 4 filmes: o SVD de posto 2 prevê nota 4,96 para um filme parecido com o que o usuário gostou |

## Por que a detecção de anomalias falhou
Detecção por erro de reconstrução funciona melhor quando o "normal" é muito padronizado, como a mesma peça fotografada na mesma posição numa linha de produção. Nossos Pikachus variam em pose, fundo e estilo (desenho, pelúcia, jogo), então o espaço aprendido não é específico deles. Essa é a explicação mais plausível, não uma que testamos.

## Outras aplicações (citadas, não testadas aqui)
- Imagens médicas e astronômicas: redução de ruído.
- Visualização de embeddings de LLMs em 2D/3D.

## O que mostrar
`5_ruido.png` e `5_anomalia.png` (os dois histogramas quase se sobrepõem).

## Perguntas prováveis
- *Netflix usa SVD puro?* Fatoração de matrizes é uma base clássica de recomendação; sistemas atuais combinam várias técnicas.
- *A recomendação usa o dataset?* Não. É um exemplo didático à parte, com números inventados.
