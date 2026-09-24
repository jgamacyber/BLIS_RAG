# Busca híbrida e Reciprocal Rank Fusion

A busca híbrida combina um recuperador lexical, tipicamente BM25, com um recuperador denso baseado em embeddings, para aproveitar a complementaridade entre eles. O problema técnico é como fundir duas listas ranqueadas cujos scores vivem em escalas completamente diferentes.

## O problema da escala

O score do BM25 é uma soma não normalizada que pode ir de zero a valores arbitrariamente altos, dependendo do tamanho do corpus e da raridade dos termos. A similaridade de cosseno vive entre -1 e 1. Somar os dois diretamente é incoerente.

Uma solução é normalizar os scores, por exemplo com min-max dentro de cada lista, e então fazer uma soma ponderada. Funciona, mas é frágil: a normalização depende do intervalo observado naquela consulta específica, e uma única consulta com um outlier distorce toda a escala.

## Reciprocal Rank Fusion

O RRF, proposto por Cormack, Clarke e Buettcher, resolve isso ignorando os scores e usando apenas as posições. O score fundido de um documento é a soma, sobre todos os rankings em que ele aparece, de 1 dividido por (k mais a posição do documento naquele ranking).

A constante k, tipicamente 60, amortece a influência das primeiras posições e evita que um único ranking domine a fusão. Com k igual a 60, o primeiro colocado contribui com 1/61 e o décimo com 1/70 — a diferença existe, mas é suave.

As vantagens são práticas. O RRF não exige normalização nem calibração, é robusto a escalas incompatíveis, e funciona com qualquer número de recuperadores. Um documento que aparece em posição mediana nas duas listas pode superar um documento que é primeiro em apenas uma, o que é exatamente o comportamento desejado: consenso entre sinais independentes vale mais que confiança isolada.

A desvantagem é descartar a magnitude dos scores. Se um recuperador está muito mais confiante que outro, o RRF não percebe. Em aplicações onde a diferença de qualidade entre os recuperadores é grande e conhecida, uma soma ponderada calibrada pode superar o RRF.
