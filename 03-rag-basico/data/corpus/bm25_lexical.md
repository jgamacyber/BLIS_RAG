# BM25 e recuperação lexical

O BM25, de Best Matching 25, é uma função de ranqueamento probabilística derivada do modelo Okapi, formalizada por Robertson e Zaragoza. Continua sendo o baseline de referência em recuperação de informação, e o benchmark BEIR mostrou que ele permanece surpreendentemente robusto em cenários zero-shot, superando vários modelos neurais densos fora da distribuição de treino.

## A fórmula

O score de um documento D para uma consulta Q é a soma, sobre cada termo q da consulta, de três fatores multiplicados.

O primeiro é o IDF, frequência inversa de documento, que mede quão raro é o termo no corpus. Termos que aparecem em quase todos os documentos, como preposições, recebem IDF baixo e quase não contribuem. A forma usada no BM25 é o logaritmo de (N - n + 0,5) dividido por (n + 0,5), mais 1, onde N é o número de documentos e n o número de documentos que contêm o termo.

O segundo é a saturação da frequência do termo. A contribuição é f multiplicado por (k1 + 1), dividido por f mais k1 vezes o fator de normalização, onde f é a frequência do termo no documento. O parâmetro k1, tipicamente entre 1,2 e 2,0, controla a saturação: repetir um termo dez vezes não vale dez vezes mais do que repetir uma vez. Essa é a diferença central em relação ao TF-IDF clássico.

O terceiro é a normalização por comprimento. O fator é (1 - b + b vezes o comprimento do documento dividido pelo comprimento médio). O parâmetro b, tipicamente 0,75, controla o quanto documentos longos são penalizados. Sem isso, documentos longos ganhariam vantagem apenas por conterem mais palavras.

## Forças e limitações

A força do BM25 é o casamento exato de termos: nomes próprios, códigos, números de versão, siglas e jargões raros são recuperados com precisão. Ele não precisa de treino, roda rápido com índice invertido e é interpretável — dá para explicar exatamente por que um documento foi ranqueado.

A limitação é o chamado vocabulary gap ou lacuna lexical. Ele só recupera documentos que contêm literalmente os termos da consulta. Perguntar por "vilão" não recupera um documento que diz "bad guy", e vice-versa. É exatamente esse o ponto que motivou a recuperação densa do DPR.

## Complementaridade

Recuperação lexical e densa erram de formas diferentes. O BM25 falha na lacuna lexical; o denso falha em termos raros e fora do vocabulário de treino, como um código de produto ou um nome próprio incomum. Por isso combinam bem: é a base da busca híbrida.
