# DPR: Dense Passage Retrieval

"Dense Passage Retrieval for Open-Domain Question Answering", de Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen e Wen-tau Yih, saiu em 2020 pela Facebook AI, University of Washington e Princeton University.

## O contexto

Até então, a recuperação em perguntas e respostas de domínio aberto era dominada por métodos esparsos como TF-IDF e BM25, que casam palavras-chave de forma eficiente usando um índice invertido. Esses métodos podem ser vistos como representações em vetores esparsos de altíssima dimensão com pesagem.

A codificação densa e latente é complementar à esparsa por natureza. Sinônimos ou paráfrases formados por tokens completamente diferentes podem ser mapeados para vetores próximos. O exemplo do artigo é a pergunta "Who is the bad guy in lord of the rings?", que deve ser respondida pelo contexto "Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy". Um sistema baseado em termos tem dificuldade de casar "bad guy" com "villain"; um sistema denso faz isso naturalmente. Além disso, representações densas são aprendíveis, o que permite adaptá-las a uma tarefa específica.

Acreditava-se que aprender boas representações densas exigiria um número enorme de pares rotulados de perguntas e contextos. Antes do ORQA, métodos densos nunca haviam superado o TF-IDF ou o BM25 nessa tarefa.

## A arquitetura

O DPR usa uma arquitetura de bi-encoder (dual-encoder): um codificador de passagens que mapeia qualquer trecho de texto para um vetor de dimensão d, e um codificador de consultas que mapeia a pergunta de entrada para um vetor da mesma dimensão. Ambos são redes BERT-base independentes, sem compartilhamento de pesos, usando a representação do token [CLS] como saída, com d igual a 768.

A similaridade entre pergunta e passagem é o produto interno dos dois vetores. Formas mais expressivas de similaridade existem, como redes com atenção cruzada, mas a função precisa ser decomponível para que as representações das passagens possam ser pré-computadas. Os autores escolheram o produto interno por simplicidade, depois de verificar em ablações que funções alternativas têm desempenho comparável.

## Treinamento

Treinar os codificadores é essencialmente um problema de aprendizado de métrica: criar um espaço vetorial onde pares relevantes de pergunta e passagem fiquem mais próximos. A função de perda é a log-verossimilhança negativa da passagem positiva, comparando-a com n passagens negativas.

A escolha dos negativos é decisiva. Os autores consideram três tipos: aleatórios, tirados de qualquer lugar do corpus; BM25, as passagens mais bem ranqueadas pelo BM25 que não contêm a resposta mas casam muitos tokens da pergunta; e gold, passagens positivas de outras perguntas do mesmo lote. O melhor modelo usa passagens gold do mesmo mini-lote mais uma passagem negativa do BM25.

A técnica de negativos no lote (in-batch negatives) reaproveita o cálculo: com B perguntas no lote, a matriz de similaridade B x B fornece B ao quadrado pares de treinamento, dos quais B são positivos e o restante negativos. É uma forma barata de aumentar muito o número de exemplos.

## Resultados

O DPR supera um sistema Lucene-BM25 forte por 9 a 19 pontos percentuais absolutos em acurácia de recuperação top-20. Em acurácia top-5, o DPR atinge 65,2% contra 42,9% do BM25. Essa melhora na precisão da recuperação se traduz em melhor acurácia de ponta a ponta: 41,5% contra 33,3% do ORQA no Natural Questions.

Na inferência, o codificador de passagens é aplicado a todas as passagens offline e os vetores são indexados com FAISS. O corpus é a Wikipédia de 20 de dezembro de 2018, dividida em blocos disjuntos de 100 palavras, totalizando 21.015.324 passagens. Cada passagem recebe o título do artigo como prefixo, separado por um token [SEP]. O valor de k usado na prática fica entre 20 e 100.
