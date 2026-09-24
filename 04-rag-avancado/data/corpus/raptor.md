# RAPTOR: recuperação em árvore com sumarização recursiva

"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", de Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie e Christopher D. Manning, da Stanford University, foi publicado como artigo de conferência no ICLR 2024.

## O problema

A maioria dos métodos de recuperação traz apenas alguns trechos curtos e contíguos do corpus. Isso limita a capacidade de representar e aproveitar a estrutura do discurso em larga escala. O problema aparece em perguntas temáticas, que exigem integrar conhecimento de várias partes de um texto, como entender um livro inteiro.

O exemplo do artigo é o conto da Cinderela e a pergunta "Como a Cinderela chegou ao seu final feliz?". Os k trechos curtos mais bem ranqueados não contêm contexto suficiente para responder, porque a resposta está distribuída ao longo da narrativa.

## A construção da árvore

A construção começa segmentando o corpus em textos curtos e contíguos de comprimento 100, como na recuperação tradicional. Se uma frase excede o limite de 100 tokens, a frase inteira é movida para o próximo chunk, em vez de ser cortada no meio. Isso preserva a coerência contextual e semântica do texto dentro de cada chunk. Os chunks são embeddados com SBERT, no modelo multi-qa-mpnet-base-cos-v1, e formam as folhas da árvore.

Os chunks são então agrupados por clusterização. Uma vez agrupados, um modelo de linguagem resume os textos de cada grupo. Esses resumos são re-embeddados, e o ciclo de embedding, clusterização e sumarização se repete até que continuar agrupando se torne inviável. O resultado é uma representação estruturada em múltiplas camadas do corpus original, construída de baixo para cima.

## A clusterização

Um aspecto distintivo é o uso de clusterização suave (soft clustering), em que um nó pode pertencer a vários clusters sem exigir um número fixo de grupos. Essa flexibilidade é essencial porque um trecho de texto frequentemente contém informação relevante para vários tópicos, justificando sua inclusão em múltiplos resumos.

O algoritmo usa Gaussian Mixture Models. Como a alta dimensionalidade dos embeddings prejudica as métricas de distância, aplica-se antes o UMAP para redução de dimensionalidade. O parâmetro de vizinhos mais próximos do UMAP é variado para criar uma estrutura hierárquica: primeiro identifica clusters globais, depois faz clusterização local dentro deles. O número ótimo de clusters é determinado pelo Critério de Informação Bayesiano (BIC), e os parâmetros do GMM são estimados por Expectation-Maximization.

Se o contexto combinado de um cluster local excede o limite de tokens do modelo de sumarização, a clusterização é aplicada recursivamente dentro do cluster.

## As duas estratégias de consulta

O método de travessia da árvore (tree traversal) percorre a árvore camada por camada, podando e selecionando os nós mais relevantes em cada nível. Começa na camada raiz, calcula a similaridade de cosseno entre o embedding da consulta e os embeddings de todos os nós daquela camada, escolhe os top-k, considera os filhos desses nós na camada seguinte, e repete até chegar às folhas.

O método de árvore colapsada (collapsed tree) avalia todos os nós de todas as camadas coletivamente, encontrando os mais relevantes de uma vez, independentemente do nível em que estão.

## Resultados

O RAPTOR supera métodos tradicionais de recuperação aumentada em várias tarefas. Acoplado ao GPT-4, melhora em 20 pontos percentuais absolutos o melhor desempenho no benchmark QuALITY. Estabelece novos estados da arte em três tarefas de perguntas e respostas: NarrativeQA sobre livros e filmes, QASPER sobre artigos de PLN, e QuALITY sobre passagens de comprimento médio. Um estudo de anotação revelou que cerca de 4% dos resumos continham alucinações menores, que não se propagaram para os nós pais nem tiveram efeito perceptível nas tarefas de perguntas e respostas.
