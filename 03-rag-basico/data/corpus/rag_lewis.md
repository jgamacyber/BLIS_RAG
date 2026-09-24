# RAG: Retrieval-Augmented Generation

O artigo "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", de Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel e Douwe Kiela, foi publicado em 2020 por pesquisadores do Facebook AI Research (FAIR), University College London e New York University.

## O problema

Modelos de linguagem pré-treinados armazenam conhecimento factual nos próprios parâmetros, o que os autores chamam de memória paramétrica. Essa memória tem três defeitos: não pode ser expandida ou revisada com facilidade, não permite inspecionar a origem de uma afirmação, e produz alucinações. Em tarefas intensivas em conhecimento, o desempenho desses modelos fica atrás de arquiteturas específicas para a tarefa.

## A arquitetura

O RAG combina uma memória paramétrica (um modelo seq2seq pré-treinado) com uma memória não paramétrica (um índice vetorial denso da Wikipédia, acessado por um recuperador neural pré-treinado). O recuperador é o DPR (Dense Passage Retriever) e o gerador é o BART-large, com 400 milhões de parâmetros. Os dois componentes são treinados de ponta a ponta, tratando o documento recuperado como uma variável latente sobre a qual se marginaliza.

## RAG-Sequence e RAG-Token

Os autores propõem duas formulações que diferem em como marginalizam sobre os documentos latentes.

O RAG-Sequence usa o mesmo documento recuperado para gerar a sequência inteira. Trata o documento como uma única variável latente marginalizada por uma aproximação top-K: recupera os K documentos e o gerador produz a probabilidade da sequência de saída para cada um deles, somando as contribuições.

O RAG-Token permite que cada token gerado use um documento diferente. Recupera os top-K documentos, produz uma distribuição para o próximo token a partir de cada documento, marginaliza, e repete o processo para o token seguinte. Isso deixa o gerador combinar conteúdo de vários documentos numa única resposta.

Para tarefas de classificação de sequência, as duas formulações são equivalentes, já que a classe alvo é uma sequência de comprimento um.

## Detalhes de implementação

A fonte de conhecimento é um único dump da Wikipédia de dezembro de 2018. Cada artigo é dividido em blocos disjuntos de 100 palavras, resultando em 21 milhões de documentos. O índice MIPS é construído com FAISS usando uma aproximação Hierarchical Navigable Small World (HNSW). Durante o treinamento recupera-se K entre 5 e 10 documentos por consulta.

Durante o treinamento, atualizar o codificador de documentos seria caro, pois exigiria reindexar periodicamente todo o corpus, como faz o REALM. Os autores concluíram que esse passo não é necessário para bom desempenho: mantêm o codificador de documentos e o índice fixos, e ajustam apenas o codificador de consultas e o gerador BART.

## Resultados

O RAG estabeleceu o estado da arte em três tarefas de perguntas e respostas de domínio aberto: Natural Questions, WebQuestions e CuratedTrec, superando tanto modelos seq2seq puramente paramétricos quanto arquiteturas de recuperação e extração específicas para a tarefa. Em geração de linguagem, o RAG produz texto mais específico, diverso e factual do que um baseline BART equivalente. Na verificação de fatos do FEVER, fica a 4,3% de modelos que usam supervisão forte de recuperação.

Os autores também demonstram que a memória não paramétrica pode ser substituída para atualizar o conhecimento do modelo conforme o mundo muda, sem retreinar nada.
