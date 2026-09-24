# MTEB: Massive Text Embedding Benchmark

"MTEB: Massive Text Embedding Benchmark", de Niklas Muennighoff, Nouamane Tazi, Loïc Magne e Nils Reimers, da Hugging Face e da Cohere, foi publicado em 2023.

## O problema

Embeddings de texto costumavam ser avaliados em um conjunto pequeno de dados de uma única tarefa, que não cobria os casos de uso possíveis. O SimCSE e o SBERT, por exemplo, reportavam resultados apenas em similaridade textual semântica (STS) e classificação, deixando em aberto a transferibilidade para busca ou clusterização. Sabe-se que o desempenho em STS correlaciona mal com outros casos de uso reais.

Isso leva à aplicação "cega" desses modelos em novos casos de uso, ou exige trabalho incremental de reavaliação em cada tarefa. Detalhes de implementação, como pré-processamento e hiperparâmetros, também influenciam os resultados, tornando difícil saber se um ganho vem do modelo ou de um pipeline de avaliação favorável.

## O benchmark

O MTEB abrange 8 tarefas de embedding, cobrindo um total de 58 conjuntos de dados e 112 idiomas. As tarefas são: mineração bitext, classificação, clusterização, classificação de pares, reranking, recuperação, STS e sumarização. Dos 58 conjuntos, 10 são multilíngues. O BEIR é incorporado como componente da parte de recuperação.

O benchmark é construído sobre quatro desejos declarados: diversidade, para entender a usabilidade em vários casos de uso; simplicidade, com uma API que aceita qualquer modelo capaz de produzir um vetor por texto; extensibilidade, permitindo adicionar conjuntos com um único arquivo; e reprodutibilidade.

## A conclusão

Foram avaliados 33 modelos, incluindo modelos open-source e modelos acessíveis por API, como o endpoint de embeddings da OpenAI. A avaliação incluiu também medições de velocidade e memória.

O achado central é que nenhum método específico de embedding de texto domina em todas as tarefas. Isso sugere que o campo ainda não convergiu para um método universal de embedding e que nenhum foi escalado o suficiente para produzir resultados de estado da arte em todas as tarefas. O benchmarking revela fraquezas e forças específicas de cada modelo: o SimCSE, por exemplo, tem desempenho baixo em clusterização e recuperação apesar de ser forte em STS.

A implicação prática para quem monta um sistema de RAG é direta: a escolha do modelo de embedding precisa ser validada na sua tarefa e no seu idioma, e não herdada de um ranking geral.
