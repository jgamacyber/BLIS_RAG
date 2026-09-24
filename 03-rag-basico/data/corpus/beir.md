# BEIR: benchmark heterogêneo para avaliação zero-shot de recuperação

"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", de Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava e Iryna Gurevych, do laboratório UKP-TUDA da Technische Universität Darmstadt, foi publicado em 2021.

## A motivação

Modelos neurais de recuperação de informação vinham sendo estudados em cenários homogêneos e estreitos, geralmente treinados e avaliados no mesmo conjunto, como Natural Questions ou MS MARCO. Isso limitava muito o entendimento sobre a capacidade de generalização fora da distribuição. Como criar um corpus grande de treino é caro e demorado, muitos sistemas acabam aplicados em cenário zero-shot, sem dados de treino disponíveis.

## O benchmark

O BEIR, de Benchmarking IR, reúne 18 conjuntos de dados públicos cobrindo 9 tarefas distintas de recuperação: verificação de fatos, predição de citação, recuperação de perguntas duplicadas, recuperação de argumentos, recuperação de notícias, perguntas e respostas, recuperação de tweets, recuperação biomédica e recuperação de entidades.

A diversidade é deliberada. Os domínios vão de tópicos gerais da Wikipédia a assuntos especializados como publicações sobre COVID-19. Os tipos textuais variam de artigos de notícias a tweets. Os tamanhos vão de 3,6 mil a 15 milhões de documentos. O comprimento médio das consultas varia de 3 a 192 palavras e o dos documentos de 11 a 635 palavras.

## Os achados

Os autores avaliam dez sistemas de recuperação de cinco arquiteturas amplas: lexical, esparsa, densa, de interação tardia e de reranking.

O resultado mais citado é que o BM25 é um baseline robusto para recuperação zero-shot. Nenhuma abordagem domina consistentemente as outras em todos os conjuntos.

Modelos de reranking e de interação tardia alcançam, em média, o melhor desempenho zero-shot, mas a um custo computacional alto. Já modelos de recuperação densa e esparsa são computacionalmente muito mais eficientes, porém frequentemente ficam abaixo das outras abordagens, evidenciando o espaço considerável para melhoria na capacidade de generalização.

Outro achado importante: o desempenho dentro da distribuição de treino não se correlaciona bem com a capacidade de generalização. Modelos ajustados com dados de treino idênticos podem generalizar de formas muito diferentes.

Os autores também notam que pode haver um viés lexical forte nos conjuntos incluídos, provavelmente porque modelos lexicais são predominantemente usados na anotação ou criação dos dados, o que desfavorece injustamente abordagens não lexicais.

A métrica principal reportada no BEIR é o nDCG@10.
