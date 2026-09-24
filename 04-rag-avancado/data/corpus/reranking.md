# Reranking com cross-encoders

O reranking é uma segunda etapa de ordenação aplicada sobre um conjunto pequeno de candidatos já recuperados. A arquitetura típica de busca moderna é em dois estágios: um recuperador rápido traz os top-50 ou top-100 de um corpus de milhões, e um reranker caro e preciso reordena apenas esses candidatos.

## Bi-encoder contra cross-encoder

O recuperador denso é um bi-encoder: codifica a consulta e o documento separadamente, em vetores independentes, e compara por produto interno. Essa separação é o que torna a busca viável, porque os vetores dos documentos podem ser pré-computados e indexados offline. Mas é também a limitação: a consulta nunca "vê" o documento durante a codificação, e nenhuma interação entre os termos dos dois é modelada.

O cross-encoder concatena consulta e documento em uma única entrada e passa tudo por um transformer, que aplica atenção cruzada entre todos os tokens dos dois textos. O resultado é um único score de relevância. É muito mais preciso, porque modela a interação diretamente.

O custo é proibitivo para busca: não há nada para pré-computar, então avaliar um corpus de um milhão de documentos exigiria um milhão de passagens pelo transformer por consulta. Daí a arquitetura em dois estágios — aplicar o cross-encoder a 50 candidatos é perfeitamente viável.

O BEIR confirmou empiricamente essa dinâmica: modelos de reranking e de interação tardia obtêm, em média, o melhor desempenho zero-shot, mas a um custo computacional alto.

## Reranking com LLM

Sem um cross-encoder treinado à mão, é possível usar um LLM de instrução como reranker. Há três formatos principais.

No formato pointwise, o modelo recebe a consulta e uma passagem por vez e atribui uma nota de relevância, por exemplo de 0 a 10. É simples, paralelizável e o custo cresce linearmente com o número de candidatos. A fragilidade é a calibração: notas atribuídas em chamadas independentes não são diretamente comparáveis.

No formato pairwise, o modelo compara duas passagens e diz qual é mais relevante. É mais confiável, mas exige um número quadrático de comparações.

No formato listwise, o modelo recebe todas as passagens numeradas de uma vez e devolve a ordem. Aproveita o contexto comparativo em uma única chamada, mas sofre com o problema do Lost in the Middle: passagens no meio da lista tendem a ser subvalorizadas.

O ganho real do reranking vem de aumentar o número de candidatos na primeira etapa. Recuperar 20 e reranquear para 4 dá ao sistema a chance de corrigir erros de ordenação do recuperador — algo impossível se apenas 4 forem recuperados desde o início.
