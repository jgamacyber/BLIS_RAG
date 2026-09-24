# HyDE: Hypothetical Document Embeddings

"Precise Zero-Shot Dense Retrieval without Relevance Labels", de Luyu Gao, Xueguang Ma, Jimmy Lin e Jamie Callan, foi publicado em 2022 pela Carnegie Mellon University e pela University of Waterloo.

## O problema

A recuperação densa funciona bem, mas construir um sistema totalmente zero-shot é difícil quando não existe nenhum rótulo de relevância. A dificuldade está na própria definição: é preciso aprender duas funções de codificação, uma para consultas e outra para documentos, que mapeiem para o mesmo espaço onde o produto interno capture relevância. Sem julgamentos de relevância para ajustar, esse aprendizado se torna intratável.

A maior parte dos trabalhos assume uma configuração de aprendizado por transferência, treinando o recuperador no MS-MARCO e aplicando em outras tarefas. Mas o MS-MARCO é um conjunto enorme e caro de produzir, restringe uso comercial, e nem sempre pode ser assumido como disponível.

## A ideia

O HyDE decompõe a recuperação densa em duas tarefas: uma tarefa generativa executada por um modelo de linguagem que segue instruções, e uma tarefa de similaridade documento-documento executada por um codificador contrastivo.

Primeiro, a consulta é enviada ao modelo generativo com a instrução de escrever um documento que responda à pergunta. O resultado é um documento hipotético. Esse documento não é real, pode conter erros factuais e detalhes inventados. Isso não importa: espera-se apenas que ele capture o padrão de relevância, dando um exemplo do que seria uma resposta.

Segundo, um codificador contrastivo não supervisionado, como o Contriever, codifica esse documento hipotético em um vetor. Esse vetor identifica uma vizinhança no espaço de embeddings do corpus, onde documentos reais similares são recuperados por similaridade vetorial.

## Por que funciona

O ponto sutil é o papel do codificador. A função de codificação atua como um compressor com perdas: ao produzir um vetor denso, os detalhes extras e alucinados do documento hipotético são filtrados e deixados de fora. Esse segundo passo ancora o documento gerado no corpus real.

Com a fatoração do HyDE, o score de similaridade entre consulta e documento deixa de ser modelado explicitamente. A tarefa de recuperação é convertida em duas tarefas de compreensão e geração de linguagem natural. Em outras palavras, a modelagem de relevância é transferida do modelo de representação para um modelo generativo, que generaliza muito mais facilmente.

## A agregação

Na prática, os autores amostram N documentos hipotéticos do modelo generativo, em vez de um só, e fazem a média dos embeddings. Consideram também a própria consulta como uma hipótese possível. O vetor final de consulta é a média de N+1 vetores: os N documentos hipotéticos codificados mais a consulta codificada. Essa inclusão da consulta original é uma salvaguarda importante, porque impede que uma geração muito ruim desvie completamente a busca.

As gerações usam temperatura 0,7 e instruções específicas por tarefa, do tipo "escreva um parágrafo que responda à pergunta".

## Resultados

O HyDE supera significativamente o Contriever, o estado da arte em recuperação densa não supervisionada, e tem desempenho comparável ao de recuperadores ajustados com dados de relevância. Os ganhos aparecem em tarefas variadas, incluindo busca na web, perguntas e respostas e verificação de fatos, e em vários idiomas, como suaíli, coreano e japonês. Nenhum modelo é treinado no HyDE: tanto o modelo generativo quanto o codificador contrastivo permanecem intactos.
