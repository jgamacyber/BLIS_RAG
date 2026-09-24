# Lost in the Middle: como os modelos usam contextos longos

"Lost in the Middle: How Language Models Use Long Contexts", de Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni e Percy Liang, foi publicado em 2023 por Stanford University, University of California Berkeley e Samaya AI.

## A pergunta

Modelos recentes aceitam contextos de entrada cada vez maiores, com janelas de 4096, 32 mil e até 100 mil tokens. Mas aceitar um contexto longo não é o mesmo que usá-lo bem. Se os modelos usassem a informação de forma robusta, o desempenho seria minimamente afetado pela posição da informação relevante dentro do contexto.

## O experimento

Os autores analisam duas tarefas: perguntas e respostas sobre múltiplos documentos e recuperação de chave-valor. Na primeira, o modelo recebe uma pergunta e k documentos, dos quais exatamente um contém a resposta e os outros k-1 são distratores. Controlam duas variáveis: o comprimento do contexto, mudando o número de documentos, e a posição da informação relevante, reordenando os documentos.

Os modelos avaliados incluem MPT-30B-Instruct, LongChat-13B com 16 mil tokens, GPT-3.5-Turbo e Claude-1.3.

## A curva em U

O resultado central é uma curva de desempenho em formato de U. O desempenho é mais alto quando a informação relevante está no começo do contexto, o que os autores chamam de viés de primazia, ou no final, viés de recência. E degrada significativamente quando o modelo precisa acessar informação no meio do contexto.

O caso mais chamativo: com 20 documentos recuperados, quando a passagem com a resposta é colocada no meio do contexto, o desempenho do GPT-3.5-Turbo na tarefa de múltiplos documentos fica abaixo do desempenho no cenário closed-book, ou seja, sem nenhum documento (56,1%). Colocar informação relevante no lugar errado é pior do que não fornecer informação alguma.

Modelos com contexto estendido frequentemente têm desempenho idêntico ao de suas contrapartes de contexto normal, indicando que uma janela maior não significa melhor uso da janela.

## Outros achados

Modelos encoder-decoder são relativamente robustos a mudanças de posição, mas apenas quando avaliados em sequências dentro do comprimento visto no treino; acima disso, a curva em U reaparece.

A contextualização ciente da consulta, que coloca a pergunta antes e depois dos documentos, resolve quase perfeitamente a tarefa sintética de chave-valor, mas muda pouco a tendência em perguntas e respostas sobre múltiplos documentos.

Mesmo modelos base, sem ajuste por instrução, exibem a curva em U.

## Implicação para RAG

Há um trade-off ao fornecer mais contexto: mais informação pode ajudar, mas também aumenta o volume sobre o qual o modelo precisa raciocinar, potencialmente reduzindo a acurácia. Em um cenário de recuperação sobre o NaturalQuestions-Open, o desempenho do modelo satura muito antes da revocação do recuperador saturar. Usar 50 documentos em vez de 20 melhora apenas cerca de 1,5% no GPT-3.5-Turbo e cerca de 1% no Claude-1.3.

A conclusão prática para sistemas de RAG é dupla: recuperar mais não é recuperar melhor, e a ordem em que as passagens entram no prompt importa. Uma mitigação direta é reordenar as passagens de modo que as mais relevantes ocupem o começo e o fim do contexto, deixando as menos relevantes no meio.
