# Estratégias de chunking

O chunk é a unidade de recuperação em um sistema de RAG: é ele que é embeddado, indexado, ranqueado e entregue ao gerador como contexto. A escolha de como dividir os documentos afeta todas as etapas seguintes.

## O trade-off central

Chunks grandes carregam mais contexto e têm maior chance de conter a resposta inteira, mas diluem o sinal do embedding. Um vetor de 1536 dimensões que representa duas mil palavras sobre cinco assuntos diferentes fica "no meio" de todos eles e próximo de nenhum. Chunks grandes também consomem mais da janela de contexto, o que interage mal com o efeito Lost in the Middle.

Chunks pequenos produzem embeddings mais precisos e focados, mas correm o risco de fragmentar a informação: a resposta fica dividida entre dois chunks e nenhum deles isoladamente é suficiente. Também perdem referências anafóricas — um chunk que começa com "esse método resolve o problema" é inútil sem saber qual método.

## Tamanho fixo

A estratégia mais simples é uma janela deslizante de N caracteres ou tokens com sobreposição. É rápida, previsível e independente de idioma. A desvantagem é cortar palavras, frases e tabelas ao meio.

A sobreposição, tipicamente de 10 a 20% do tamanho do chunk, mitiga parcialmente o problema da fronteira: uma frase cortada no fim de um chunk reaparece inteira no começo do seguinte. O custo é inflar o índice e recuperar conteúdo duplicado.

## Por frases

Agrupar frases inteiras até encher um orçamento de tamanho garante que nenhuma frase seja cortada. É a estratégia adotada pelo RAPTOR, que move a frase inteira para o próximo chunk quando ela estoura o limite de 100 tokens.

## Recursiva

A estratégia recursiva tenta primeiro respeitar a maior unidade estrutural — seções, depois parágrafos —, e só desce para frases e caracteres quando a unidade não cabe no orçamento. É o padrão recomendado na maioria dos casos porque degrada de forma controlada: preserva a estrutura quando possível e nunca falha quando não é possível.

## Estratégias que usam o conteúdo

O chunking semântico calcula embeddings de frases consecutivas e corta onde a similaridade entre frases vizinhas cai abaixo de um limiar, identificando mudanças de assunto. É mais caro, porque exige embeddar o corpus duas vezes.

O chunking estrutural respeita a marcação do documento: cabeçalhos de markdown, funções em código, linhas de tabela. Quando o documento tem estrutura explícita, costuma ser a melhor escolha, porque a estrutura já codifica a organização semântica pensada pelo autor.

## Enriquecimento de contexto

Uma prática que ataca diretamente a perda de contexto é prefixar cada chunk com metadados: o título do documento e os cabeçalhos da seção a que pertence. O DPR faz uma versão disso ao prefixar cada passagem com o título do artigo da Wikipédia, separado por um token [SEP]. O custo é baixo e o ganho em desambiguação costuma ser alto.

## Como decidir

Não existe tamanho ótimo universal. A decisão deve ser medida no seu corpus, com um conjunto de perguntas de avaliação, comparando Hit@k e nDCG@k entre as configurações. O que funciona para artigos científicos não funciona para transcrições de chamadas ou para código-fonte.
