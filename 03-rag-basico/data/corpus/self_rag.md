# Self-RAG: recuperar, gerar e criticar por auto-reflexão

"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", de Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil e Hannaneh Hajishirzi, foi publicado em 2023 pela University of Washington, Allen Institute for AI e IBM Research AI.

## A crítica ao RAG convencional

O RAG padrão recupera um número fixo de passagens, independentemente de a recuperação ser necessária ou de as passagens serem relevantes. Isso traz dois problemas. Primeiro, recuperar indiscriminadamente prejudica a versatilidade do modelo e pode levar a respostas piores: se a pergunta não exige conhecimento factual externo, como "escreva uma redação sobre suas férias de verão", o contexto recuperado só atrapalha. Segundo, não há garantia de que a saída seja consistente com as passagens recuperadas, porque os modelos não são explicitamente treinados para seguir os fatos fornecidos.

## Os tokens de reflexão

O Self-RAG treina um único modelo de linguagem arbitrário para recuperar sob demanda e refletir sobre as passagens recuperadas e sobre a própria geração, usando tokens especiais chamados tokens de reflexão. Eles se dividem em tokens de recuperação e tokens de crítica.

São quatro tipos. O token Retrieve decide quando recuperar, com valores "yes", "no" ou "continue". O token IsREL avalia se a passagem recuperada fornece informação útil para resolver a entrada, com valores "relevant" ou "irrelevant". O token IsSUP verifica se as afirmações da saída são sustentadas pela passagem, com valores "fully supported", "partially supported" ou "no support". O token IsUSE avalia a utilidade geral da resposta em uma escala de 1 a 5.

## O algoritmo de inferência

Dado um prompt de entrada e a geração anterior, o modelo prevê o token Retrieve. Se for "yes", recupera as passagens relevantes com o recuperador. Em seguida, processa múltiplas passagens em paralelo: para cada passagem, prevê o token IsREL, gera o segmento de resposta correspondente, e prevê os tokens IsSUP e IsUSE. Por fim, ranqueia os segmentos candidatos com base nos tokens de crítica e escolhe o melhor. Se Retrieve for "no", o modelo gera o segmento diretamente e depois prevê IsUSE.

Como os tokens de reflexão são gerados na fase de inferência, o comportamento do modelo se torna controlável em tempo de execução. É possível ajustar a frequência de recuperação para diferentes aplicações e customizar o comportamento por preferências, usando a soma linear ponderada das probabilidades dos tokens de reflexão como score de segmento em uma busca em feixe.

## Treinamento

O treinamento usa dois modelos. Um modelo crítico é supervisionado por dados de tokens de reflexão coletados ao consultar um modelo proprietário, o GPT-4. Usando o modelo crítico, o corpus de treinamento é atualizado offline com tokens de reflexão inseridos no texto. Depois, o modelo gerador final é treinado com o objetivo convencional de modelagem de linguagem, para que aprenda a gerar os tokens de reflexão sozinho, sem depender do crítico na inferência. Isso elimina a sobrecarga do crítico em tempo de execução.

## Resultados

O Self-RAG com 7 e 13 bilhões de parâmetros supera significativamente modelos maiores e modelos aumentados por recuperação em um conjunto diverso de tarefas. Supera o ChatGPT e o Llama2-chat com recuperação em perguntas e respostas de domínio aberto, raciocínio e verificação de fatos, com ganhos expressivos em factualidade e em acurácia de citação para gerações longas.
