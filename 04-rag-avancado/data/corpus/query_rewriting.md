# Reescrita e expansão de consultas

A consulta do usuário raramente está na forma ideal para recuperação. Pode ser curta demais, ambígua, usar um vocabulário diferente do corpus, ou conter várias perguntas embutidas. A reescrita de consultas transforma a pergunta original antes da busca.

## Multi-query

A técnica de multi-query pede ao LLM que gere várias reformulações da pergunta original, cada uma com vocabulário e enfoque diferentes. Todas são usadas para buscar, e os resultados são fundidos, tipicamente por Reciprocal Rank Fusion.

A intuição é cobrir a lacuna de vocabulário por força bruta: se o corpus usa "aprendizado de máquina" e o usuário escreveu "machine learning", uma das reformulações provavelmente acertará o termo. Cada reformulação amostra uma região diferente do espaço de embeddings, e a fusão recompensa documentos que aparecem de forma consistente em várias delas.

## Step-back prompting

A técnica de step-back pede ao modelo que formule uma pergunta mais geral e abstrata a partir da pergunta específica. Por exemplo, de "qual o valor de k1 usado no BM25 neste artigo?" para "como funciona o BM25?".

A pergunta abstrata recupera o contexto conceitual de fundo, enquanto a pergunta original recupera o detalhe específico. Combinar as duas recuperações dá ao gerador tanto o princípio quanto o dado pontual, o que ajuda especialmente em perguntas que exigem raciocínio.

## Decomposição

Perguntas compostas, que exigem múltiplos saltos de raciocínio, se beneficiam de decomposição em subperguntas independentes. A pergunta "o DPR usa a mesma divisão de passagens do RAG?" vira duas: "como o DPR divide as passagens?" e "como o RAG divide as passagens?". Cada subpergunta é recuperada separadamente e os contextos são unidos.

## Descontextualização

Em sistemas conversacionais, a pergunta atual frequentemente depende do histórico: "e quanto ao segundo?" não significa nada isolada. A descontextualização reescreve a pergunta em uma forma autocontida, usando o histórico da conversa, antes de enviá-la ao recuperador. Sem esse passo, o embedding da pergunta é essencialmente ruído.

## O custo

Toda reescrita adiciona pelo menos uma chamada ao LLM antes da busca, aumentando latência e custo. Em produção, vale medir se o ganho de revocação compensa. Uma prática comum é aplicar reescrita apenas quando a primeira busca retorna scores baixos, o que indica que a consulta provavelmente está mal formulada.
