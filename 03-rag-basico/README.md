# Módulo 03 — RAG Básico

Pipeline completo de Retrieval-Augmented Generation, implementado do zero: ingestão de documentos, chunking, embeddings, armazenamento vetorial e geração de resposta com o contexto recuperado.

Sem frameworks de RAG. O índice vetorial, as estratégias de chunking e as métricas de avaliação estão escritos à mão, para que cada etapa seja inspecionável.

## Artigos de referência

| Artigo | O que aparece no código |
|---|---|
| Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | A arquitetura geral recuperar→gerar; chunks de ~100 palavras; o gerador condicionado ao contexto |
| Karpukhin et al. (2020), *Dense Passage Retrieval* | Bi-encoder, similaridade por produto interno, prefixação do título na passagem |
| Muennighoff et al. (2023), *MTEB* | A escolha do modelo de embeddings como decisão a ser validada, não herdada |
| Liu et al. (2023), *Lost in the Middle* | O limite prático do `top_k`: recuperar mais não é recuperar melhor |

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Cliente OpenRouter, modelos e parâmetros, lidos do `.env` |
| `ingest.py` | Carrega `.txt`, `.md` e `.pdf` de um diretório |
| `chunking.py` | Três estratégias: `fixo`, `frases`, `recursivo` |
| `embeddings.py` | Embedder da OpenRouter, embedder local offline e cache em disco |
| `vector_store.py` | Índice plano com busca exata por cosseno e persistência |
| `generator.py` | Prompt de geração com citação obrigatória e abstenção |
| `rag_pipeline.py` | Orquestra indexação e consulta |
| `evaluate.py` | Hit@k, Recall@k, Precision@k, MRR e nDCG |
| `main.py` | CLI |
| `testes_offline.py` | 30 verificações sem API e sem custo |
| `testes_mock.py` | 21 verificações das chamadas de API, com LLM simulado |

## Uso

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # preencha OPENROUTER_API_KEY

python testes_offline.py                      # lógica, sem API
python testes_mock.py                         # chamadas de API, com LLM simulado
python main.py indexar                        # 14 docs -> 103 chunks
python main.py perguntar "sua pergunta"
python main.py chat                           # modo interativo
python main.py avaliar                        # métricas do retriever
python main.py comparar-chunking              # compara as 3 estratégias
python main.py demo                           # roteiro completo
```

## Decisões de projeto

**Busca exata, não aproximada.** O RAG original indexa 21 milhões de passagens com FAISS + HNSW, uma aproximação necessária nessa escala. Aqui, com ~100 chunks, a busca exata roda em milissegundos e serve de referência de acurácia — além de deixar a conta do cosseno visível no código.

**O gerador é obrigado a se abster.** O prompt exige citação por passagem e define uma frase exata para quando o contexto não basta. Um RAG que sempre responde não é melhor que um que às vezes admite não saber é só menos honesto.

**Cache de embeddings.** Reindexar o corpus é a operação mais repetida durante o estudo. O cache em `indice/cache_embeddings.json` evita recalcular e gastar créditos à toa.

**Modo offline.** `EMBEDDING_PROVIDER=local` usa um embedder determinístico de hashing de n-gramas. A qualidade semântica é baixa de propósito —-> serve para validar o encanamento sem API, não para produzir resultados.

## O que observar

1. **A abstenção.** Pergunte algo fora do corpus (`data/perguntas_sem_resposta.json`). O sistema deve recusar, não inventar.
2. **O efeito do chunking.** `comparar-chunking` mostra que não existe estratégia universalmente melhor — depende do corpus.
3. **Recall vs. Precision.** Aumentar o `top_k` melhora o recall e piora a precision. O Lost in the Middle explica por que aumentar indefinidamente não ajuda.
4. **A diferença entre os embedders.** Rode `avaliar` com `EMBEDDING_PROVIDER=local` e depois com `openrouter`. A distância entre os dois é o valor concreto de um bom modelo de embeddings.

## Resultados de referência

Medidos com o embedder **offline**, 24 consultas, top-k=4:

| Métrica | Valor |
|---|---|
| Hit@4 | 79,2% |
| Recall@4 | 75,0% |
| MRR@4 | 0,628 |
| nDCG@4 | 0,653 |

Com `text-embedding-3-small` esses números sobem. Veja [REPRODUTIBILIDADE.md](../REPRODUTIBILIDADE.md) para o roteiro completo.

## Próximo passo

O [módulo 04](../04-rag-avancado) parte deste pipeline e acrescenta busca híbrida, HyDE, reranking, reescrita de consultas, Self-RAG e RAPTOR — cada técnica medida isoladamente.
