# BLIS RAG — Estudos de Retrieval-Augmented Generation

Implementações práticas e documentadas de **RAG (Retrieval-Augmented Generation)**, do pipeline básico às técnicas avançadas, cada uma fundamentada em um artigo da área. Os dois módulos são autocontidos, rodam via CLI e usam a **API da OpenRouter**.

Todo o código foi escrito do zero, sem frameworks de RAG (não há LangChain nem LlamaIndex): BM25, fusão RRF, k-means e o índice vetorial estão implementados à mão, porque o objetivo é **entender o mecanismo**, não montar o pipeline mais rápido.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de arquiteturas de recuperação e geração. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os métodos publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada técnica são **citados ao longo de todo o trabalho**, nos docstrings dos módulos, nos comentários das decisões de projeto e na lista de referências ao final. Os PDFs **não** estão incluídos no repositório por questão de direitos autorais. Use as referências para localizá-los nas fontes originais.

## Módulos

| Módulo | Tema | Artigos de referência |
|--------|------|----------------------|
| [`03-rag-basico`](./03-rag-basico) | Ingestão, chunking, embeddings, armazenamento vetorial e resposta com contexto | RAG (Lewis et al., 2020) · DPR (Karpukhin et al., 2020) · MTEB (Muennighoff et al., 2023) · Lost in the Middle (Liu et al., 2023) |
| [`04-rag-avancado`](./04-rag-avancado) | Busca híbrida BM25+vetor, HyDE, reranking, reescrita de consultas, Self-RAG e RAPTOR | Self-RAG (Asai et al., 2023) · HyDE (Gao et al., 2022) · RAPTOR (Sarthi et al., 2024) · BEIR (Thakur et al., 2021) |

## O que cada módulo faz

### Módulo 03 — RAG Básico

Pipeline completo, do documento à resposta:

```
documentos → chunking → embeddings → índice vetorial
pergunta → embedding → busca top-k → prompt com contexto → resposta com citações
```

Inclui três estratégias de chunking comparáveis entre si (fixo, por frases, recursivo), índice vetorial com busca exata por cosseno, cache de embeddings em disco e avaliação do retriever com Hit@k, Recall@k, MRR e nDCG.

### Módulo 04 — RAG Avançado

Sete técnicas sobre o pipeline básico, cada uma ligável/desligável por flag para medir sua contribuição isoladamente:

| Técnica | O que faz | Artigo |
|---------|-----------|--------|
| **Busca híbrida** | BM25 + denso fundidos por Reciprocal Rank Fusion | BEIR |
| **HyDE** | Gera um documento hipotético e busca com o embedding dele | Gao et al. (2022) |
| **Reranking** | LLM reordena os candidatos (pointwise ou listwise) | BEIR |
| **Reescrita de consultas** | Multi-query, step-back e decomposição | — |
| **Self-RAG** | Tokens de reflexão `Retrieve`, `IsREL`, `IsSUP`, `IsUSE` | Asai et al. (2023) |
| **Ordenação de contexto** | Passagens mais relevantes nas bordas do prompt | Liu et al. (2023) |
| **RAPTOR** | Árvore de resumos recursivos por clusterização | Sarthi et al. (2024) |

## Início rápido

```bash
cd 03-rag-basico                     # ou 04-rag-avancado

python -m venv venv
source venv/bin/activate              # Linux/macOS
# venv\Scripts\activate               # Windows

pip install -r requirements.txt
cp .env.example .env                  # preencha OPENROUTER_API_KEY

python testes_offline.py              # valida a lógica sem gastar créditos
python testes_mock.py                 # valida as chamadas de API (LLM simulado)
python main.py indexar
python main.py perguntar "Qual a diferença entre RAG-Sequence e RAG-Token?"
```

O passo a passo completo, com os resultados esperados de cada comando e a estimativa de custo, está em **[REPRODUTIBILIDADE.md](./REPRODUTIBILIDADE.md)**.

## Configuração da API

Os dois módulos usam a [OpenRouter](https://openrouter.ai/) como gateway, tanto para geração quanto para embeddings. Ambos os endpoints são compatíveis com o SDK da OpenAI:

```
OPENROUTER_API_KEY=sua_chave_aqui
MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_PROVIDER=openrouter
```

> Existe um **modo offline** (`EMBEDDING_PROVIDER=local`) com um embedder determinístico por hashing, sem rede e sem custo. Serve apenas para validar o encanamento do pipeline, a qualidade semântica é baixa, porque é um saco-de-n-gramas projetado, não um modelo treinado.

## Corpus de exemplo

Os dois módulos vêm com um corpus de 14 documentos em português (`data/corpus/`) sobre os próprios artigos estudados, e um conjunto de 24 perguntas com documentos relevantes anotados (`data/qa_eval.json`) para medir o retriever. Basta substituir a pasta para usar seus próprios documentos.

Os documentos do corpus são **resumos autorais em português**, escritos a partir da leitura dos artigos. Não são traduções nem reproduções dos textos originais.

## Resultado de exemplo

Ablação do módulo 04, medindo só a etapa de recuperação (24 consultas, top-k=4, embedder offline):

| Configuração | Hit@4 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|
| denso puro | 79,2% | 75,0% | 0,622 | 0,650 |
| + BM25 híbrido (RRF) | **100,0%** | **95,8%** | **0,858** | **0,877** |

O salto é grande porque o embedder offline é fraco em semântica, e o BM25 cobre exatamente esse ponto cego. Com embeddings reais o baseline sobe bastante e o ganho do híbrido fica menor. Reproduza com sua chave e compare.

## Licença

MIT, ver [`LICENSE`](./LICENSE).

A licença cobre o **código deste repositório**. Os artigos citados pertencem a seus respectivos autores e editoras.

## Referências

Todos os artigos abaixo são citados no código, nos pontos em que a técnica correspondente é implementada. Os PDFs não são distribuídos aqui.

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
2. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
3. Muennighoff, N. et al. (2023). *MTEB: Massive Text Embedding Benchmark.* EACL.
4. Liu, N. F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL.
5. Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks.
6. Gao, L. et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL.
7. Asai, A. et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR.
8. Sarthi, P. et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR.
