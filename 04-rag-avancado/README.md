# Módulo 04 — RAG Avançado

Sete técnicas aplicadas sobre o pipeline do [módulo 03](../03-rag-basico), cada uma ligável e desligável por flag, para que sua contribuição seja **medida isoladamente**.

O requisito do módulo era "pelo menos duas melhorias sobre o RAG anterior". Aqui estão sete, todas implementadas do zero — incluindo BM25, Reciprocal Rank Fusion e k-means.

## As técnicas

| # | Técnica | O que faz | Artigo |
|---|---|---|---|
| 1 | **Busca híbrida** | BM25 + denso fundidos por Reciprocal Rank Fusion | BEIR (Thakur et al., 2021) |
| 2 | **HyDE** | Gera um documento hipotético e busca com o embedding dele | Gao et al. (2022) |
| 3 | **Reranking** | LLM reordena os candidatos (pointwise ou listwise) | BEIR |
| 4 | **Reescrita de consultas** | Multi-query, step-back e decomposição | — |
| 5 | **Self-RAG** | Tokens `Retrieve`, `IsREL`, `IsSUP`, `IsUSE` | Asai et al. (2023) |
| 6 | **Ordenação de contexto** | Passagens mais relevantes nas bordas do prompt | Liu et al. (2023) |
| 7 | **RAPTOR** | Árvore de resumos recursivos por clusterização | Sarthi et al. (2024) |

## O fluxo

```
pergunta
  │
  ├─[Retrieve]──── precisa recuperar? ──── não ──> resposta direta, sem busca
  │ sim
  ├─[reescrita]─── multi-query / step-back
  ├─[HyDE]──────── documento hipotético -> vetor de consulta
  │
  ├─[busca densa]──┐
  │                ├─[RRF]─> 20 candidatos
  ├─[busca BM25]───┘
  │
  ├─[IsREL]─────── descarta passagens irrelevantes
  ├─[rerank]────── LLM reordena e corta para top-4
  ├─[ordenação]─── relevantes nas bordas do prompt
  │
  ├─[geração]───── resposta com citações
  └─[IsSUP/IsUSE]─ auto-crítica e ressalva
```

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `bm25.py` | BM25 Okapi do zero: IDF, saturação por `k1`, normalização por `b`, stopwords em português |
| `fusao.py` | Reciprocal Rank Fusion e soma ponderada com normalização min-max |
| `hyde.py` | Documentos hipotéticos e a agregação da Eq. 8 do artigo |
| `query_rewriting.py` | Multi-query, step-back e decomposição |
| `reranker.py` | Reranking pointwise, listwise e o filtro `IsREL` |
| `context_ordering.py` | Quatro estratégias de ordenação contra o Lost in the Middle |
| `self_rag.py` | Tokens de reflexão por prompting |
| `raptor.py` | Clusterização k-means + sumarização recursiva em árvore |
| `advanced_pipeline.py` | Orquestra tudo, com `Config` por flags |
| `main.py` | CLI com presets e ablação |
| `testes_offline.py` | 66 verificações sem API e sem custo |
| `testes_mock.py` | 72 verificações das chamadas de API, com LLM simulado |

Os arquivos de núcleo (`chunking.py`, `embeddings.py`, `vector_store.py`, `ingest.py`, `generator.py`, `evaluate.py`) são os mesmos do módulo 03, copiados para que cada módulo rode sozinho.

## Uso

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # preencha OPENROUTER_API_KEY

python testes_offline.py                          # lógica, sem API
python testes_mock.py                             # chamadas de API, com LLM simulado
python main.py indexar                            # denso + BM25
python main.py bm25 "sua pergunta"                # inspeciona o lexical (grátis)
python main.py comparar --rapido                  # ablação sem custo
python main.py perguntar "sua pergunta" --preset completo --verboso
```

### Presets

| Preset | Técnicas ativas |
|---|---|
| `basico` | só busca densa (equivale ao módulo 03) |
| `hibrido` | denso + BM25 por RRF |
| `hyde` | híbrido + HyDE |
| `rerank` | híbrido + reranking |
| `completo` | tudo |

## Resultados de referência

Ablação da **etapa de recuperação**, 24 consultas, top-k=4, embedder offline:

| Configuração | Hit@4 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|
| denso puro | 79,2% | 75,0% | 0,622 | 0,650 |
| + BM25 híbrido (RRF) | **100,0%** | **95,8%** | **0,858** | **0,877** |

O salto é grande porque o embedder offline é fraco em semântica e o BM25 cobre exatamente esse ponto cego. **Com embeddings reais o baseline sobe e o ganho diminui** — reproduza com sua chave e reporte os seus números.

Rode `python main.py comparar` (sem `--rapido`) para medir também reescrita, HyDE e reranking. Custo estimado: US$ 0,15 a 0,40.

## Notas de implementação

**Self-RAG por prompting, não por treinamento.** O artigo original *treina* um modelo para emitir tokens de reflexão como parte do vocabulário, usando dados anotados por um modelo crítico. Aqui os mesmos sinais são obtidos por prompting de um LLM de instrução. O comportamento é análogo, a mecânica é diferente, e os resultados **não são comparáveis aos do artigo**.

**RAPTOR simplificado.** O artigo usa UMAP + Gaussian Mixture Models com clusterização suave (um nó pode pertencer a vários clusters) e seleção do número de clusters por BIC. Aqui há k-means com clusterização rígida, implementado no próprio arquivo, sem `scikit-learn`. O comportamento qualitativo, abstração crescente camada a camada é preservado.

**Por que RRF em vez de somar scores.** O BM25 produz scores não normalizados que podem ir de 0 a valores arbitrariamente altos; o cosseno vive entre -1 e 1. O RRF ignora a magnitude e funde por posição, o que dispensa calibração. A soma ponderada também está implementada, em `fusao.py`, para comparação.

**O reranking só ajuda se houver candidatos.** O ganho vem de recuperar 20 e cortar para 4, assim o sistema pode corrigir erros de ordenação do recuperador. Recuperar 4 e reranquear 4 não corrige nada. É por isso que `CANDIDATOS=20` e `TOP_K=4` são valores diferentes.

**Falhas não derrubam o pipeline.** Se o HyDE não gerar, usa-se a consulta original. Se o reranking falhar numa passagem, ela recebe nota neutra. Se o `IsREL` descartar tudo, todos os candidatos são mantidos, um contexto ruidoso é melhor que um contexto vazio.

## O que observar

1. **Onde o BM25 ganha do denso.** Use `python main.py bm25 "..."` em perguntas com termos raros (siglas, números, nomes próprios). É o ponto cego dos embeddings.
2. **O `Retrieve` pulando a busca.** `python main.py perguntar "Escreva um haicai sobre o mar." --preset completo` deve mostrar `[Retrieve=no]`.
3. **O documento hipotético do HyDE.** Com `--verboso`, compare o texto gerado com as passagens que ele recuperou. Ele costuma estar factualmente errado e ainda assim recuperar o certo — é exatamente a tese do artigo.
4. **A reordenação do contexto.** Com `--verboso`, o mapa `posição no prompt <- posição no ranking` mostra a mais relevante indo para a primeira posição e a segunda para a última.
5. **O custo de cada técnica.** Os tempos por etapa saem em toda resposta. Reranking pointwise sobre 20 candidatos são 20 chamadas ao LLM — o ganho precisa justificar isso.

## Reprodução

O roteiro completo, com custos e saídas esperadas, está em [REPRODUTIBILIDADE.md](../REPRODUTIBILIDADE.md).
