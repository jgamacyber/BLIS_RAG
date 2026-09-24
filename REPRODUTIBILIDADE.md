# Documento de Reprodutibilidade

Roteiro para reproduzir todos os resultados deste repositório na sua máquina, com sua chave da OpenRouter. Cada comando traz o que esperar de saída e o custo aproximado.

---

## 1. Ambiente

### Requisitos

- Python **3.10 ou superior** (o código usa `X | None`, sintaxe de união de tipos do 3.10)
- Conexão com a internet para os comandos que chamam a API
- Uma chave da OpenRouter: https://openrouter.ai/keys

### Verificar a versão do Python

```bash
python --version      # deve mostrar 3.10+
```

Se a sua distribuição tiver `python3` em vez de `python`, use `python3` em todos os comandos.

### Instalação

Faça isso **dentro da pasta do módulo** que vai rodar (`03-rag-basico` ou `04-rag-avancado`). Cada módulo é autocontido e tem seu próprio ambiente:

```bash
cd 03-rag-basico

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
```

Dependências instaladas: `openai`, `python-dotenv`, `numpy` e, opcionalmente, `pypdf` (só é necessário se você colocar PDFs no corpus).

### Configuração

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Edite o `.env` e preencha a chave:

```
OPENROUTER_API_KEY=sk-or-v1-...sua_chave...
MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_PROVIDER=openrouter
```

---

## 2. Validação sem custo

Antes de gastar qualquer crédito, rode as duas suítes de teste. Nenhuma delas faz uma única chamada de rede.

### 2.1 Testes offline — a lógica que não depende do LLM

Exercita chunking, BM25, fusão RRF, ordenação de contexto, k-means, índice vetorial e métricas.

```bash
python testes_offline.py
```

**Esperado:** `Todos os testes passaram.` e código de saída 0.

- Módulo 03: 30 verificações
- Módulo 04: 66 verificações

### 2.2 Testes com LLM simulado — as chamadas de API

Substitui o cliente da OpenRouter por um dublê que devolve respostas controladas. Verifica que as chamadas são montadas no formato certo (modelo, mensagens, `temperature`, batching de embeddings), que as respostas são interpretadas corretamente, e que respostas malformadas não derrubam o pipeline.

```bash
python testes_mock.py
```

**Esperado:** `Todos os testes passaram.`

- Módulo 03: 21 verificações
- Módulo 04: 72 verificações, incluindo o pipeline avançado ponta a ponta

> Esta suíte é a que mais economiza token: um erro de digitação em `client.embeddings.create`, ou um parser de JSON frágil, só apareceria em produção — aqui aparece de graça.

Se algo falhar nas duas seções acima, é bug no código ou incompatibilidade de ambiente. Não vale seguir para os comandos pagos.

### Pipeline completo sem gastar créditos

O modo `local` troca a API de embeddings por um embedder determinístico de hashing. A recuperação funciona (com qualidade semântica baixa), mas a **geração ainda exige a chave**, porque não há LLM local.

```bash
EMBEDDING_PROVIDER=local python main.py indexar
EMBEDDING_PROVIDER=local python main.py avaliar          # módulo 03
EMBEDDING_PROVIDER=local python main.py comparar --rapido # módulo 04
```

No Windows (PowerShell), defina a variável antes:

```powershell
$env:EMBEDDING_PROVIDER="local"; python main.py indexar
```

---

## 3. Módulo 03 — RAG Básico

### 3.1 Indexar o corpus

```bash
python main.py indexar
```

**Esperado:** 14 documentos → **103 chunks** (estratégia `recursivo`, `chunk_size=600`, `overlap=100`), média de ~409 caracteres por chunk. O índice é salvo em `indice/vector_store.json`.

**Custo:** 1 chamada de embeddings com 103 entradas (~11 mil tokens). Com `text-embedding-3-small`, algo em torno de **US$ 0,0002**.

> Os embeddings ficam em cache em `indice/cache_embeddings.json`. Reindexar o mesmo corpus não gera custo novo.

### 3.2 Fazer uma pergunta

```bash
python main.py perguntar "Qual a diferença entre RAG-Sequence e RAG-Token?"
```

**Esperado:** 4 passagens recuperadas (a maioria de `rag_lewis`) e uma resposta citando os identificadores, no formato `[rag_lewis#002]`.

**Custo:** 1 embedding da pergunta + 1 chamada de chat (~1.500 tokens de prompt). Menos de **US$ 0,001**.

### 3.3 Verificar a abstenção

```bash
python main.py perguntar "Qual é a capital da Mongólia?"
```

**Esperado:** a resposta deve ser exatamente *"O contexto fornecido não contém informação suficiente para responder."* — o corpus não cobre o assunto.

Este é o teste mais importante do módulo. Se o sistema inventar uma resposta aqui, o prompt de geração não está sendo respeitado. Outras perguntas fora do corpus estão em `data/perguntas_sem_resposta.json`.

### 3.4 Avaliar o retriever

```bash
python main.py avaliar
```

**Esperado:** métricas sobre as 24 consultas de `data/qa_eval.json`.

Referência medida com o embedder **offline** (`EMBEDDING_PROVIDER=local`):

| Métrica | Valor |
|---|---|
| Hit@4 | 79,2% |
| Recall@4 | 75,0% |
| Precision@4 | 27,8% |
| MRR@4 | 0,628 |
| nDCG@4 | 0,653 |

Com `text-embedding-3-small` os números devem ser **claramente melhores** — é justamente esse o ponto do experimento. Anote os seus e compare.

**Custo:** 24 embeddings de pergunta (o índice já está em cache). Desprezível.

### 3.5 Comparar estratégias de chunking

```bash
python main.py comparar-chunking
```

**Esperado:** tabela com `fixo`, `frases` e `recursivo`. Referência offline:

| Estratégia | Chunks | Hit@4 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|---|
| fixo | 103 | 87,5% | 85,4% | 0,622 | 0,679 |
| frases | 88 | 87,5% | 81,2% | 0,715 | 0,716 |
| recursivo | 103 | 79,2% | 75,0% | 0,628 | 0,653 |

> Com este corpus pequeno e homogêneo, o `recursivo` não lidera — resultado legítimo e que vale registrar. O efeito do chunking cresce com o tamanho e a heterogeneidade do corpus. **Não generalize a partir de 14 documentos.**

**Custo:** reindexação com 3 estratégias ≈ 3× o custo de 3.1, ou **~US$ 0,0006**.

### 3.6 Roteiro completo

```bash
python main.py demo
```

Indexa, responde 4 perguntas (3 no corpus, 1 fora) e comenta a abstenção. **Custo:** ~US$ 0,005.

---

## 4. Módulo 04 — RAG Avançado

```bash
cd ../04-rag-avancado
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # preencha a chave
python testes_offline.py           # 66 verificações
python testes_mock.py              # 72 verificações
```

### 4.1 Indexar

```bash
python main.py indexar
```

**Esperado:** 103 chunks indexados + índice lexical BM25 com **1.762 termos** de vocabulário e `avgdl=44,5` tokens.

**Custo:** igual a 3.1 (o BM25 não usa API).

### 4.2 Inspecionar a busca lexical

```bash
python main.py bm25 "Como o HyDE constrói o vetor final da consulta?"
```

**Esperado:** lista de termos com seus IDFs (`vetor` 2,940 · `final` 2,940 · `hyde` 2,504 · `consulta` 1,970 · `constroi` 0,000 — ausente do corpus) e o top-4, liderado por `hyde#006`.

**Custo: zero.** Nenhuma chamada de API.

### 4.3 Ablação — a medição principal

Mede a contribuição de cada técnica na **etapa de recuperação**, acumulando uma sobre a outra.

Versão sem custo (só técnicas que não chamam o LLM):

```bash
python main.py comparar --rapido
```

Referência offline:

| Configuração | Hit@4 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|
| 1. denso puro | 79,2% | 75,0% | 0,622 | 0,650 |
| 2. + BM25 híbrido (RRF) | **100,0%** | **95,8%** | **0,858** | **0,877** |

Versão completa (inclui reescrita, HyDE e reranking — **chama o LLM**):

```bash
python main.py comparar
```

**Custo estimado:** cerca de 24 consultas × (1 reescrita + 1 HyDE + 20 notas de reranking) ≈ **500 chamadas de chat**. Com `gpt-4o-mini`, estime **US$ 0,15 a 0,40**. É o comando mais caro do repositório — rode uma vez e guarde a saída.

> **Sobre o salto para 100%:** ele é grande porque o embedder offline é fraco em semântica, e o BM25 cobre exatamente esse ponto cego. Com embeddings reais o baseline denso sobe muito, e o ganho do híbrido fica bem menor. Reproduza com sua chave: o número que importa para o seu relatório é o seu.

### 4.4 Comparar presets numa pergunta

```bash
python main.py perguntar "Como o HyDE constrói o vetor final da consulta?" --preset basico
python main.py perguntar "Como o HyDE constrói o vetor final da consulta?" --preset hibrido
python main.py perguntar "Como o HyDE constrói o vetor final da consulta?" --preset completo --verboso
```

Presets disponíveis: `basico`, `hibrido`, `hyde`, `rerank`, `completo`.

Com `--verboso`, o preset `completo` mostra as consultas reescritas, o documento hipotético gerado pelo HyDE e o mapa de reposicionamento das passagens no prompt.

**Esperado:** a resposta correta menciona a média de **N+1 vetores** (N documentos hipotéticos + a consulta), conforme a Eq. 8 do artigo.

**Custo:** `basico` ~US$ 0,001; `completo` ~US$ 0,02 (reescrita + HyDE + 20 notas de reranking + geração + crítica).

### 4.5 Verificar o `Retrieve` sob demanda

```bash
python main.py perguntar "Escreva um haicai sobre o mar." --preset completo
```

**Esperado:** o token `Retrieve` decide `no`, o pipeline **pula a recuperação** e responde direto. A saída mostra `[Retrieve=no]` com o motivo.

Este teste valida a contribuição central do Self-RAG: não recuperar quando não é preciso.

### 4.6 Verificar a auto-crítica

Com `--preset completo`, toda resposta passa por `IsSUP` e `IsUSE`. A saída traz um bloco `Auto-reflexão (Self-RAG)`. Quando a resposta não é totalmente sustentada, uma ressalva é anexada ao texto final.

Para provocar uma crítica negativa, pergunte algo parcialmente coberto pelo corpus, por exemplo:

```bash
python main.py perguntar "Quantos parâmetros tem o modelo Self-RAG e qual seu custo de treino?" --preset completo
```

O corpus menciona 7 e 13 bilhões de parâmetros, mas não o custo de treino — espere `partially` em `IsSUP`.

### 4.7 RAPTOR (opcional, mais caro)

```bash
python main.py indexar --raptor
```

**Esperado:** construção da árvore nível a nível. Com 103 folhas e `tamanho_cluster=5`, o nível 1 gera ~20 clusters e o nível 2 gera ~4, totalizando cerca de **127 nós**.

**Custo:** ~24 chamadas de sumarização (~US$ 0,02) + embeddings dos resumos.

Depois de indexar com RAPTOR, perguntas temáticas amplas devem recuperar nós de resumo (identificados como `resumo_n1_XX`):

```bash
python main.py perguntar "Quais são as principais técnicas para melhorar a recuperação em RAG?"
```

---

## 5. Resumo de custos

Valores para `openai/gpt-4o-mini` + `openai/text-embedding-3-small`, em setembro de 2026. Confira os preços atuais em https://openrouter.ai/models.

| Comando | Chamadas | Custo aprox. |
|---|---|---|
| `testes_offline.py` | 0 | **US$ 0** |
| `testes_mock.py` | 0 | **US$ 0** |
| `comparar --rapido` | 0 | **US$ 0** |
| `bm25 "..."` | 0 | **US$ 0** |
| `indexar` | 1 embedding em lote | ~US$ 0,0002 |
| `perguntar` (básico) | 1 emb + 1 chat | <US$ 0,001 |
| `perguntar --preset completo` | ~25 chamadas | ~US$ 0,02 |
| `avaliar` | 24 embeddings | ~US$ 0,0001 |
| `comparar-chunking` | 3 reindexações | ~US$ 0,0006 |
| `demo` (módulo 03) | ~5 chamadas | ~US$ 0,005 |
| `indexar --raptor` | ~24 sumarizações | ~US$ 0,02 |
| `comparar` (completo) | ~500 chamadas | **US$ 0,15 – 0,40** |

Reproduzir tudo, uma vez, fica abaixo de **US$ 0,50**.

---

## 6. Determinismo e variação entre execuções

O que é **determinístico** e deve reproduzir exatamente:

- Chunking, tokenização e scores do BM25
- Fusão RRF e ordenação de contexto
- K-means do RAPTOR (semente fixa, `semente=42`)
- Embedder local de hashing
- Todas as métricas de recuperação, **quando o embedder é o local**

O que **varia** entre execuções:

- Embeddings da API podem mudar entre versões do modelo, alterando ligeiramente as métricas
- Geração de texto: mesmo com `temperature=0.0` o resultado não é bit-a-bit reprodutível em APIs
- **HyDE usa `temperature=0.7` por definição** (é o valor do artigo) — os documentos hipotéticos mudam a cada execução, e portanto a recuperação com HyDE também
- As notas do reranking por LLM podem oscilar em ±1 ponto

Para um relatório, rode as configurações com LLM **3 vezes** e reporte média e desvio, em vez de um número único.

---

## 7. Problemas comuns

**`OPENROUTER_API_KEY não encontrada`**
O `.env` não foi criado ou está na pasta errada. Ele precisa estar **dentro da pasta do módulo**, ao lado de `main.py`.

**`O índice foi criado com 'X', mas o embedder atual é 'Y'`**
Você indexou com um embedder e está consultando com outro (por exemplo, indexou com `local` e consultou com `openrouter`). Vetores de modelos diferentes não são comparáveis. Apague a pasta `indice/` e reindexe.

**`Índice não encontrado`**
Rode `python main.py indexar` antes de consultar.

**Erro 401 ou 402 da OpenRouter**
Chave inválida ou sem créditos. Confira em https://openrouter.ai/credits.

**Erro 404 no modelo de embeddings**
O identificador do modelo mudou ou não está disponível na sua conta. Liste os modelos de embedding em https://openrouter.ai/models?output_modalities=embeddings e ajuste `EMBEDDING_MODEL` no `.env`.

**Rate limit (429)**
O `OpenRouterEmbedder` já repete com backoff exponencial (até 4 tentativas). Para o reranking, reduza `CANDIDATOS` no `.env` de 20 para 10.

**`SyntaxError` com `|` em anotações de tipo**
Python anterior ao 3.10. Atualize o interpretador.

**PDF ignorado no corpus**
`pypdf` não está instalado (`pip install pypdf`) ou o PDF é escaneado, sem camada de texto.

---

## 8. Usar seu próprio corpus

1. Substitua os arquivos de `data/corpus/` pelos seus `.txt`, `.md` ou `.pdf`.
2. Reescreva `data/qa_eval.json` com perguntas do seu domínio. O campo `docs_relevantes` usa o **nome do arquivo sem extensão** como identificador:

```json
[
  {
    "pergunta": "sua pergunta",
    "docs_relevantes": ["nome_do_arquivo_sem_extensao"],
    "resposta_referencia": "opcional, para conferência manual"
  }
]
```

3. Apague a pasta `indice/` e rode `python main.py indexar` de novo.
4. Inclua perguntas **sem resposta no corpus** no seu conjunto de testes. Sem elas, não dá para distinguir um sistema que sabe responder de um que sempre responde.

---

## 9. Checklist de reprodução

- [ ] Python 3.10+ confirmado
- [ ] `venv` criado e ativado **dentro da pasta do módulo**
- [ ] `pip install -r requirements.txt` sem erros
- [ ] `.env` criado a partir do `.env.example`, com a chave preenchida
- [ ] `python testes_offline.py` → todos passam (30 no módulo 03, 66 no 04)
- [ ] `python testes_mock.py` → todos passam (21 no módulo 03, 72 no 04)
- [ ] `python main.py indexar` → 103 chunks
- [ ] `python main.py perguntar "..."` → resposta com citações `[doc#NNN]`
- [ ] Pergunta fora do corpus → abstenção explícita
- [ ] `python main.py avaliar` → métricas anotadas
- [ ] Módulo 04: `comparar --rapido` → híbrido supera o denso puro
- [ ] Módulo 04: `--preset completo` numa pergunta criativa → `[Retrieve=no]`
