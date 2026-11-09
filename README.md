# Chatbot sobre PDF com RAG e LangChain

Recentemente, auditei o curso "Mastering Generative AI Project: RAG and LangChain App" da IBM, no [edX](https://www.edx.org/). O projeto final consistiu em um chatbot usando RAG e Langchain sobre um documento PDF em uma interface Gradio servida na nuvem (laboratório virtual).

Aqui, refiz o projeto com pacotes mais atualizados, na minha própria máquina, sem o uso de cloud.

## O esqueleto do projeto (v1.0):

1. A interface web é construída com Gradio e servida localmente (localhost).
2. O PDF (livre) é carregado, dividido em trechos (chunks), transformados em vetores (embeddings) e armazenados no banco de dados vetorial (Chroma).
3. A pergunta do usuário (query) também é transformada em vetor e os *top k* vetores mais parecidos (similaridade) são recuperados.
4. O LLM recebe a pergunta e os *k* trechos com instruções explícitas para usá-los na geração da resposta e informar quando a resposta não está nos trechos.
5. A resposta gerada é apresentada, bem como os trechos recuperados.
6. Telemetria: tempo de recuperação, tempo de geração e total são exibidos (adicionei posteriormente, pra avaliar performance).

## Adaptações em relação ao projeto do curso:

| Componente | No curso | Neste projeto |
| --- | --- | --- |
| LLM | `ibm/granite-4-h-small` via watsonx | `granite4:micro` via Ollama |
| Embeddings | `ibm/granite-embedding-278m-multilingual` via watsonx | `ibm/granite-embedding-278m-multilingual` via Hugging Face |
| Banco de dados vetorial | Chroma em memória | Chroma persistido em disco, uma coleção por PDF |
| Orquestração | `RetrievalQA` (LangChain) | Recuperação + prompt explícito (LangChain Core) |

A classe `RetrievalQA` é obsoleta, sendo chamada apenas com LangChain Classic (legado). Essa foi a maior adaptação feita. Neste projeto, escolhi fazer recuperação e prompt separados (sem chains) inicialmente, deixando para um momento posterior investigar uma implementação mais atualizada.

## O que foi revisado:

### ChromaDB:
1. No curso, o PDF era reindexado a cada pergunta: a função que carregava o PDF, criava chunks e os transformava em vetores era chamada dentro da função que fazia a pergunta - enviando os chunks similares - e gerava a resposta, configurada no Gradio como *fn*.
2. O ChromaDB não era persistido, mas cada pergunta - seja sobre um PDF novo ou um que já tinha sido carregado antes - na mesma sessão do Gradio gerava novos vetores.
3. Todos os chunks - de um documento ou de vários - eram adicionados com o mesmo nome de coleção.
4. O botão 'Clear' do Gradio apenas limpava os campos do lado do front-end, sem afetar a base vetorial (o número de vetores seguiria acumulando até o servidor ser reiniciado).

A solução foi persistir os vetores em disco, usar como índice o hash do conteúdo do PDF e criar uma função *fn* para o Gradio que reaproveitasse os vetores caso já estivessem indexados.

### Outros ajustes:
Ajuste de limites de tokens, exibição de trechos da referência ativada, redução da temperatura para respostas com mais fidelidade e instrução no prompt para indentificar a ausência de informação relevante (off-topic).

## Como rodar:

**Pré-requisitos:** Python 3.10+ e o [Ollama](https://ollama.com/download) instalado.

```bash
# 1. Baixe o modelo de linguagem
ollama pull granite4:micro

# 2. Crie um ambiente virtual e instale as dependências
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Rode a aplicação
python app.py
```

Abra http://127.0.0.1:7860, envie um PDF e faça uma pergunta. Na primeira execução, o modelo de embedding (cerca de 1 GB) é baixado do Hugging Face; depois fica em cache.

## Notebook passo a passo:

O notebook `notebooks/rag_passo_a_passo_v1.ipynb` segue o mesmo pipeline, para estudo e experimentação.

## Variáveis do projeto:

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `LLM_MODEL` | `granite4:micro` | Qualquer modelo baixado no Ollama (ex.: `granite4:tiny-h`, `qwen2.5:3b`) |
| `EMBEDDING_MODEL` | `ibm-granite/granite-embedding-278m-multilingual` | Modelo de embedding do Hugging Face (ex.: `BAAI/bge-m3`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Tamanho e sobreposição dos trechos, em caracteres |
| `TOP_K` | `4` | Quantos trechos são recuperados por pergunta |
| `TEMPERATURE` | `0.1` | Criatividade do LLM |
| `MAX_TOKENS` | `350` | Limite de tokens gerados por resposta (evita respostas longas demais em CPU) |
| `INDEX_DIR` | `.chroma` | Onde os índices ficam salvos |

## Estrutura:

```
app.py                                  # interface Gradio (local, Ollama)
rag.py                                  # pipeline: carga, chunking, embeddings, Chroma, prompt, resposta
requirements.txt                        # dependências da versão local
notebooks/rag_passo_a_passo_v1.ipynb   # o pipeline local, etapa por etapa, com anotações
```

## Limitações:

- Sem suporte para PDFs complexos (escaneados, por exemplo).
- Chunking por caracteres (é possível usar outra estratégia, como por estrutura do documento).
- Sem avaliação da qualidade das respostas.
- Modelos pequenos (como o `granite4:micro`) podem ter limitações (ainda não avaliadas).

## Próximos passos:

1. Versão com uso de modelos usando APIs (ex. Groq).
2. Versão com orquestração nativa LangChain (ex. LangGraph).
3. Explorar outras estratégias de chunking.
4. Incluir mais telemetria.

v1.0 - 2026-11-08