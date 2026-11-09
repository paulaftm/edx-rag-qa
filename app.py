"""Interface Gradio para o RAG local."""

import gradio as gr

from rag import EMBEDDING_MODEL, LLM_MODEL, answer

demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.File(label="PDF", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Pergunta", lines=2, placeholder="Digite sua pergunta..."),
    ],
    outputs=[
        gr.Textbox(label="Resposta"),
        gr.Textbox(label="Trechos recuperados (fontes)", lines=12),
        gr.Number(label="Tempo de recuperação (s)"),
        gr.Number(label="Tempo de geração (s)"),
        gr.Number(label="Tempo total (s)"),
    ],
    title="RAG local com PDF",
    description=(
        "Envie um PDF e faça perguntas. A resposta usa apenas trechos do documento "
        f"e mostra as fontes. LLM: {LLM_MODEL} (Ollama) · Embeddings: {EMBEDDING_MODEL}."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
