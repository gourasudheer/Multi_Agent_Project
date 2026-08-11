# Agentic AI Blog Generator

This project is a production-ready agentic AI workflow based on the `sequential_workflow.ipynb` notebook.
It creates a blog in three steps: outline planning, draft generation, and editorial review.

## Files added
- `agentic_ai_blog.py` — backend workflow and CLI entrypoint.
- `streamlit_app.py` — Streamlit frontend for interactive blog generation.
- `requirements.txt` — Python dependencies.
- `.env.example` — sample environment configuration.

## Setup
1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run from command line

```bash
python agentic_ai_blog.py --prompt "Write a blog about the benefits of AI in business" --show-intermediate
```

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Notes
- Use `OPENAI_API_KEY` for OpenAI service.
- Use `OPENROUTER_API_KEY` and optionally `OPENAI_API_BASE` or `OPENROUTER_API_BASE` for OpenRouter.
- Override the default model with `AGENTIC_AI_MODEL` if needed.
