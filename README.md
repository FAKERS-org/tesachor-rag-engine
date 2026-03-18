# How to run the project

This project is designed to be run in a Unix-like environment. The instructions below assume that you are using a Unix-like shell.

## Step 1: Install dependencies

First, install the dependencies required by the project.

## Step 2: Run the project

After installing the dependencies, you can run the project.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001

uvicorn app.main:app --port 8001
```

Run the following command to start the project:

## Environment Variables

When using a provider endpoint such as Featherless (`/v1/chat/completions`), set a provider key:

```env
HF_LLM_ENDPOINT=https://api.featherless.ai/v1/chat/completions
HF_LLM_PROVIDER=featherless-ai
LLM_MODEL=HuggingFaceH4/zephyr-7b-beta

# Preferred for Featherless/OpenAI-compatible endpoints
FEATHERLESS_API_KEY=your_featherless_api_key

# HF token used for HF-native flows (e.g., embeddings)
HF_API_TOKEN=your_hf_token
```

For Featherless endpoints, `FEATHERLESS_API_KEY` is required.

## Build Vector Store

```bash
uv run python -m app.scripts.build_vectorstore
```

This command will execute the `build_vectorstore.py` script, which processes the dataset and builds the vector store for the RAG engine.

# Note

## QNA

1. Why are there only 5 records in the file specified to be taken to chunking, but then there are Transformed 1433 chunks from conversations?

**Answer:** The reason you see "Transformed 1433 chunks from conversations" when there are only 5 records in your input file is because each record is a conversation thread, and the ConversationTransformer creates a chunk for every assistant response within each conversation.

- Each line in \_part_1.jsonl is a full conversation (not a single Q&A).
  If a conversation has many turns (user/assistant exchanges), each assistant response becomes a chunk.

- For example, if each conversation has 10 assistant responses, 5 conversations would produce up to 50 chunks.

- If your conversations are very long, the number of chunks can be much higher.

So, the chunk count is not equal to the number of records—it depends on how many assistant responses are in each conversation.
