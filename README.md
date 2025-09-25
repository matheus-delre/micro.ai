# Micro.AI — Local RAG + Tiny LLM (C# + TorchSharp + RocksDB)

> **TL;DR**: This project ships a complete retrieval-and-generation stack that runs 100% locally:
> - **Custom BPE tokenizer** (trained and persisted in RocksDB)
> - **Trainable embedder** (contrastive training in TorchSharp)
> - **Tiny GPT** (decoder‑only Transformer) trained from scratch (self-supervised) and fine‑tuned (SFT)
> - **RAG pipeline** (lexical + vector) with lightweight endpoints

---

## Features

- **BPE tokenizer (from scratch)**  
  Pair-merge training with persistence in RocksDB. Encode/Decode aligned with merges and special tokens.

- **Embedding model (TorchSharp)**  
  Simple contrastive training (positive vs. rolled negative) on `nn.Embedding`; vectors are used by the retriever.

- **TinyGPT (TorchSharp)**  
  - Decoder‑only blocks with LayerNorm → MHA (Q/K/V projections, causal attention with proper triangular mask) → MLP (GELU) + skips
  - **Weight tying**: output head shares weights with token embedding
  - **Curriculum**: grow sequence length from `CurriculumStartSeq` up to `MaxSeq`
  - **Warmup + cosine LR decay**
  - **Top‑k → Top‑p (nucleus)** sampling at generation time; **ban special tokens**

- **RocksDB persistence**  
  Column families: `default,bpe,docs,postings,vectors,nn,meta`. Snapshots/checkpoints stored as raw floats + metadata.

- **Endpoints**  
  - BPE: train/inspect state
  - RAG: index documents, ask questions
  - Torch/Embeddings: encode text, train contrastive, rebuild index vectors
  - LLM: self‑supervised pretrain, SFT (QA), generate, save/load

---

## Requirements

- .NET 8+
- TorchSharp
- RocksDB native binaries (via NuGet `RocksDb.Native`, etc.)
- CPU by default; CUDA optional if available

---

## Configuration

The app reads options from `appsettings.json` under the `Options` root key.

```json
{
  "Serilog": {
    "MinimumLevel": {
      "Default": "Information",
      "Override": {
        "Microsoft": "Warning",
        "System": "Warning"
      }
    },
    "WriteTo": [{ "Name": "Console" }],
    "Enrich": ["FromLogContext"]
  },
  "Options": {
    "Tokenizer": {
      "Lowercase": true,
      "EncodeCacheSize": 2048,
      "DecodeCacheSize": 2048,
      "GarbledPolicy": "Replace",
      "GarbledThreshold": 0.02,
      "MinPairCount": 2,
      "Normalize": "NFKC",
      "SpecialTokens": {
        "PAD": 0, "BOS": 1, "EOS": 2, "MASK": 3, "SEP": 4, "UNK": 5, "GARBLE": 6
      }
    },
    "Pipeline": {
      "DefaultTopK": 5,
      "MaxAnswerTokens": 512,
      "LexWeight": 0.4,
      "VecWeight": 0.6
    },
    "Storage": {
      "Provider": "Rocks",
      "BasePath": "state",
      "BpeStateFile": "bpe_state.json"
    },
    "Torch": {
      "Device": "cpu",
      "Seed": 42,
      "Dim": 256,
      "Epochs": 2,
      "BatchSize": 16,
      "NegativesPerPositive": 2
    },
    "LlmOptions": {
      "VocabSize": 0,
      "Dim": 256,
      "Heads": 8,
      "Layers": 4,
      "MaxSeq": 128,
      "Dropout": 0.0,
      "Seed": 42,
      "Epochs": 200,
      "BatchSize": 8,
      "LearningRate": 0.0003,
      "WeightDecay": 0.0,
      "GradClip": 1.0,
      "CurriculumStartSeq": 64,
      "WarmupSteps": 200,
      "MinLearningRate": 1e-5,
      "TopP": 0.0,
      "DefaultMaxNewTokens": 64,
      "Temperature": 0.9,
      "TopK": 0,
      "Device": "cpu"
    },
    "RocksDb": {
      "BasePath": "state/rocks",
      "EnableStatistics": true,
      "BlockCacheMB": 512,
      "MaxOpenFiles": 512,
      "UseBloomFilter": true
    }
  }
}
```

> **Notes**
> - `Options.LlmOptions.VocabSize=0` makes the LLM adopt the tokenizer’s current vocabulary size.
> - Switch `"Device": "cpu"` to `"cuda"` where supported by your environment.

---

## Run

```bash
dotnet run -c Release
# Swagger UI: https://localhost:7054/swagger
```

---

## API Overview

### BPE
- `POST /bpe/vocabulary` – Train merges on a corpus and persist to RocksDB
- `POST /bpe/vocabulary/save` – Save the vocabulary 
- `POST /bpe/vocabulary/load` – Load the vocabulary
- `GET  /bpe/vocabulary/state` – Current vocab/merges/specials metrics

### RAG
- `POST /rag/index/corpus` – Index a list of raw strings (`doc:1`, `doc:2`, ...)
- `POST /rag/ask` – Retrieve → rerank (light) → LLM + contexts

### Torch / Embeddings
- `POST /torch/train` – Train contrastive on pairs (A,B,Label)
- `POST /torch/save` – Save embedder checkpoint
- `POST /torch/rebuild` – Recompute all vectors in the index

### LLM
- `POST /llm/train/self` – Self‑supervised training on token stream
- `POST /llm/train/sft` – Supervised fine‑tuning on prompt→answer pairs (loss masked to the answer span)
- `POST /llm/generate` – Generation with temperature, top‑k, top‑p, banned tokens
- `POST /llm/save` – Save LLM checkpoint

---

## End‑to‑End Tutorial (Pokémon)

Below we’ll build a tiny Pokémon QA assistant:

### 1) Train BPE on a Pokémon corpus

```bash
curl -X POST https://localhost:7054/bpe/vocabulary \
 -H "Content-Type: application/json" \
 -d '{
  "Corpus": [
    "Pikachu evolves from Pichu and into Raichu.",
    "Bulbasaur is a Grass/Poison-type Pokémon.",
    "Charmander evolves into Charmeleon and then Charizard.",
    "Charizard is Fire/Flying, weak to Rock, Water, and Electric moves.",
    "Squirtle is a Water-type; it evolves to Wartortle and Blastoise."
  ],
  "NumMerges": 800
}'
```

Save state:

```bash
curl -X GET https://localhost:7054/bpe/vocabulary/save
```

### 2) Index your Pokémon knowledge

```bash
curl -X POST https://localhost:7054/rag/index/corpus \
 -H "Content-Type: application/json" \
 -d '{
  "Corpus": [
    "Pikachu evolves from Pichu and into Raichu.",
    "Bulbasaur is a Grass/Poison-type Pokémon.",
    "Charizard is Fire/Flying, weak to Rock, Water, and Electric moves.",
    "Squirtle is a Water-type; evolves to Wartortle and Blastoise.",
    "Electric moves are super effective against Water and Flying."
  ],
  "prefix":"poke:"
}'
```

### 3) Train the embedder (contrastive)

```bash
curl -X POST https://localhost:7054/torch/train \
 -H "Content-Type: application/json" \
 -d '{
  "Pairs":[
    { "A":"What does Pikachu evolve into?",
      "B":"Pikachu evolves from Pichu and into Raichu.", "Label": 1 },
    { "A":"Which types is Charizard weak to?",
      "B":"Charizard is weak to Rock, Water, and Electric.", "Label": 1 },
    { "A":"What type is Bulbasaur?",
      "B":"Bulbasaur is a Grass/Poison-type Pokémon.", "Label": 1 }
  ]
}'
```

Optionally rebuild vectors for the whole index (if content changed significantly):

```bash
curl -X POST https://localhost:7054/torch/rebuild
```

### 4) Pretrain the Tiny LLM (self‑supervised)

Use the same domain corpus to pretrain language modeling:

```bash
curl -X POST https://localhost:7054/llm/train/self \
 -H "Content-Type: application/json" \
 -d '{
  "Corpus": [
    "Pikachu evolves from Pichu and into Raichu.",
    "Bulbasaur is a Grass/Poison-type Pokémon.",
    "Charizard is Fire/Flying, weak to Rock, Water, and Electric moves.",
    "Squirtle evolves to Wartortle and Blastoise.",
    "Electric moves are super effective against Water and Flying."
  ]
}'
```

### 5) SFT: teach prompt→answer patterns (optional but recommended)

```bash
curl -X POST https://localhost:7054/llm/train/sft \
 -H "Content-Type: application/json" \
 -d '{
  "pairs":[
    { "prompt":"question: What does Pikachu evolve into?\nanswer:",
      "answer":"Pikachu evolves into Raichu (after Pichu evolves into Pikachu)." },
    { "prompt":"question: Which types are super effective against Charizard?\nanswer:",
      "answer":"Rock and Water are super effective; Electric is effective due to Flying." }
  ],
  "epochs": 200
}'
```

### 6) Generate

```bash
curl -X POST https://localhost:7054/llm/generate \
 -H "Content-Type: application/json" \
 -d '{
  "Prompt": "question: Which types are super effective against Charizard?",
  "MaxNewTokens": 48,
  "Temperature": 0.7,
  "TopK": 50,
  "TopP": 0.9
}'
```

### 7) Ask via RAG

```bash
curl -X POST https://localhost:7054/rag/ask \
 -H "Content-Type: application/json" \
 -d '{ "Query": "What does Pikachu evolve into?" }'
```

This returns a lightweight composed answer and the top retrieved contexts.

---

## How it works (internals)

- **Tokenizer (BPE)**: trains merges per word (no merging across whitespace). Special tokens defined in config. State serialized to RocksDB (`bpe:state`).

- **Embedder**: `nn.Embedding` → mean‑pooling → L2‑norm. Contrastive loss via cosine similarity of A vs. B and a rolled negative. Checkpoints saved under CF `nn` (`nn:torch:weights`, `nn:torch:meta`).

- **TinyGPT**:
  - Token & positional embeddings → N × blocks (MHA + MLP) → LayerNorm → logits.
  - **Weight tying**: logits computed with the token embedding weights (sample efficiency).
  - **Causal mask**: `triu(ones(T,T), 1)` prevents attending to future tokens.
  - **Training**: Negative log‑likelihood over next-token prediction; curriculum grows T; LR uses warmup + cosine. Gradient clipping available.
  - **Generation**: temperature scaling → top‑k shortlist → top‑p nucleus cut → sample; special tokens can be banned from output.

- **RocksDB**: column families are created/opened at startup. Checkpoints use simple flat float arrays + shapes; state/metadata kept in `meta` CF.

---

## Tips & Troubleshooting

- Always **train BPE** before indexing, embedding training, and LLM pretraining.
- If LLM outputs look off, perform **SFT** with several Pokémon QA pairs.
- Changing model shape (`Dim/Heads/Layers/MaxSeq`) invalidates old LLM checkpoints (loader will skip incompatible snapshots).
- CPU is fine for small demos; CUDA speeds up training significantly.

---

## Roadmap

- Abstractive composer for RAG (LLM summarizes retrieved contexts)
- Repetition penalty / no‑repeat n‑gram for generation
- Quantization for CPU; FP16/BF16 for CUDA
- Alternative norms (RMSNorm), RoPE, SwiGLU blocks
- Proper eval set + early stopping

---

## License

MIT. This repository is for educational and experimental use.
