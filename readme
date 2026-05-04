# 🤖 RAG Document Assistant — Project 1

> A production-grade Retrieval Augmented Generation (RAG) pipeline built with LangChain, Gemini, ChromaDB, FastAPI, and Streamlit. Deployed on AWS EC2 with Nginx, Systemd, SSL, and a free custom domain.

---

## 🌐 Live Demo

| Interface | URL |
|---|---|
| Customer UI | https://ragdocassistant.duckdns.org |
| API Docs | http://YOUR_ELASTIC_IP:8000/docs |

---

## 📌 What This Project Does

- Upload one or more PDF documents
- Ask questions in natural language
- Get grounded answers with **page-level citations**
- Supports **follow-up questions** via conversation memory
- Auto-detects **vague vs specific** questions and routes accordingly
- Caches answers — repeated questions cost **zero API calls**
- Handles greetings and casual chat gracefully

---

## 🏗️ Architecture

```
User
  ↓
https://ragdocassistant.duckdns.org
  ↓
DuckDNS DNS → AWS Elastic IP (fixed)
  ↓
AWS EC2 t3.micro (Ubuntu 24.04)
  ↓
Nginx (port 80/443) + SSL (Let's Encrypt)
  ↓
┌─────────────────┬──────────────────┐
│  Streamlit UI   │   FastAPI API    │
│  (port 8501)    │   (port 8000)    │
└────────┬────────┴────────┬─────────┘
         │                 │
         └────────┬────────┘
                  ↓
         RAG Core (Parts 1-4)
                  ↓
    ChromaDB (EC2 disk) + Gemini API
```

---

## 🧠 RAG Pipeline — How It Works

```
PDF uploaded
    ↓
Part 1 → loader.py     → reads PDF page by page
    ↓
Part 2 → chunker.py    → splits into 1000 char chunks (200 overlap)
    ↓
Part 3 → embedder.py   → converts chunks to vectors (embedding-001)
         registry.py   → tracks file hashes (detects new/edit/delete)
         cache.py      → caches answers to disk
    ↓
ChromaDB               → stores vectors + text + metadata on disk
    ↓
User asks question
    ↓
Part 4 → retriever.py
    Layer 1 → Casual?       → warm reply, no API call
    Layer 2 → Docs loaded?  → friendly nudge if empty
    Layer 3 → Cache hit?    → instant answer, zero cost
    Layer 4 → Intent check
              Vague    → MultiQuery (3 variations × ChromaDB)
              Specific → Normal retriever (top 3 chunks)
    ↓
Memory → rewrites follow-up questions using chat history
    ↓
Gemini → grounded answer from retrieved chunks only
    ↓
Answer + source citations (file name + page number)
```

---

## 📁 Project Structure

```
Project_1/
│
├── core/
│   ├── loader.py          # Part 1: PDF reading
│   ├── chunker.py         # Part 2: Text splitting
│   ├── embedder.py        # Part 3: Embeddings + ChromaDB
│   ├── cache.py           # Part 3: Answer caching
│   ├── registry.py        # Part 3: File hash tracking
│   └── retriever.py       # Part 4: Full retrieval logic
│
├── api/
│   ├── main.py            # FastAPI backend (5 endpoints)
│   └── schemas.py         # Pydantic request/response models
│
├── documents/             # Drop PDFs here (gitignored)
├── chroma_db/             # Vector store (auto created, gitignored)
│
├── app.py                 # Streamlit UI (production)
├── main.py                # Terminal testing tool
├── requirements.txt       # All dependencies
└── .env                   # API keys (never pushed to GitHub)
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google embedding-001 |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Cloud | AWS EC2 t3.micro |
| Process Manager | Systemd |
| Reverse Proxy | Nginx |
| SSL | Let's Encrypt (Certbot) |
| DNS | DuckDNS (free) |
| Fixed IP | AWS Elastic IP |

---

## 🚀 Features

### Smart Retrieval
- **Intent routing** — detects vague vs specific questions automatically
- **MultiQuery retrieval** — generates 3 query variations for broad questions, merges results
- **Normal retrieval** — single precise search for specific questions

### Memory
- **Context-aware query rewriting** — follow-up questions rewritten using chat history
- **Prompt memory** — last 3 turns included in every Gemini prompt

### Document Management
- **Incremental embedding** — only new or changed chunks embedded (saves API cost)
- **MD5 content hashing** — edits auto-detected, old vectors replaced
- **Atomic delete** — removes vectors + cache + file + registry in one operation
- **Multi-document support** — unlimited PDFs, all searchable together

### Cost Optimisation
- **Query cache** — same question answered twice = zero API cost
- **Cache invalidation** — cache cleared automatically when documents change
- **Casual layer** — greetings never touch ChromaDB or Gemini

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — is API alive? |
| GET | `/docs-list` | List all loaded documents |
| POST | `/upload` | Upload and embed a PDF |
| POST | `/query` | Ask a question |
| DELETE | `/document/{filename}` | Remove a document |

### Example Query Request

```json
POST /query
{
  "question": "What is the probation period?",
  "chat_history": [
    {"role": "user", "content": "Tell me about employment"},
    {"role": "assistant", "content": "NexaCore classifies..."}
  ]
}
```

### Example Query Response

```json
{
  "question": "What is the probation period?",
  "answer": "The probation period at NexaCore is 6 months...",
  "sources": [
    {"file": "NexaCore_HR_Policy.pdf", "page": 3}
  ],
  "retriever_type": "normal",
  "from_cache": false
}
```

---

## 🛠️ Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/Project-AI-ML-Learning/Project_1.git
cd Project_1
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` file

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run Streamlit UI

```bash
streamlit run app.py
```

### 6. Run FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

---

## ☁️ AWS Deployment

### Infrastructure Setup (via AWS CLI)

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name rag-project-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/rag-project-key.pem
chmod 400 ~/.ssh/rag-project-key.pem

# Create security group
aws ec2 create-security-group \
    --group-name rag-sg \
    --description "RAG project security group"

# Open ports
aws ec2 authorize-security-group-ingress --group-name rag-sg --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name rag-sg --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name rag-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name rag-sg --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Launch EC2
aws ec2 run-instances \
    --image-id ami-05cf1e9f73fbad2e2 \
    --instance-type t3.micro \
    --key-name rag-project-key \
    --security-groups rag-sg \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-project}]'
```

### EC2 Server Setup

```bash
# SSH in
ssh -i ~/.ssh/rag-project-key.pem ubuntu@YOUR_IP

# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx

# Add swap memory
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Clone and setup
git clone https://github.com/Project-AI-ML-Learning/Project_1.git
cd Project_1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nano .env    # add GOOGLE_API_KEY
```

### Systemd Services

```bash
# API service at /etc/systemd/system/rag-api.service
# UI service at  /etc/systemd/system/rag-ui.service

sudo systemctl daemon-reload
sudo systemctl enable rag-api rag-ui nginx
sudo systemctl start rag-api rag-ui nginx
```

### SSL Certificate

```bash
sudo certbot --nginx -d ragdocassistant.duckdns.org
```

---

## 📊 Cost Profile

| Operation | API Calls | Cost |
|---|---|---|
| Casual message ("Hi") | 0 | $0.00 |
| Cache hit | 0 | $0.00 |
| Specific question | 1 Gemini | ~$0.0006 |
| Vague question | 2 Gemini | ~$0.0012 |
| Follow-up question | 1 extra Gemini | ~$0.0006 |
| Upload PDF (23 chunks) | 23 embedding | ~$0.0005 |
| Upload unchanged PDF | 0 | $0.00 |

**Monthly cost at 100 unique questions/day: ~$2-3**
**With caching at 10 unique/day: ~$0.20**

---

## 🔒 Security

- API keys stored in `.env` — never pushed to GitHub
- `.gitignore` excludes all sensitive files
- HTTPS enforced via Let's Encrypt SSL
- HTTP → HTTPS redirect via Nginx
- File upload restricted to PDF only
- ChromaDB stored locally on EC2 disk

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/sample.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "chat_history": []}'

# Query with memory
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What happens after that?",
    "chat_history": [
      {"role": "user", "content": "What is the probation period?"},
      {"role": "assistant", "content": "6 months from date of joining"}
    ]
  }'
```

---

## 📝 Key Design Decisions

**Why ChromaDB over Pinecone?**
ChromaDB runs locally on disk — zero cost, no external dependency, data never leaves EC2.

**Why manual MultiQuery over LangChain's built-in?**
LangChain's MultiQueryRetriever has version instability. Manual implementation gives full control and is easier to explain in interviews.

**Why MD5 hashing for change detection?**
Deterministic, fast, and collision-resistant enough for document tracking. Any edit to a file produces a completely different hash — triggering automatic re-embedding.

**Why cache at query level not chunk level?**
Query-level caching is simpler and covers the most common case — users asking the same question repeatedly. The entire answer is cached, not just retrieved chunks.

**Why Systemd over Docker?**
Simpler for a single-server deployment. Systemd is built into Ubuntu, requires no additional tooling, and auto-restarts on crash and reboot.

---

## 👤 Author

**Nikhil Kumar R**
Trust & Safety Professional → AI/ML Engineer
- GitHub: [Project-AI-ML-Learning](https://github.com/Project-AI-ML-Learning)
- Background: 6+ years Trust & Safety at YouTube scale
- Focus: LLM Evaluation, AI Safety, Agentic AI

---

## 📄 License

MIT License — free to use, modify, and distribute.