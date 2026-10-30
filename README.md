# 🔬 OpenResearch AI — Best Open-Source AI Project

[![Hacktoberfest Hack Day](https://img.shields.io/badge/Hacktoberfest-Best%20Open--Source%20AI%20Project-fe7a36.svg)](https://hacktoberfest.com/)
[![MLH Open-Source AI](https://img.shields.io/badge/MLH-Open--Source%20AI%20Challenge-059669.svg)](https://www.mlh.com/opensource-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Model: Gemma 3 4B](https://img.shields.io/badge/SLM%20(%3C10B)-Google%20Gemma%203%204B-4285F4.svg)](https://ai.google.dev/gemma)
[![Inference: Ollama](https://img.shields.io/badge/Local%20Inference-Ollama-black.svg)](https://ollama.com/)
[![Agent Skills Standard](https://img.shields.io/badge/Agent%20Skills-Compliant-blueviolet.svg)](https://agentskills.io/specification)

**OpenResearch AI** is an open-source, evidence-grounded research assistant built for the **Best Open-Source AI Project** challenge at the **Hacktoberfest Hack Day / MLH Open-Source AI Hackathon**.

It autonomously searches the web for multi-source evidence using the **Tavily Search API**, analyzes findings locally through an open-weight Small Language Model (**Google Gemma 3 4B** via **Ollama**), cross-compares evidence to detect consensus and contradictory claims, and synthesizes a structured, citation-verified research report.

---

## 🏆 Hackathon Challenge Compliance Checklist

| Challenge Requirement | How OpenResearch AI Complies | Reference in Repo |
| :--- | :--- | :--- |
| **Open-Weight / Open-Source AI** | Uses **Google Gemma 3 4B** (open-weight SLM <10B parameters) running locally via **Ollama**. No proprietary hosted LLM lock-in. | [app.py](app.py), [Ollama Library](https://ollama.com/library/gemma3) |
| **Clear AI Role in Demo** | Gemma performs multi-source synthesis, strict grounding, consensus detection, and writes a 6-part report. | See [Live Demo Walkthrough](#-2-minute-live-demo-walkthrough) below |
| **Open Source License** | Licensed under the OSI-approved **MIT License**. | [LICENSE](LICENSE) |
| **Open-Weight License & Terms** | Clearly documents and links to the **Google Gemma Terms of Use** and Model Card. Distinguishes code license from model weights. | See [Licensing & Usage Terms](#-licensing--usage-terms) |
| **Agent Skills Open Standard** | Includes a compliant Agent Skill specification following the open standard. | [skills/openresearch-synthesizer/SKILL.md](skills/openresearch-synthesizer/SKILL.md) |
| **Deployment & Reproducibility** | Complete `requirements.txt`, `.env.example`, `Dockerfile`, and `docker-compose.yml` for **DigitalOcean** or local use. | [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) |

---

## 🌟 What OpenResearch AI Solves

Modern AI research assistants frequently suffer from **hallucinations**, **fabricated citations**, and **single-source bias**. 

**OpenResearch AI** solves this through a deterministic 5-stage pipeline:
1. **🔍 Search**: Dispatches queries to Tavily's advanced search engine.
2. **📥 Collect**: Ingests, normalizes, and scores up to 5 verified web sources (`[1]` to `[5]`).
3. **🧠 Analyze**: Injects verified evidence blocks into local Gemma 3 with strict zero-hallucination constraints.
4. **⚖️ Compare**: Generates a cross-source matrix highlighting undisputed agreements versus conflicting claims (e.g. timelines vs costs).
5. **📑 Report**: Synthesizes a structured 6-section research brief with inline bracketed markdown citations.

---

## 📐 Architecture

```mermaid
flowchart TD
    User([User Research Topic]) --> UI[Streamlit Green-and-White Dashboard]
    UI --> Search[Tavily Search API / Fallback]
    Search --> Filter[Top 5 Evidence Sources with URLs & Snippets]
    Filter --> Ollama[Local Ollama Inference Engine]
    Ollama --> Gemma[Google Gemma 3 4B - Open-Weight SLM]
    Gemma --> Synthesis[Cross-Source Comparison & Conflict Matrix]
    Synthesis --> Report[Structured 6-Section Report with [1]-[5] Citations]
    Report --> Export[Markdown .md Export]
```

---

## 🤖 Model Details: Google Gemma 3 4B (Small Language Model)

- **Model Identifier**: `gemma3:4b` (Instruction-tuned, 4 Billion parameters)
- **Classification**: **Small Language Model (SLM)** — well below the challenge's 10-billion parameter threshold.
- **Why Gemma 3?**: High reasoning benchmark performance per parameter, native support for multilingual research, and lightweight memory footprint (~3.3 GB download, ~4-5 GB RAM on consumer hardware).
- **Ultralight Fallback**: `gemma3:1b` (~1.2 GB download, ~2 GB RAM) for lower-spec machines.
- **Official Model Card**: [Google Gemma 3 Documentation](https://ai.google.dev/gemma/docs/gemma3-report)
- **Model Distribution**: [Ollama Gemma 3 Library](https://ollama.com/library/gemma3)

---

## ⚡ 2-Minute Live Demo Walkthrough (For Judges)

1. **Launch Dashboard**: Open `http://localhost:8501`.
2. **Inspect Pipeline**: Point to the emerald sidebar tracking the 5 autonomous research stages.
3. **Select Quick Topic**: Click the example pill:
   - 🔋 *"Solid-State Battery Milestones"*
   - 🔐 *"Post-Quantum Cryptography Migration"*
   - 🤖 *"Small Language Models (SLMs) on Edge"*
4. **Trigger Agent**: Click **🚀 Start Research**.
5. **Show Live Evidence**: Navigate to the `📥 Collected Evidence` tab to show the 5 real-world sources with confidence scores and snippets.
6. **Show Cross-Source Matrix**: Open `⚖️ Source Comparison Matrix` to show how Gemma contrasts claims across sources.
7. **Verify Grounding**: Open `📑 Research Report` to show the 6 structured sections with numbered citations (`[1]`, `[2]`, etc.) linking to the actual sources.
8. **Export**: Click **📥 Download Research Report (.md)**.

---

## 💻 Local Setup & Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13.1)
- Git

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/openresearch-ai.git
cd openresearch-ai
pip install -r requirements.txt
```

### 3. Configure `.env`
Copy [.env.example](.env.example) to `.env`:
```powershell
Copy-Item .env.example .env
```
Fill in your keys:
```env
# Tavily API Key for live web research (Free at https://tavily.com)
TAVILY_API_KEY="tvly-your-actual-api-key"

# Ollama local settings
OLLAMA_BASE_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="gemma3:4b"
```

### 4. Install Ollama & Download Gemma 3
- Download from [ollama.com](https://ollama.com/download) or install via winget:
  ```powershell
  winget install Ollama.Ollama
  ```
- Pull the model:
  ```powershell
  ollama run gemma3:4b
  ```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Visit `http://localhost:8501`.

---

## 🌊 DigitalOcean Deployment (Hack Day Promo)

You can deploy OpenResearch AI directly to **DigitalOcean** using your Hack Day promo credits:

### Option A: DigitalOcean App Platform
1. Push your repository to GitHub.
2. In DigitalOcean Console, go to **App Platform** → **Create App** → Link your GitHub repository.
3. Select Dockerfile as the build type (uses the included [Dockerfile](Dockerfile)).
4. Add environment variables: `TAVILY_API_KEY` and `OLLAMA_BASE_URL`.

### Option B: DigitalOcean Droplet with Docker Compose
1. Create an Ubuntu GPU / high-memory Droplet.
2. SSH into your droplet and run:
   ```bash
   git clone https://github.com/your-username/openresearch-ai.git
   cd openresearch-ai
   docker compose up -d
   ```
3. Pull Gemma 3 inside the container:
   ```bash
   docker exec -it openresearch-ollama ollama pull gemma3:4b
   ```
4. Access your live app at `http://<YOUR_DROPLET_IP>:8501`.

---

## 📜 Licensing & Usage Terms

To ensure full compliance with the Hacktoberfest open-source AI criteria:

1. **Original Source Code**: Licensed under the **[MIT License](LICENSE)** (Permissive, open source, reusable).
2. **Open-Weight AI Model**: Google Gemma 3 weights are distributed under the **[Google Gemma Terms of Use](https://ai.google.dev/gemma/terms)**.
3. **Agent Skills Open Standard**: The synthesizer skill follows the open specification at [AgentSkills.io](https://agentskills.io/specification).
