---
name: openresearch-synthesizer
version: 1.0.0
description: Autonomous multi-source evidence extraction and cross-source research synthesis skill powered by open-weight Small Language Models (Google Gemma 3 4B) and web retrieval.
license: MIT
author: OpenResearch AI Contributors
tags:
  - agent-skill
  - research
  - gemma
  - ollama
  - open-source-ai
  - tavily
---

# OpenResearch Synthesizer Agent Skill

A standardized skill adhering to the **Agent Skills Open Standard** (https://agentskills.io/specification) for rigorous, multi-source scientific and technological evidence synthesis.

## Skill Overview

The `openresearch-synthesizer` skill coordinates evidence gathering across web search engines (e.g., Tavily API) and dispatches normalized evidence blocks into an open-weight Small Language Model (SLM), specifically **Google Gemma 3 4B** (<10B parameters) via **Ollama**.

## Core Capabilities

1. **Evidence Normalization**: Ingests up to 5 verified web sources, indexing them as `[1]` through `[5]`.
2. **Strict Zero-Hallucination Grounding**: Forces model outputs to cite exclusively verified source spans.
3. **Cross-Source Contradiction Detection**: Extracts points of consensus versus technical/economic divergence across sources.
4. **Structured Report Assembly**: Produces an executive-grade 6-section research brief with inline markdown citations.

## Input Parameters

```json
{
  "topic": {
    "type": "string",
    "description": "The target research question or technical topic to investigate.",
    "required": true
  },
  "max_sources": {
    "type": "integer",
    "description": "Maximum number of web sources to retrieve (default: 5).",
    "default": 5
  },
  "model": {
    "type": "string",
    "description": "Open-weight SLM model name (default: 'gemma3:4b').",
    "default": "gemma3:4b"
  }
}
```

## Workflow Execution Steps

1. **Dispatch Search**: Execute web search query using Tavily API.
2. **Rank & Extract**: Collect top 5 snippets, URLs, and titles.
3. **Prompt Gemma 3**:
   - Inject system prompt mandating bracketed citations `[1]`, `[2]`.
   - Provide raw evidence blocks.
4. **Enforce 6-Section Schema**:
   - `1. Executive Summary`
   - `2. Key Findings`
   - `3. Comparison of Different Sources`
   - `4. Agreements and Conflicting Claims`
   - `5. Evidence-Based Recommendations`
   - `6. Limitations and Conclusion`
5. **Format References**: Append verified reference links at the end of the report.

## Integration Example

```python
from app import get_demo_evidence, synthesize_fallback_report, query_ollama_gemma

# Execute skill logic
topic = "Breakthroughs in solid-state batteries in 2025"
evidence = get_demo_evidence(topic)
success, report = query_ollama_gemma(topic, evidence, model_name="gemma3:4b")
print(report)
```
