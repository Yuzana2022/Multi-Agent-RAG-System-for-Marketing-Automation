# A Multi-Agent Intelligent Marketing System

This repository presents the newest version of a multi-agent intelligent marketing system that implements a closed-loop, end-to-end automation pipeline for marketing intelligence and content operations. The system integrates retrieval-augmented generation (RAG), structured multi-agent orchestration, and human-in-the-loop (HITL) validation to support reliable insight extraction, content generation, automated publishing, and post-publication performance analysis.

Built upon LangChain and LangGraph, the proposed architecture emphasizes modularity, grounding, and extensibility, enabling controlled large language model (LLM) deployment in real-world marketing workflows. Compared with earlier iterations, this version introduces architectural optimizations, clearer agent responsibility boundaries, and enhanced reproducibility. It also incorporates comprehensive logging services, unit and integration testing frameworks, and robust retry/backoff mechanisms across all agents to improve reliability, observability, and fault tolerance in production-ready environments.

---

## System Overview

The proposed system adopts a sequential multi-agent architecture, in which each agent is responsible for a well-defined stage in the marketing intelligence lifecycle. Agents communicate via structured outputs validated by Pydantic schemas, ensuring consistency and robustness across the pipeline.

The complete workflow forms a closed feedback loop, beginning with external and internal knowledge acquisition and ending with automated performance evaluation to support iterative strategy refinement.

---

## Agent Architecture and Pipeline
### **Search Agent**

The Search Agent retrieves professional and industry-grade textual sources to ground downstream reasoning and mitigate hallucination risks in LLM-generated outputs.

External information is collected using Google SerpAPI.

Retrieved documents serve as authoritative references for subsequent retrieval and generation tasks.

### **Search Document Loader Agent**

This agent extracts raw textual content from URLs provided by the Search Agent and applies standardized preprocessing procedures, including text cleaning, normalization, and noise removal.

### **Local Document Loader Agent**

To support domain-specific and proprietary knowledge integration, the Local Document Loader Agent enables users to upload PDF documents from local storage. These documents undergo the same preprocessing pipeline as externally sourced materials, ensuring uniform representation.

### **Retrieval-Augmented Generation (RAG) Agent**

The RAG Agent implements a Map–Reduce-style retrieval framework:

*** Map Phase ***

- Document chunking and embedding

- Vector storage using a temporary Chroma database

- Sources include both online documents and locally uploaded files

*** Reduce Phase ***

- Semantic retrieval and ranking of the most relevant document chunks in response to user queries

- This design grounds LLM reasoning in retrieved evidence and improves factual consistency.

### **Insights Agent**

The Insights Agent synthesizes retrieved information into a concise set of five high-level, actionable insights. These insights function as an intermediate semantic abstraction layer, decoupling knowledge retrieval from content generation.

### **Content Generation Agent**

The Content Generation Agent produces publication-ready marketing content using strictly structured outputs defined via Pydantic schemas. The output schema includes:

- Insight identifier linkage

- Content format specification (e.g., article, blog, case study)

- Headline and full textual body

- Call-to-action and target audience definition

- Recommended distribution channels

- Explicit source references

While the current implementation focuses on text generation, the architecture supports future extensions to multimodal generation (e.g., video scripts, visual assets) and multilingual content production via additional agents.

### **Human-in-the-Loop (HITL)**

To ensure quality control and ethical deployment, a Human-in-the-Loop mechanism is incorporated. Generated content undergoes manual review, validation, and optional revision before publication, ensuring alignment with organizational standards.

### **Auto Publish Agent**

The Auto Publish Agent schedules and disseminates approved content across social media or marketing platforms.

A Facebook API integration is provided as a reference implementation.

The design is platform-agnostic and supports extensibility.

### **Auto Analysis Report Agent**

Following publication, the Auto Analysis Report Agent generates analytical summaries evaluating engagement metrics, effectiveness, and return on investment (ROI). These reports enable data-driven optimization of future marketing strategies, closing the system feedback loop.

---

## Design Principles

The system is guided by the following design principles:

- Modularity: Each agent is independently extensible and replaceable

- Grounded Generation: RAG is employed to constrain LLM outputs to retrieved evidence

- Structured Communication: Pydantic schemas enforce data consistency across agents

- Human Oversight: HITL safeguards are integrated at critical decision points

Version 3.0 prioritizes architectural clarity and experimental reproducibility. Production-level optimizations such as asynchronous execution, caching, and persistent long-term memory are intentionally excluded to maintain research focus.

---

## Implementation Details
### **Technology Stack**

- python 3.12.7
- chromadb 1.4.1
- langchain_community 0.4.1
- langchain_core 1.2.9
- langchain_openai 1.1.8
- langchain_text_splitters 1.1.0
- langchain 1.2.8
- langchain-openai 1.1.8
- langgraph 1.0.7
- openai 2.17.0
- pydantic 2.8.2
- python-dotenv 1.2.2
- requests 2.32.5
- tenacity 8.2.3

### **How to Test**
- Install the project dependencies using:
   pip install -e .
   (Recommended for development environments)
- Alternatively, install dependencies from the requirements file:
   pip install -r requirements.txt
- Run all unit tests:
   pytest tests/unit_tests/
- Run all integration tests:
   pytest tests/integration_tests/
- To test the orchestration workflow independently, run:
   python run_agents_separately.py
   When prompted, enter `1` to select the default workflow. You may also modify the query input to evaluate the system on different topics and use cases.

---

## GenAI Assistance Disclosure

The system architecture, agent framework, and implementation strategy presented in this work were independently conceived and developed by the author. Generative AI tools were employed only in a supplementary capacity: Claude assisted with code refactoring and optimization, while ChatGPT, Gemini, and DeepSeek were used for debugging assistance, technical writing enhancement, and documentation review. These tools did not contribute to the underlying research design, architectural decisions, implementation methodology, or technical innovations. All intellectual contributions, engineering decisions, and final outputs remain the sole responsibility of the author.
