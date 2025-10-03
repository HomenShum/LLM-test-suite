# LLM Test Suite


## 🌐 Live App

- Deployed Streamlit app: https://llm-test-suite-cafecorner.streamlit.app/

A comprehensive evaluation framework for testing and benchmarking Large Language Models (LLMs) with multi-agent orchestration, cost tracking, and interactive dashboards.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Orchestration**: Hierarchical agent system with supervisor and specialized leaf agents
- **Provider-Agnostic Cost Tracking**: Track tokens and costs across OpenAI, Anthropic, Google Gemini, OpenRouter, and more
- **Interactive Streamlit Dashboard**: Real-time visualization of test execution, performance metrics, and cost analysis
- **Comprehensive Test Suite**: Classification, tool sequencing, context pruning, and custom evaluation scenarios
- **Persistent Logging**: Historical test run tracking with detailed execution timelines
- **Dynamic Model Discovery**: Automatic pricing and model metadata fetching via Linkup API

### Agent Architecture
The system implements a true multi-agent scaffold with:
- **Supervisor Agent**: High-level orchestrator managing task decomposition and result synthesis
- **Specialized Leaf Agents**:
  - Web Research Agent: Information gathering and web search
  - Code Executor Agent: Python execution and computational tasks
  - Knowledge Retriever Agent: Domain knowledge and documentation retrieval
  - Content Generator Agent: Structured content creation
  - Validator Agent: Quality assurance and hallucination detection
  - Editor Agent: Content refinement and polishing

### Cost Tracking System
- **Pluggable Usage Extractors**: Support for any LLM provider with custom extractors
- **Autonomous Pricing Discovery**: Uses Linkup Search API to find current pricing from official sources
- **Smart Caching**: Caches pricing to disk (30-day TTL) to avoid repeated lookups
- **Detailed Analytics**: Per-call and aggregate statistics with provider/model breakdowns

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/HomenShum/LLM-test-suite.git
cd LLM-test-suite
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
Create a `.streamlit/secrets.toml` file or set environment variables:

```toml
# API Keys
OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"
OPENROUTER_API_KEY = "your-openrouter-key"
LINKUP_API_KEY = "your-linkup-key"  # Optional: for dynamic pricing

# Model Configuration
OPENAI_MODEL = "openai/gpt-5-mini"
GEMINI_MODEL = "google/gemini-2.5-flash"
OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct"

# API Routing
API_ROUTING_MODE = "openrouter"  # or "native"

# Dataset Configuration
DATASET_DIR = "test_dataset"
```

## 🎯 Quick Start

### Run the Streamlit Dashboard

```bash
streamlit run streamlit_test_v5.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Basic Usage Example

```python
from cost_tracker import CostTracker, combined_price_lookup
from leaf_agent_scaffold import SupervisorAgent, WebResearchAgent, CodeExecutorAgent

# Initialize cost tracker
ct = CostTracker()

# Create specialized agents
agents = [
    WebResearchAgent(llm_client),
    CodeExecutorAgent(llm_client)
]

# Create supervisor
supervisor = SupervisorAgent(agents)

# Execute complex task
result = await supervisor.execute(
    "Research the latest AI trends and create a summary report",
    mode="research"
)

# Track costs
ct.update(
    provider="OpenAI",
    model="gpt-5-mini",
    api="chat.completions",
    raw_response_obj=response,
    pricing_resolver=combined_price_lookup
)

print(f"Total cost: ${ct.totals['total_cost_usd']}")
```

## 📊 Test Scenarios

The suite includes several pre-configured test scenarios:

### 1. General Research (Web Search)
Tests basic web research and information synthesis capabilities.

### 2. PI Agent (Motor Control & Policy RAG)
Simulates vision-guided robotic manipulation with policy-based decision making.

### 3. Email Cybersecurity (Risk Scoring)
Analyzes phishing threats using threat intelligence and risk scoring.

### 4. General Classification (Inference)
Tests sentiment analysis and text classification capabilities.

## 🏗️ Project Structure

```
LLM_test_suite/
├── streamlit_test_v5.py          # Main Streamlit app entrypoint
├── requirements.txt              # Python dependencies
├── ui/                           # Streamlit UI modules
│   ├── test6_visual_llm.py       # Test 6 UI (modes, analysis, caching)
│   ├── test6_advanced_results.py # Visualizations, leaderboard, judge, Q&A
│   ├── test6_synthesis_display.py
│   ├── agent_dashboard.py
│   ├── sidebar.py, footer.py, data_generation.py, test_tabs.py
├── core/                         # Core analysis logic
│   ├── visual_qa_interface.py
│   ├── visual_meta_analysis.py
│   ├── visual_results_synthesis.py
│   ├── vision_visualizations.py
│   ├── visual_llm_clients.py, pricing.py
│   ├── image_collector.py, rating_extractor.py, models.py
│   ├── orchestrator.py, unified_orchestrator.py, dynamic_visual_analysis.py
│   ├── master_llm_curator.py, judges.py, analysis_history.py, api_clients.py
│   ├── summaries.py, test_runners.py, vision_model_discovery.py, reporting.py
├── utils/                        # Utilities & helpers
│   ├── execution_tracker.py, dashboard_logger.py
│   ├── advanced_visualizations.py, visualizations.py, gantt_charts.py, plotly_config.py
│   ├── data_helpers.py, helpers.py, ui_components.py
│   ├── model_discovery.py, model_metadata.py, stateful_components.py, test_image_generator.py
├── cost_tracker/                 # Provider-agnostic cost tracking
│   ├── tracker.py, extractors.py, pricing_resolver.py, readme.md
├── pricing_cache/                # Pricing/model cache (auto-generated)
│   ├── openrouter_pricing.json, openrouter_vision_models.json
├── test_dataset/                 # Datasets and sample images
│   ├── classification_dataset.csv, context_pruning_dataset.csv, tool_sequence_dataset.csv
│   └── visual_llm_images/ [...]
├── analysis_history/             # Past runs: images and results (auto-generated)
│   ├── images/<timestamps>/..., results/<timestamps>/...
├── config/
│   └── scenarios.py
├── scripts/
│   └── cleanup_docs.ps1
├── examples/
│   └── dynamic_visual_analysis_examples.py
├── agent_dashboard_logs/         # Execution logs (auto-generated)
├── model_costs/                  # (currently empty)
├── pricing_cache/                # OpenRouter cache (auto-generated)
├── CHANGELOG.md, README.md, .gitignore
└── .streamlit/secrets.toml       # Local secrets (not committed)
```

## 🔧 Configuration

### API Routing Modes

**OpenRouter Mode** (Recommended):
- Routes all API calls through OpenRouter
- Unified pricing and simpler configuration
- Access to 100+ models from multiple providers

**Native Mode**:
- Uses native SDKs (OpenAI SDK, Google Genai SDK)
- Access to advanced provider-specific features
- Requires separate API keys for each provider

### Model Selection

The system supports dynamic model discovery from:
- **Google Gemini**: gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite
- **OpenAI**: gpt-5, gpt-5-mini, gpt-5-nano
- **OpenRouter**: 100+ models including Mistral, Claude, Llama, DeepSeek

## 📈 Dashboard Features

### Execution Timeline
- Real-time Gantt chart visualization
- Agent hierarchy and dependencies
- Performance metrics and bottleneck detection

### Cost Analytics
- Per-call cost breakdown
- Provider/model comparison
- Token usage statistics
- Historical cost trends

### Memory & Security
- Memory snapshot viewer
- Security audit logs
- Self-correction history
- Policy update tracking

## 🧪 Testing

### Run Classification Tests
Tests model performance on sentiment classification with F1 scores and confusion matrices.

### Run Tool Sequence Tests
Evaluates agent's ability to execute correct tool sequences for complex tasks.

### Run Context Pruning Tests
Tests intelligent context management and action decision-making.

### Smoke Tests
Quick validation tests for critical functionality:
- General research capabilities
- Policy-based decision making
- Risk scoring and analysis
- Basic inference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by OpenAI, Google Gemini, and OpenRouter APIs
- Cost tracking inspired by provider-agnostic design principles
- Multi-agent architecture based on hierarchical task decomposition

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is an active research project. Features and APIs may change as the framework evolves.

