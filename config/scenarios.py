"""
Static configuration data for test scenarios, prompts, and constants.
Extracted from streamlit_test_v5.py to reduce main file size.
"""

# ============================================================
# DEMONSTRATION SCENARIO CONFIGURATIONS
# ============================================================

PI_AGENT_GOAL_PROMPT = """Autonomously fold a crumpled shirt using vision-guided robotic manipulation. The agent must:

1. PERCEPTION: Analyze the shirt's current state (spatial coordinates, fabric type, wrinkle severity)
2. PLANNING: Determine optimal folding strategy based on current Folding Policy
3. EXECUTION: Generate precise motor control commands for robotic manipulation
4. VERIFICATION: Assess folding quality through perception feedback
5. ADAPTATION: Update Folding Policy if the attempt fails

EXECUTION REQUIREMENTS:
- Use Web Research Agent → repurposed as Vision/Perception Agent (analyze shirt state)
- Use Code Executor Agent → repurposed as Motor Control Agent (generate manipulation commands)
- Use Validator Agent → repurposed as Perception Feedback Agent (verify folding quality)
- Use Content Generator Agent → repurposed as Task Summary Agent (compile results)
- IF folding fails: Trigger policy update with corrective constraints

SUCCESS CRITERIA:
- Shirt successfully folded (simulated) OR
- Folding Policy Block updated with failure analysis and corrective constraints"""

FOLDING_POLICY_BLOCK = """FOLDING POLICY v1.0:
- Locate shirt corners using edge detection
- Fold along vertical centerline first
- Apply 2N force for fabric manipulation
- Verify fold alignment within 5mm tolerance
- CONSTRAINT: Monitor for object slippage during manipulation"""

CYBERSECURITY_GOAL_PROMPT = """Analyze a suspected phishing email link (http://suspicious-login.ly/verify) for security threats. The agent must:

1. RECONNAISSANCE: Research domain reputation, registration details, and hosting infrastructure
2. VERIFICATION: Validate findings and detect potential hallucinations in threat intelligence
3. RISK SCORING: Calculate threat risk score using Threat Policy formula
4. REPORTING: Generate defense policy recommendation with actionable steps

EXECUTION REQUIREMENTS:
- Use Web Research Agent to gather domain intelligence (WHOIS, hosting, brand similarity)
- Use Validator Agent to verify findings and detect fabricated threat indicators
- Use Code Executor Agent to calculate risk score using policy formula
- Use Content Generator Agent to compile threat report with defense recommendations
- IF risk score > 0.7: Update Threat Policy Block with new indicators

SUCCESS CRITERIA:
- Threat risk score calculated using policy formula AND
- Defense recommendation provided (ALLOW/WARN/BLOCK)"""

THREAT_POLICY_BLOCK = """THREAT POLICY v1.0:
- Risk Score Formula: (brand_impersonation * 0.4) + (network_reputation * 0.3) + (domain_age * 0.3)
- Brand Impersonation Threshold: similarity > 0.8 = HIGH RISK
- Network Reputation Sources: VirusTotal, URLhaus, PhishTank
- Defense Actions:
  * risk < 0.3 = ALLOW
  * 0.3-0.7 = WARN
  * > 0.7 = BLOCK
- CONSTRAINT: Flag domains with > 0.9 brand similarity and < 7 days age as CRITICAL"""

# ============================================================
# SMOKE TEST SCENARIOS FOR LIVE DASHBOARD VALIDATION
# ============================================================

SMOKE_TEST_SCENARIOS = {
    "1. General Research (Web Search)": {
        "goal": "Find the main product lines and headquarters location for Symbolica AI. Summarize concisely.",
        "agents": ["web_researcher", "content_generator", "validator"],
        "policy": None,
        "mode": "research",
    },
    "2. PI Agent (Motor Control & Policy RAG)": {
        "goal": PI_AGENT_GOAL_PROMPT,
        "agents": ["web_researcher", "code_executor", "validator"],
        "policy": FOLDING_POLICY_BLOCK,
        "mode": "research",
    },
    "3. Email Cybersecurity (Risk Scoring)": {
        "goal": "Using the Threat Policy, quickly calculate the risk score for Impersonation=0.9, Reputation=0.1, Age=5 days. Output JUST the Python code for the calculation.",
        "agents": ["web_researcher", "code_executor"],
        "policy": THREAT_POLICY_BLOCK,
        "mode": "analysis",
    },
    "4. General Classification (Inference)": {
        "goal": "Classify the sentiment of the text: 'The agent framework is fast and stable.'",
        "agents": ["web_researcher", "content_generator"],
        "policy": None,
        "mode": "inference",
    }
}

# ============================================================
# SUGGESTED PROMPTS FOR DATA GENERATION
# ============================================================

SUGGESTED_PROMPTS = {
    "Classification": [
        "A classification dataset for identifying sentiment in social media posts, with labels: positive, negative, neutral.",
        "Generate queries asking for technical support for a banking application, categorized by: login_issue, transaction_error, account_settings, general_help.",
        "Create short news headlines classified into five topics: sports, finance, politics, tech, weather."
    ],
    "Tool/Agent Sequence": [
        "Queries that require a sequence of internal tools like: crm_lookup -> schedule_callback, or check_inventory -> order_part.",
        "Generate customer support questions requiring tool sequences for flight booking: check_availability -> select_seat -> confirm_payment.",
        "Queries needing a single tool call followed by a database query: user_auth -> internal_db_read."
    ],
    "Context Pruning": [
        "Conversational turns where the new question only requires the `tool_logs` and `new_question` to take the action `kb_lookup` (summary is irrelevant).",
        "Scenarios where the `instruction` and `summary` are critical to deciding the `tool_call` action.",
        "Generate examples where the `new_question` is a simple follow-up, and the action should be `general_answer`, keeping only `summary`."
    ]
}

# ============================================================
# DEFAULT DATASET GENERATION PROMPTS
# ============================================================

DEFAULT_DATASET_PROMPTS = {
    "Classification": """Generate a diverse classification dataset for an IT support chatbot with three categories:
1. 'general_chat' - casual greetings, small talk, non-technical questions
2. 'kb_lookup' - questions about IT concepts, tools, best practices, knowledge base queries
3. 'tool' - requests that require calling specific tools or APIs (e.g., database queries, system lookups, device information)

Create varied, realistic user queries that cover common IT support scenarios. Include questions about network devices, incidents, IP addresses, locations, and general IT knowledge.""",

    "Tool/Agent Sequence": """Generate queries that require multi-step tool sequences for an IT operations agent:
- Single tool calls: check_device, get_incident, lookup_ip
- Two-step sequences: authenticate_user -> query_database, check_inventory -> create_order
- Three-step sequences: validate_request -> fetch_data -> update_system

Include realistic IT operations scenarios like device management, incident tracking, and network operations.""",

    "Context Pruning": """Generate context pruning test cases for a conversational AI agent with three action types:
1. 'general_answer' - Simple questions that only need conversation summary
2. 'kb_lookup' - Knowledge base queries that need instruction and tool logs
3. 'tool_call' - Actions requiring full context (instruction, summary, tool logs)

Create realistic multi-turn conversations with varying context requirements. Include banking, IT support, and customer service scenarios."""
}

# ============================================================
# SKELETON COLUMNS FOR DATASETS
# ============================================================

SKELETON_COLUMNS = [
    "query",
    "classification",
    "classification_result_openrouter_mistral",
    "classification_result_openrouter_mistral_rationale",
    "classification_result_openrouter_mistral_confidence",
    "probabilities_openrouter_mistral_raw",
    "probabilities_openrouter_mistral_calibrated",
    "latency_openrouter_mistral",
    "classification_result_openai",
    "classification_result_openai_rationale",
    "classification_result_openai_confidence",
    "probabilities_openai_raw",
    "probabilities_openai_calibrated",
    "latency_openai",
    "classification_result_third",
    "classification_result_third_rationale",
    "classification_result_third_confidence",
    "probabilities_third_raw",
    "probabilities_third_calibrated",
    "latency_third",
]

# ============================================================
# ROW LIMIT OPTIONS
# ============================================================

ROW_LIMIT_OPTIONS = {"First 5": 5, "First 25": 25, "First 100": 100, "All": None}

# ============================================================
# LABEL NORMALIZATION MAP
# ============================================================

CANON_MAP = {
    # general
    "general": "general_chat",
    "general_chat": "general_chat",
    "general chat": "general_chat",
    "chat": "general_chat",
    "greeting": "general_chat",
    "small_talk": "general_chat",
    "smalltalk": "general_chat",
    "casual": "general_chat",
    
    # kb_lookup
    "kb": "kb_lookup",
    "kb_lookup": "kb_lookup",
    "kb lookup": "kb_lookup",
    "knowledge": "kb_lookup",
    "knowledge_base": "kb_lookup",
    "knowledgebase": "kb_lookup",
    "search": "kb_lookup",
    "query": "kb_lookup",
    "info": "kb_lookup",
    "information": "kb_lookup",
    
    # tool
    "tool": "tool",
    "tool_call": "tool",
    "tool call": "tool",
    "action": "tool",
    "api": "tool",
    "api_call": "tool",
    "function": "tool",
    "execute": "tool",
    "command": "tool",
}

# ============================================================
# TEST FLOW DESCRIPTIONS
# ============================================================

TEST_FLOWS = {
    1: "📊 Load Data → 🤖 Classify (2 Models) → 📈 Evaluate F1/Latency → ❌ Error Analysis",
    2: "📊 Load Data → 🔍 Classify (3 Models) → ⚖️ Weighted Ensemble → 📈 Evaluate Performance",
    3: "📊 Load Data → 🔍 Classify (3 Models) → ⚖️ Weighted Scores → 👨‍⚖️ LLM Judge → 📈 Evaluate",
    4: "📊 Load Context → ✂️ Prune Context → 🎯 Predict Action → 📊 Measure Accuracy",
    5: "🎯 Goal → 🧩 Decompose → 📊 Prioritize → 🔄 Execute → ✅ Verify → 📚 Index → 🔁 Converge"
}

# ============================================================
# JUDGE SCHEMA AND INSTRUCTIONS
# ============================================================

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_choice_model": {
            "type": "string",
            "description": "One of: mistral, gpt5, third"
        },
        "final_label": {
            "type": "string"
        },
        "judge_rationale": {
            "type": "string"
        }
    },
    "required": ["final_choice_model", "final_label", "judge_rationale"],
    "additionalProperties": False
}

JUDGE_INSTRUCTIONS = "You are a neutral judge... Return ONLY JSON with: final_choice_model, final_label, judge_rationale."

