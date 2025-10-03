"""
UnifiedOrchestrator class extracted from streamlit_test_v5.py

This module contains the UnifiedOrchestrator class which provides three modes:
1. Direct Inference: Pattern matching (classification, prediction)
2. Computational Analysis: Statistics, simulations, optimization
3. Research Tasks: Multi-source information gathering with decomposition

And three coordination patterns:
1. Solo: Single agent executes independently
2. Subagent: Hierarchical delegation with specialized subagents
3. Multi-Agent: Peer collaboration with proposals, review, and consensus
"""

import asyncio
import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from google import genai
from google.genai import types
import streamlit as st

# Import orchestrator base classes
from core.orchestrator import (
    Budget,
    TurnMetrics,
    OrchestratorResult,
    Task,
    VerificationResult,
    TaskCache,
    KnowledgeIndex,
    AgentCoordinationPattern,
    GeminiLLMClient,
    GeminiTaskPlanner,
    GeminiResultSynthesizer
)

# Import leaf agent scaffold
from leaf_agent_scaffold import (
    SupervisorAgent,
    AgentType,
    WebResearchAgent,
    CodeExecutorAgent,
    ContentGeneratorAgent,
    ValidatorAgent,
    TaskPlanner,
    ResultSynthesizer
)

# Import execution tracker
from utils.execution_tracker import ExecutionTracker

# Get API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class UnifiedOrchestrator:
    """
    Three modes for different problem types:
    1. Direct Inference: Pattern matching (classification, prediction)
    2. Computational Analysis: Statistics, simulations, optimization
    3. Research Tasks: Multi-source information gathering with decomposition

    Three coordination patterns:
    1. Solo: Single agent executes independently
    2. Subagent: Hierarchical delegation with specialized subagents
    3. Multi-Agent: Peer collaboration with proposals, review, and consensus

    This creates a 3×3 matrix of agent architectures (mode × pattern).
    """
    def __init__(
        self,
        goal: str,
        test_data: Optional[List[Dict]] = None,
        budget: Budget = None,
        mode: Optional[str] = None,
        coordination_pattern: Optional[str] = None,
        peer_agent_roles: Optional[List[str]] = None
    ):
        self.goal = goal
        self.test_data = test_data or []
        self.budget = budget or Budget(mode="turns", max_turns=5)

        # Auto-detect mode if not specified
        if mode is None:
            self.mode = self._detect_mode(goal)
        else:
            self.mode = mode  # "inference", "analysis", or "research"

        # Auto-detect coordination pattern if not specified
        if coordination_pattern is None:
            self.coordination_pattern = self._detect_coordination_pattern(goal, self.mode)
        else:
            self.coordination_pattern = coordination_pattern  # "solo", "subagent", or "multi_agent"

        # Multi-agent configuration
        self.peer_agent_roles = peer_agent_roles or self._get_default_peer_roles(self.mode)

        # Mode-specific components
        self.cache = TaskCache()
        self.index = KnowledgeIndex()
        self.policies: Dict[str, float] = defaultdict(float)  # learnable weights
        self.dataset_collector = FineTuneDatasetCollector()
        self.research_results = []

        # Turn-based tracking
        self.turn_history: List[TurnMetrics] = []
        self.best_performance = 0.0
        self.best_code = None
        self.best_turn = 0

        # CRITICAL: Initialize iteration counter
        self.iteration = 0

        # Track per-item failures for inference mode
        self.failure_analysis: List[Dict] = []

        # USE UNIFIED EXECUTION TRACKER (not custom event system)
        self.tracker = st.session_state.get('execution_tracker')
        if not self.tracker:
            self.tracker = ExecutionTracker()
            st.session_state['execution_tracker'] = self.tracker

        # Initialize persistent dashboard logger for Test 5
        self.dashboard_logger = None
        if 'test5_dashboard_logger' in st.session_state:
            self.dashboard_logger = st.session_state['test5_dashboard_logger']

        # Initialize stateful components (memory, security, self-correction)
        self.memory_manager = None
        self.security_agent = None
        self.self_correction_manager = None

        # These will be initialized when run() is called
        self._stateful_components_initialized = False

    def _detect_mode(self, goal: str) -> str:
        """Auto-detect which mode to use based on goal."""
        goal_lower = goal.lower()

        # Research indicators
        research_keywords = ['search', 'research', 'find information', 'tell me about',
                            'what is', 'who is', 'compare', 'analyze market']
        if any(kw in goal_lower for kw in research_keywords):
            return "research"

        # Analysis indicators
        analysis_keywords = ['analyze', 'compute', 'calculate', 'simulate',
                           'optimize', 'compare performance']
        if any(kw in goal_lower for kw in analysis_keywords):
            return "analysis"

        # Default to inference if test data provided
        if self.test_data:
            return "inference"

        return "research"  # Default fallback

    def _detect_coordination_pattern(self, goal: str, mode: str) -> str:
        """
        Auto-detect coordination pattern based on task complexity.

        Heuristics:
        - Solo: Simple, single-step tasks
        - Subagent: Complex tasks requiring decomposition and specialization
        - Multi-Agent: Tasks requiring diverse perspectives or consensus
        """
        goal_lower = goal.lower()

        # Multi-agent indicators (diverse perspectives needed)
        multi_agent_keywords = [
            'consensus', 'multiple perspectives', 'diverse', 'compare approaches',
            'peer review', 'collaborative', 'team', 'different angles',
            'cross-validate', 'reconcile'
        ]
        if any(kw in goal_lower for kw in multi_agent_keywords):
            return AgentCoordinationPattern.MULTI_AGENT.value

        # Subagent indicators (complex workflow requiring specialization)
        subagent_keywords = [
            'pipeline', 'workflow', 'multi-step', 'decompose', 'delegate',
            'specialized', 'hierarchical', 'orchestrate', 'coordinate',
            'iterative refinement', 'evaluate and refine'
        ]
        if any(kw in goal_lower for kw in subagent_keywords):
            return AgentCoordinationPattern.SUBAGENT.value

        # Complexity heuristics
        word_count = len(goal.split())
        has_multiple_objectives = goal.count(',') >= 2 or goal.count('and') >= 2

        # Complex tasks → Subagent
        if word_count > 30 or has_multiple_objectives:
            return AgentCoordinationPattern.SUBAGENT.value

        # Default to solo for simple tasks
        return AgentCoordinationPattern.SOLO.value

    def _get_default_peer_roles(self, mode: str) -> List[str]:
        """Get default peer agent roles based on mode."""
        if mode == "inference":
            return ["Pattern Analyst", "Rule Designer", "Evaluator"]
        elif mode == "analysis":
            return ["Data Scientist", "Algorithm Designer", "Code Reviewer"]
        elif mode == "research":
            return ["Market Analyst", "Technical Expert", "Domain Specialist"]
        else:
            return ["Agent 1", "Agent 2", "Agent 3"]

    async def decompose(self, goal: str, context: Dict) -> List[Task]:
        """Break goal into prioritized subtasks with proper error handling.

        Mode-specific behavior:
        - inference: Single refinement task analyzing failures
        - analysis: Single code generation task
        - research: Multiple parallel research tasks
        """

        # Track iteration properly
        iteration = len(self.turn_history) + 1

        if self.mode == "inference":
            # ITERATIVE REFINEMENT MODE: Single task that analyzes failures and improves code
            if iteration == 1:
                # First turn: generate initial solution
                prompt = f"""
Generate a complete Python script that predicts tool sequences.

TEST DATA (first 5 examples):
{json.dumps(self.test_data[:5], indent=2)}

Requirements:
- Complete predict_tools(query) function
- Test harness that creates 'predictions' list
- Print "Accuracy: X.XXXX" to stdout

Return ONLY executable Python code.
"""
            else:
                # Subsequent turns: analyze failures and refine
                failures_count = int((1 - self.best_performance) * len(self.test_data))

                prompt = f"""
CURRENT CODE (accuracy {self.best_performance:.3f}):
```python
{self.best_code}
```

This code fails on {failures_count} of {len(self.test_data)} test cases.

SAMPLE TEST DATA (showing patterns to learn):
{json.dumps(self.test_data[:5], indent=2)}

YOUR TASK:
1. Analyze why the current code fails on {failures_count} cases
2. Identify patterns in the data (keywords? tool combinations? edge cases?)
3. Modify the EXISTING code to fix these issues
4. Return the COMPLETE improved code

Focus on fixing actual errors, not rewriting from scratch.
Return ONLY executable Python code.
"""

            # Single task for iterative refinement
            task = Task(
                id=f"refine_iteration_{iteration}",
                goal="Analyze failures and improve the predictor code",
                novelty_score=0.7,
                estimated_roi=0.9
            )

            # Skip LLM call for task generation in iterative mode
            st.info(f"✅ Iterative mode: Turn {iteration} refinement task")
            return [task]

        else:
            # RESEARCH MODE: Multiple parallel tasks (original behavior)
            prompt = f"""
You are designing code to predict tool sequences from queries.

CURRENT STATE:
- Iteration: {iteration}
- Best Accuracy: {self.best_performance:.3f}

SAMPLE DATA:
{json.dumps(self.test_data[:3], indent=2)}

Generate 3 CONCRETE coding tasks. Each must be implementable as a single Python function.

GOOD TASK: "Write a keyword matching function that maps 'check status' → ['check_device']"
BAD TASK: "Design a tool registry architecture"

GOOD TASK: "Add regex pattern matching for IP addresses to trigger 'get_ip' tool"
BAD TASK: "Implement deterministic parsing logic"

GOOD TASK: "Create a dictionary mapping common verbs (check, get, set) to tool names"
BAD TASK: "Improve the prompt template"

Return JSON array with SPECIFIC, CODE-FOCUSED tasks:
[
  {{
    "id": "task_1",
    "goal": "Write keyword_to_tools() dict mapping common words to tool names",
    "depends_on": [],
    "novelty": 0.8,
    "roi": 0.9
  }},
  {{
    "id": "task_2",
    "goal": "Add if/elif logic to check for 'status' keyword and return ['check_device']",
    "depends_on": [],
    "novelty": 0.7,
    "roi": 0.8
  }},
  {{
    "id": "task_3",
    "goal": "Implement query.lower().split() to extract words for matching",
    "depends_on": [],
    "novelty": 0.6,
    "roi": 0.7
  }}
]

Each task should describe EXACTLY what code to write, not abstract concepts.
"""

        try:
            # Use the selected judge/pruner model (flexible routing)
            model = st.session_state.get('judge_model', 'openai/gpt-5-mini')

            # Route based on provider
            provider = _get_provider_from_model_id(model)

            if provider == "openai" and OPENAI_API_KEY:
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                native_model = _to_native_model_id(model)
                resp = await client.chat.completions.create(
                    model=native_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

                # Track cost
                st.session_state.cost_tracker.update(
                    provider="OpenAI", model=native_model,
                    api="chat.completions.create",
                    raw_response_obj=resp,
                    pricing_resolver=combined_price_lookup
                )

                content = resp.choices[0].message.content

            elif OPENROUTER_API_KEY:
                # Fallback to OpenRouter
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                body = {
                    "model": _to_openrouter_model_id(model),
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}
                }

                async with httpx.AsyncClient(timeout=60) as client_http:
                    resp = await client_http.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=body
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # Track cost
                    st.session_state.cost_tracker.update(
                        provider="OpenRouter",
                        model=_to_openrouter_model_id(model),
                        api="chat.completions",
                        raw_response_json=data,
                        pricing_resolver=custom_openrouter_price_lookup
                    )

                    content = data["choices"][0]["message"]["content"]
            else:
                st.error("❌ No LLM provider available for decomposition")
                return self._get_fallback_tasks()

            # Parse response
            tasks_data = json.loads(content)

            # Handle both direct array and wrapped formats
            if isinstance(tasks_data, dict):
                tasks_data = tasks_data.get("subtasks", tasks_data.get("tasks", []))

            tasks = []
            for t in tasks_data:
                task = Task(
                    id=t["id"],
                    goal=t["goal"],
                    parent_ids=set(t.get("depends_on", [])),
                    novelty_score=float(t.get("novelty", 0.5)),
                    estimated_roi=float(t.get("roi", 0.5))
                )
                tasks.append(task)

            # CRITICAL FIX: Ensure we never return empty list
            if not tasks or len(tasks) == 0:
                st.warning("⚠️ LLM returned no tasks. Using fallback.")
                return self._get_fallback_tasks()

            st.info(f"✅ Decomposed into {len(tasks)} tasks")
            return tasks

        except Exception as e:
            st.error(f"❌ Decomposition failed: {e}")
            st.exception(e)  # Show full traceback

            # Return fallback tasks instead of empty list
            return self._get_fallback_tasks()

    def _get_fallback_tasks(self) -> List[Task]:
        """Return fallback tasks when decomposition fails."""
        return [
            Task(
                id="fallback_1",
                goal="Implement basic tool sequence parser with regex pattern matching",
                novelty_score=0.8,
                estimated_roi=0.9
            ),
            Task(
                id="fallback_2",
                goal="Add error handling and logging for failed predictions",
                novelty_score=0.6,
                estimated_roi=0.7
            ),
            Task(
                id="fallback_3",
                goal="Create test harness to evaluate accuracy on sample data",
                novelty_score=0.7,
                estimated_roi=0.8
            )
        ]

    def prioritize(self, tasks: List[Task]) -> List[Task]:
        """Score tasks by novelty * coverage * ROI"""
        for task in tasks:
            # Apply learned policies
            policy_weight = self.policies.get(task.goal[:20], 1.0)

            task.priority = (
                task.novelty_score * 0.4 +
                task.estimated_roi * 0.4 +
                policy_weight * 0.2
            )

        return sorted(tasks, key=lambda t: t.priority, reverse=True)

    def parents_verified(self, task: Task) -> bool:
        """Check if all parent tasks are verified"""
        return all(pid in self.cache.verified_tasks for pid in task.parent_ids)

    def execute_and_evaluate(self, code: str, sample_size: int = 25) -> float:
        """Execute generated code and measure accuracy against test data subset with safety checks."""
        try:
            # SECURITY: Validate code syntax before execution
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                st.error(f"❌ Invalid Python syntax in generated code: {e}")
                return 0.0

            # Use same subset size as in dispatch_sub_agent
            test_subset = self.test_data[:sample_size]

            # Create isolated namespace with test data
            namespace = {
                "test_data": test_subset,
                "json": json,
                "re": re,
                "List": List,
                "Dict": Dict,
                "Any": Any
            }

            # SECURITY: Execute with timeout (10 seconds for small subset)
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timeout")

            # Set timeout (only works on Unix-like systems)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10 second timeout

                # Execute the code
                exec(code, namespace)

                signal.alarm(0)  # Cancel timeout
            except AttributeError:
                # Windows doesn't support SIGALRM, use threading timeout instead
                import threading

                exec_exception = None

                def exec_with_exception():
                    nonlocal exec_exception
                    try:
                        exec(code, namespace)
                    except Exception as e:
                        exec_exception = e

                exec_thread = threading.Thread(target=exec_with_exception)
                exec_thread.daemon = True
                exec_thread.start()
                exec_thread.join(timeout=10.0)

                if exec_thread.is_alive():
                    st.error("❌ Code execution timeout (10 seconds)")
                    return 0.0

                if exec_exception:
                    raise exec_exception

            # Extract predictions (code should create 'predictions' variable)
            predictions = namespace.get('predictions', [])

            if not predictions:
                st.warning("⚠️ Code did not create 'predictions' variable")
                return 0.0

            # Validate predictions structure
            if not isinstance(predictions, list):
                st.warning(f"⚠️ 'predictions' should be a list, got {type(predictions).__name__}")
                return 0.0

            if len(predictions) != len(test_subset):
                st.warning(f"⚠️ Prediction count mismatch: {len(predictions)} vs {len(test_subset)}")
                # Try to calculate accuracy anyway with available predictions
                test_subset = test_subset[:len(predictions)]

            # Calculate accuracy
            correct = 0
            for i, (pred, expected) in enumerate(zip(predictions, test_subset)):
                # Validate prediction structure
                if not isinstance(pred, dict):
                    continue

                pred_seq = pred.get('predicted_sequence', [])
                exp_seq = expected.get('expected_sequence', [])

                # Normalize to strings for comparison
                if isinstance(pred_seq, list):
                    pred_seq = [str(x).strip() for x in pred_seq]
                else:
                    pred_seq = []

                if isinstance(exp_seq, list):
                    exp_seq = [str(x).strip() for x in exp_seq]
                else:
                    exp_seq = []

                if pred_seq == exp_seq:
                    correct += 1

            accuracy = correct / len(test_subset) if test_subset else 0.0
            return accuracy

        except TimeoutError:
            st.error("❌ Code execution timeout")
            return 0.0
        except Exception as e:
            st.warning(f"⚠️ Code execution failed: {e}")
            import traceback
            st.text(traceback.format_exc()[:500])
            return 0.0

    async def dispatch_sub_agent(self, task: Task) -> Any:
        """Execute task via Gemini code execution with PROGRESSIVE BATCHING."""

        # CRITICAL FIX: Progressive batch sizing to avoid Gemini timeout
        # Gemini Code Execution can't handle 149 examples in one run (resource/time limits)
        # Solution: Start small (20), grow with confidence (50, 100)
        iteration = len(self.turn_history) + 1

        if iteration == 1:
            batch_size = 20  # Start small to ensure success
        elif iteration == 2:
            batch_size = 50  # Expand if Turn 1 worked
        else:
            batch_size = min(100, len(self.test_data))  # Max 100 at a time

        test_subset = self.test_data[:batch_size]

        st.info(f"🔄 **Turn {iteration}: Training on {batch_size} examples** (progressive batching to avoid timeout)")
        st.caption(f"Note: Will evaluate on full {len(self.test_data)} examples locally after code generation")

        # Show diverse examples for learning
        diverse_samples = []
        if len(test_subset) >= 10:
            diverse_samples = [
                test_subset[0],  # First
                test_subset[len(test_subset)//4],  # 25%
                test_subset[len(test_subset)//2],  # 50%
                test_subset[3*len(test_subset)//4],  # 75%
                test_subset[-1]  # Last
            ]
        else:
            diverse_samples = test_subset[:min(5, len(test_subset))]

        sample_data = json.dumps(diverse_samples, indent=2)

        if iteration == 1:
            # First turn: generate initial solution (FUNCTION ONLY)
            prompt = f"""
Generate a predict_tools function that predicts tool sequences from queries.

SAMPLE DATA (batch 1 of ~8 batches, showing {len(diverse_samples)} diverse examples):
{sample_data}

Your function will be tested on ALL {len(self.test_data)} examples across multiple batches.

Requirements:
- Function signature: def predict_tools(query: str) -> list
- Return list of tool names like ["check_device"], ["get_incident"], etc.
- Handle various patterns: device checks, incidents, IP lookups, authentication, database queries
- Must work on ANY query, not just these examples

EXAMPLE PATTERNS TO HANDLE:
- "check device status" → ["check_device"]
- "incident INC-123" → ["get_incident"]
- "lookup IP 192.168.1.1" → ["get_ip_info"]
- "authenticate user and query database" → ["authenticate_user", "query_database"]

Return ONLY the predict_tools function code (no test harness, no test data).
"""
        else:
            # Subsequent turns: refine based on failures
            failures_count = int((1 - self.best_performance) * len(self.test_data))

            prompt = f"""
CURRENT FUNCTION (accuracy {self.best_performance:.3f} on full {len(self.test_data)} examples):
```python
{self.best_code}
```

This function fails on {failures_count} of {len(self.test_data)} test cases.

SAMPLE DATA (showing diverse examples):
{sample_data}

TASK: Improve the predict_tools function to handle more patterns.

Common failure patterns to address:
- Multi-step sequences (e.g., authenticate then query)
- IP address detection
- Incident ID patterns
- Inventory/ordering flows
- Edge cases

Return ONLY the improved predict_tools function code.
"""

        try:
            # CRITICAL FIX: Create client without async context manager
            # genai.Client doesn't support async context managers
            client = genai.Client(api_key=GEMINI_API_KEY)

            def sync_call():
                return client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )

            # Add 90 second timeout (Gemini code execution can be slow)
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(sync_call),
                    timeout=90.0
                )
            except asyncio.TimeoutError:
                st.error(f"⏱️ Task {task.id} timed out after 90 seconds")
                return {
                    "task_id": task.id,
                    "code": None,
                    "accuracy": 0.0,
                    "error": "Timeout: Gemini code execution exceeded 90 seconds"
                }

            # Track cost
            st.session_state.cost_tracker.update(
                provider="Google", model="gemini-2.5-flash",
                api="generate_content",
                raw_response_obj=response,
                pricing_resolver=custom_gemini_price_lookup
            )

            # Update budget
            cost = st.session_state.cost_tracker.totals['total_cost_usd']
            tokens = st.session_state.cost_tracker.totals['total_tokens']
            self.budget.consume(cost, tokens)

            st.write(f"💰 Budget consumed: ${cost:.4f}, {tokens:,} tokens (Remaining: {self.budget.left()*100:.1f}%)")

            # DEBUG: See what we got from Gemini
            with st.expander(f"🔍 Debug: Response structure for {task.id}", expanded=False):
                st.write(f"- Candidates: {len(response.candidates)}")

                if response.candidates:
                    parts = response.candidates[0].content.parts
                    st.write(f"- Parts: {len(parts)}")

                    for i, part in enumerate(parts):
                        if part.text:
                            st.write(f"  Part {i}: text ({len(part.text)} chars)")
                            st.text(part.text[:300])  # Show first 300 chars
                        if part.executable_code:
                            st.write(f"  Part {i}: executable_code ({len(part.executable_code.code)} chars)")
                            st.code(part.executable_code.code[:500], language='python')  # Show first 500 chars
                        if part.code_execution_result:
                            st.write(f"  Part {i}: code_execution_result")
                            st.text(part.code_execution_result.output[:300])

            # Extract outputs - handle BOTH executable_code and markdown-wrapped code
            code = None
            gemini_reported_accuracy = 0.0
            execution_output = ""

            for part in response.candidates[0].content.parts:
                # Method 1: Direct executable_code field (preferred)
                if part.executable_code:
                    code = part.executable_code.code
                    st.success(f"✅ Task {task.id}: Generated {len(code)} chars (executable_code)")

                # Method 2: Code in text (markdown wrapped) - CRITICAL FIX
                if part.text and not code:
                    text = part.text

                    # Extract code from markdown blocks
                    code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        st.success(f"✅ Task {task.id}: Extracted {len(code)} chars from markdown")

                # Get execution output
                if part.code_execution_result:
                    execution_output = part.code_execution_result.output or ""
                    match = re.search(r"Accuracy:\s*(\d+\.\d+)", execution_output)
                    if match:
                        gemini_reported_accuracy = float(match.group(1))

            # CRITICAL: Validate extracted code
            if not code:
                # Show what we got for debugging
                all_text = " ".join(p.text for p in response.candidates[0].content.parts if p.text)
                st.error(f"❌ Task {task.id}: No code found in response")
                st.text(all_text[:500])
                return {
                    "task_id": task.id,
                    "code": None,
                    "accuracy": 0.0,
                    "error": "No code extracted from Gemini response"
                }

            # Validate code syntax
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                st.error(f"❌ Task {task.id}: Invalid Python syntax in generated code")
                st.code(code[:500], language='python')
                st.text(f"Syntax error: {e}")
                return {
                    "task_id": task.id,
                    "code": None,
                    "accuracy": 0.0,
                    "error": f"Invalid Python syntax: {e}"
                }

            # Check for required function (predict_tools)
            if 'def predict_tools' not in code:
                st.warning(f"⚠️ Task {task.id}: Code missing 'predict_tools' function")
                st.code(code[:500], language='python')
                # Don't fail completely - let execution try anyway
                # return {
                #     "task_id": task.id,
                #     "code": None,
                #     "accuracy": 0.0,
                #     "error": "Missing predict_tools function"
                # }

                # Return failed result
                return {
                    "task_id": task.id,
                    "code": None,
                    "accuracy": 0.0,
                    "error": "No code extracted from response",
                    "response": response
                }

            # Return code for map-reduce evaluation
            # (Accuracy will be measured by testing on ALL batches in run_turn_with_map_reduce)
            st.success(f"✅ Code extraction complete - ready for map-reduce evaluation")

            if execution_output:
                st.write(f"💬 Gemini output: {execution_output[:150]}...")

            return {"code": code, "accuracy": 0.0, "response": response, "output": execution_output}

        except Exception as e:
            st.error(f"❌ Task {task.id} execution failed: {e}")
            st.exception(e)
            return {"code": None, "accuracy": 0.0, "error": str(e)}

    async def verify(self, outputs: Dict[str, Any]) -> VerificationResult:
        """Formal verification: claims + evidence → verdict"""
        code = outputs.get("code")
        accuracy = outputs.get("accuracy", 0.0)
        task_id = outputs.get("task_id", "unknown")

        # Handle error cases
        if outputs.get("error"):
            return VerificationResult(
                task_id=task_id,
                claims=["Task failed with error"],
                evidence={"error": outputs["error"]},
                verdict="failed",
                confidence=0.0,
                outputs=outputs
            )

        if not code:
            return VerificationResult(
                task_id=task_id,
                claims=["No code generated"],
                evidence={},
                verdict="failed",
                confidence=0.0,
                outputs=outputs
            )

        # Generate claims
        claims = [
            f"Code executes without errors",
            f"Achieves accuracy of {accuracy:.3f}",
            f"Handles all test cases"
        ]

        # Collect evidence
        evidence = {
            "accuracy": accuracy,
            "code_length": len(code),
            "improvement": accuracy - self.best_performance
        }

        # Determine verdict
        if accuracy > self.best_performance:
            verdict = "verified"
            confidence = min(1.0, accuracy)
        elif accuracy >= self.best_performance * 0.95:
            verdict = "partial"
            confidence = accuracy
        else:
            verdict = "failed"
            confidence = accuracy

        return VerificationResult(
            task_id=task_id,
            claims=claims,
            evidence=evidence,
            verdict=verdict,
            confidence=confidence,
            outputs=outputs
        )

    def record_turn(self, tasks_attempted: int, tasks_verified: int, current_best: float):
        """Record metrics for current turn"""
        turn = self.budget.current_turn
        improvement = current_best - self.best_performance if self.turn_history else current_best

        metrics = TurnMetrics(
            turn=turn,
            tasks_attempted=tasks_attempted,
            tasks_verified=tasks_verified,
            best_accuracy=current_best,
            improvement=improvement,
            cost_spent=self.budget.spent_cost,
            tokens_used=self.budget.spent_tokens,
            timestamp=time.time()
        )

        self.turn_history.append(metrics)

        if current_best > self.best_performance:
            self.best_performance = current_best
            self.best_turn = turn

    def converged_by_turns(self) -> bool:
        """Check convergence based on turn history (for turn mode)"""
        if len(self.turn_history) < 3:
            return False

        # Check last 3 turns for improvements
        recent = self.turn_history[-3:]
        improvements = [t.improvement for t in recent]

        # No improvement in last 3 turns (with small tolerance)
        if all(abs(imp) <= 0.001 for imp in improvements):
            st.info("Converged: No improvement in last 3 turns")
            return True

        return False

    def converged(self) -> bool:
        """Check if marginal value is below threshold (for cost mode)"""
        if len(self.turn_history) < 3:
            return False  # Minimum iterations

        # Check recent improvement rate from turn history (more reliable than index)
        recent = self.turn_history[-3:]
        improvements = [t.improvement for t in recent]

        # Average improvement over last 3 turns
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Converged if average improvement is negligible
        if avg_improvement < 0.01:
            st.info(f"Converged: Avg improvement {avg_improvement:.4f} < 0.01")
            return True

        return False

    def reflect_and_update_policies(self):
        """Update policies based on what worked"""
        # Analyze successful tasks
        verified = [
            e for e in self.index.entries
            if e["verdict"] == "verified"
        ]

        # Update policy weights for successful task types
        for entry in verified[-3:]:  # Last 3 successes
            task_prefix = entry["task_id"][:20]
            self.policies[task_prefix] += 0.1  # Boost successful patterns

        st.info(f"Policy update: {len(verified)} verified tasks inform future prioritization")

    async def test_code_on_batch(self, code: str, batch_data: List[Dict], batch_num: int) -> Dict:
        """Test code on one batch and return results with safety checks."""
        try:
            # SECURITY: Validate code syntax before execution
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return {
                    "batch_num": batch_num,
                    "error": f"Invalid Python syntax: {e}",
                    "accuracy": 0.0,
                    "predictions": [],
                    "correct": 0,
                    "total": len(batch_data)
                }

            namespace = {"test_data": batch_data, "json": json, "re": re}

            # SECURITY: Execute with timeout
            import signal
            import threading

            exec_exception = None

            def exec_with_exception():
                nonlocal exec_exception
                try:
                    exec(code, namespace)
                except Exception as e:
                    exec_exception = e

            # Use threading for cross-platform timeout
            exec_thread = threading.Thread(target=exec_with_exception)
            exec_thread.daemon = True
            exec_thread.start()
            exec_thread.join(timeout=15.0)  # 15 second timeout for batch

            if exec_thread.is_alive():
                return {
                    "batch_num": batch_num,
                    "error": "Code execution timeout (15 seconds)",
                    "accuracy": 0.0,
                    "predictions": [],
                    "correct": 0,
                    "total": len(batch_data)
                }

            if exec_exception:
                raise exec_exception

            # Check if predict_tools exists
            if 'predict_tools' not in namespace:
                return {
                    "batch_num": batch_num,
                    "error": "predict_tools function not found in code",
                    "accuracy": 0.0,
                    "predictions": [],
                    "correct": 0,
                    "total": len(batch_data)
                }

            predict_tools = namespace['predict_tools']

            # Validate it's callable
            if not callable(predict_tools):
                return {
                    "batch_num": batch_num,
                    "error": "predict_tools is not a function",
                    "accuracy": 0.0,
                    "predictions": [],
                    "correct": 0,
                    "total": len(batch_data)
                }

            # Generate predictions
            predictions = []
            for item in batch_data:
                try:
                    pred = predict_tools(item.get('query', ''))
                    # Ensure pred is a list
                    if not isinstance(pred, list):
                        pred = [pred] if pred else []
                    predictions.append({'predicted_sequence': pred})
                except Exception as e:
                    # Log individual prediction failures but continue
                    predictions.append({'predicted_sequence': []})

            # Calculate batch accuracy
            correct = 0
            for p, e in zip(predictions, batch_data):
                pred_seq = p.get('predicted_sequence', [])
                exp_seq = e.get('expected_sequence', [])

                # Normalize both to lists of strings
                if isinstance(pred_seq, list):
                    pred_seq = [str(x).strip() for x in pred_seq]
                else:
                    pred_seq = []

                if isinstance(exp_seq, list):
                    exp_seq = [str(x).strip() for x in exp_seq]
                else:
                    exp_seq = []

                if pred_seq == exp_seq:
                    correct += 1

            return {
                "batch_num": batch_num,
                "predictions": predictions,
                "accuracy": correct / len(batch_data) if batch_data else 0.0,
                "correct": correct,
                "total": len(batch_data)
            }
        except Exception as e:
            return {
                "batch_num": batch_num,
                "error": str(e),
                "accuracy": 0.0,
                "predictions": [],
                "correct": 0,
                "total": len(batch_data)
            }

    async def run_turn_with_map_reduce(self, turn: int, batch_size: int = 20) -> float:
        """Run one turn by processing ALL data in batches (map-reduce), then merging results."""

        turn_id = f"turn_{turn}"
        test_name = "Test 5"

        # Emit turn start event
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="start",
                agent_id=turn_id,
                agent_name=f"Turn {turn}",
                agent_type="main_agent",
                total_examples=len(self.test_data),
                batch_size=batch_size
            )

        st.markdown(f"### Turn {turn}: Processing {len(self.test_data)} examples in batches of {batch_size}")

        # Step 1: Divide data into batches
        batches = []
        for i in range(0, len(self.test_data), batch_size):
            batches.append(self.test_data[i:i + batch_size])

        st.write(f"📦 Split into {len(batches)} batches")

        # Step 2: Generate code using first batch as sample
        gen_id = f"{turn_id}_generation"
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="start",
                agent_id=gen_id,
                agent_name="Code Generation",
                agent_type="sub_agent",
                parent=turn_id
            )

        context = {"verified_count": len(self.cache.verified_tasks), "best_perf": self.best_performance}
        tasks = await self.decompose(self.goal, context)

        # CRITICAL FIX: Handle empty tasks list
        if not tasks or len(tasks) == 0:
            st.error("❌ No tasks generated from decompose. Creating fallback task.")
            tasks = [Task(
                id=f"fallback_turn_{turn}",
                goal="Generate predict_tools function for tool sequence prediction",
                novelty_score=0.8,
                estimated_roi=0.9
            )]

        st.write(f"✅ Decomposed into {len(tasks)} tasks")

        st.write(f"⚙️ Generating code using batch 1 of {len(batches)} as sample...")
        result = await self.dispatch_sub_agent(tasks[0])

        generated_code = result.get("code", None)

        if not generated_code:
            st.error("❌ No code generated")
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="error",
                    agent_id=gen_id,
                    agent_name="Code Generation",
                    agent_type="sub_agent",
                    error="No code generated"
                )
                self.tracker.emit(
                    test_name=test_name,
                    event_type="error",
                    agent_id=turn_id,
                    agent_name=f"Turn {turn}",
                    agent_type="main_agent",
                    error="Code generation failed"
                )
            return 0.0

        st.success(f"✅ Code generated! ({len(generated_code)} chars)")
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="complete",
                agent_id=gen_id,
                agent_name="Code Generation",
                agent_type="sub_agent",
                output_size=len(generated_code),
                success=True
            )

        # Security audit: Check generated code
        if self.security_agent:
            is_safe, status, warnings = self.security_agent.audit_code(
                code=generated_code,
                agent="CodeGenerator",
                turn=turn
            )

            if not is_safe:
                st.warning(f"⚠️ Security warnings detected: {', '.join(warnings)}")

        # Memory: Store generated code in archival memory
        if self.memory_manager:
            self.memory_manager.insert_archival(
                content=f"Turn {turn} generated code: {generated_code[:200]}...",
                tags=["code_generation", f"turn_{turn}"],
                source_agent="CodeGenerator",
                turn=turn
            )

        # Step 3: MAP - Test code on ALL batches in parallel
        eval_id = f"{turn_id}_evaluation"
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="start",
                agent_id=eval_id,
                agent_name="Batch Evaluation",
                agent_type="sub_agent",
                parent=turn_id,
                batch_count=len(batches)
            )

        st.write(f"🔍 Testing code on all {len(batches)} batches in parallel...")

        # Add progress indicator for batch processing
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.text(f"Processing 0/{len(batches)} batches...")

        # Create tasks for parallel execution
        batch_tasks = [
            self.test_code_on_batch(generated_code, batch, i)
            for i, batch in enumerate(batches)
        ]

        # Execute all batches in parallel with progress updates
        # Note: asyncio.gather doesn't provide incremental progress, so we show indeterminate progress
        status_text.text(f"Processing {len(batches)} batches in parallel...")
        progress_bar.progress(0.5)  # Show we're in progress

        batch_results = await asyncio.gather(*batch_tasks)

        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed {len(batches)} batches")

        # Step 4: REDUCE - Combine results with validation
        all_predictions = []
        total_correct = 0
        total_examples = 0
        errors = []

        for result in batch_results:
            # Validate result structure
            if not isinstance(result, dict):
                errors.append(f"Batch returned invalid result type: {type(result).__name__}")
                continue

            # Check for errors first
            if result.get("error"):
                batch_num = result.get("batch_num", "unknown")
                errors.append(f"Batch {batch_num}: {result['error']}")
                continue

            # Validate required keys exist
            required_keys = ["predictions", "correct", "total", "accuracy", "batch_num"]
            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                batch_num = result.get("batch_num", "unknown")
                errors.append(f"Batch {batch_num}: Missing keys {missing_keys}")
                continue

            # Validate predictions is a list
            if not isinstance(result["predictions"], list):
                errors.append(f"Batch {result['batch_num']}: predictions is not a list")
                continue

            # All validations passed - accumulate results
            all_predictions.extend(result["predictions"])
            total_correct += result["correct"]
            total_examples += result["total"]
            st.write(f"  ✓ Batch {result['batch_num']}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")

        if errors:
            st.error(f"⚠️ {len(errors)} batch(es) failed:")
            for err in errors[:5]:  # Show first 5 errors
                st.warning(f"  • {err}")
            if len(errors) > 5:
                st.warning(f"  ... and {len(errors) - 5} more errors")

        # Calculate overall accuracy
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        st.metric(f"Turn {turn} Overall Accuracy", f"{overall_accuracy:.3f}",
                 delta=f"+{overall_accuracy - self.best_performance:.3f}" if overall_accuracy > self.best_performance else None,
                 help=f"Tested on all {total_examples} examples")

        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="complete",
                agent_id=eval_id,
                agent_name="Batch Evaluation",
                agent_type="sub_agent",
                accuracy=overall_accuracy,
                examples_tested=total_examples,
                batches_processed=len(batches)
            )

        # Step 5: Update best if improved
        if overall_accuracy > self.best_performance:
            improvement = overall_accuracy - self.best_performance
            self.best_performance = overall_accuracy
            self.best_code = generated_code
            self.best_turn = turn
            st.success(f"🎉 **New best!** {self.best_performance:.3f} (+{improvement:.3f})")

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=turn_id,
                    agent_name=f"Turn {turn}",
                    agent_type="main_agent",
                    accuracy=overall_accuracy,
                    is_best=True,
                    improvement=improvement
                )
        else:
            st.info(f"No improvement: {overall_accuracy:.3f} (best remains {self.best_performance:.3f})")

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=turn_id,
                    agent_name=f"Turn {turn}",
                    agent_type="main_agent",
                    accuracy=overall_accuracy,
                    is_best=False,
                    improvement=0.0
                )

        # Log memory snapshot at end of turn
        if self.memory_manager:
            self.memory_manager.get_snapshot(turn=turn)

        # Log Gantt task for this turn
        if self.dashboard_logger:
            self.dashboard_logger.log_gantt_task(
                task_id=turn_id,
                agent=f"Turn {turn}",
                agent_type="main_agent",
                start_time=time.time() - 60,  # Approximate (would need actual start time)
                end_time=time.time(),
                status="complete",
                accuracy=overall_accuracy,
                is_best=overall_accuracy > self.best_performance
            )

        return overall_accuracy

    async def run_inference_mode(self):
        """Direct inference mode: Pattern matching for prediction/classification tasks."""
        st.subheader("Direct Inference Mode: Iterative Refinement (Map-Reduce Batching)")

        for turn in range(1, self.budget.max_turns + 1):
            if self.budget.mode == "turns":
                self.budget.advance_turn()

            st.markdown(f"## Turn {turn} (Budget: {self.budget.left()*100:.1f}% remaining)")

            # Show current state
            if self.best_performance > 0:
                failures = int((1 - self.best_performance) * len(self.test_data))
                st.write(f"📊 Current best: {self.best_performance:.3f} ({failures} failures out of {len(self.test_data)} examples)")
            else:
                st.write(f"📊 Turn 1: Generating initial solution for {len(self.test_data)} examples")

            # Run turn with map-reduce batching
            accuracy = await self.run_turn_with_map_reduce(turn, batch_size=20)

            # Track metrics
            self.record_turn(
                tasks_attempted=1,
                tasks_verified=1 if accuracy > 0 else 0,
                current_best=accuracy
            )

            # Check convergence
            if self.converged_by_turns():
                st.info("🏁 Converged: no improvement in last 3 turns")
                break

        return self.best_code, self.best_performance, self.turn_history

    # ============================================
    # MODE 3: Research Tasks (Decomposition)
    # ============================================

    async def decompose_research(self, goal: str, context: Dict) -> List[Task]:
        """Break research goal into parallel information gathering tasks."""

        iteration = len(self.cache.verified_tasks)

        prompt = f"""
Break this research goal into 3-5 independent subtasks:

GOAL: {goal}

CONTEXT:
- Completed tasks: {iteration}
- Knowledge gathered: {len(self.index.entries)} entries

Create concrete, parallelizable subtasks. Each should:
- Gather specific information from distinct sources
- Be independently executable
- Contribute to answering the main goal

Examples for "Research George Morgan, Symbolica AI, and their AI Engineering position":
- Task 1: Search for George Morgan's LinkedIn profile and role
- Task 2: Find Symbolica AI company information and funding history
- Task 3: Search for recent statements on AI Engineering from Symbolica
- Task 4: Locate news about their latest funding round
- Task 5: Find technical blog posts or papers from the team

Return JSON:
[
  {{
    "id": "research_1",
    "goal": "Search LinkedIn for George Morgan at Symbolica AI",
    "source_type": "linkedin",
    "depends_on": [],
    "novelty": 0.9
  }},
  ...
]
"""

        model = st.session_state.get('judge_model', 'openai/gpt-5-mini')
        provider = _get_provider_from_model_id(model)

        if provider == "openai" and OPENAI_API_KEY:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            native_model = _to_native_model_id(model)
            resp = await client.chat.completions.create(
                model=native_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
        else:
            # OpenRouter fallback
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            body = {
                "model": _to_openrouter_model_id(model),
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
            async with httpx.AsyncClient(timeout=60) as client_http:
                resp = await client_http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=body
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

        tasks_data = json.loads(content)
        if isinstance(tasks_data, dict):
            tasks_data = tasks_data.get("subtasks", tasks_data.get("tasks", []))

        tasks = []
        for t in tasks_data:
            task = Task(
                id=t["id"],
                goal=t["goal"],
                parent_ids=set(t.get("depends_on", [])),
                novelty_score=float(t.get("novelty", 0.5)),
                estimated_roi=float(t.get("roi", 0.8))
            )
            tasks.append(task)

        return tasks

    async def execute_research_task(self, task: Task) -> Dict:
        """Execute a research task using web search or other tools."""

        st.write(f"🔍 Researching: {task.goal}")

        # Use Gemini for research
        prompt = f"""
Research this specific question: {task.goal}

Provide:
- Key findings (3-5 bullet points)
- Sources found
- Confidence level (0-1)

Return JSON: {{"findings": [...], "sources": [...], "confidence": 0.8}}
"""

        # CRITICAL FIX: Create client without async context manager
        # genai.Client doesn't support async context managers
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
        )

        result = json.loads(response.text)
        return {
            "task_id": task.id,
            "findings": result.get("findings", []),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.5)
        }

    async def run_research_mode(self):
        """Execute research workflow with task decomposition."""

        st.header("Research Mode: Multi-Source Information Gathering")
        test_name = "Test 5"

        all_findings = []

        for turn in range(1, self.budget.max_turns + 1):
            turn_id = f"research_turn_{turn}"

            # Emit turn start
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=turn_id,
                    agent_name=f"Research Turn {turn}",
                    agent_type="research_turn"
                )

            st.subheader(f"Research Turn {turn}")

            # Decompose into research tasks
            context = {"completed": len(all_findings)}
            tasks = await self.decompose_research(self.goal, context)

            st.write(f"Generated {len(tasks)} research tasks:")
            for task in tasks:
                st.write(f"  - {task.goal}")

            # Execute tasks in parallel
            research_coros = [self.execute_research_task(task) for task in tasks]
            results = await asyncio.gather(*research_coros)

            # Store in knowledge index
            for result in results:
                await self.index.store(result, VerificationResult(
                    task_id=result['task_id'],
                    claims=result['findings'],
                    evidence={"sources": result['sources']},
                    verdict="verified",
                    confidence=result['confidence'],
                    outputs=result
                ))
                all_findings.extend(result['findings'])

            # Emit turn completion
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=turn_id,
                    agent_name=f"Research Turn {turn}",
                    agent_type="research_turn",
                    findings_count=len(all_findings),
                    tasks_completed=len(results)
                )

            # Check if we have enough information
            if len(all_findings) >= 15:  # Arbitrary threshold
                st.info("Sufficient information gathered")
                break

        # Synthesize final report
        st.divider()
        st.subheader("Research Summary")

        synthesis_prompt = f"""
Synthesize these research findings into a coherent report:

ORIGINAL GOAL: {self.goal}

FINDINGS:
{json.dumps(all_findings, indent=2)}

Create a structured report with:
- Executive summary
- Key findings organized by topic
- Sources and confidence levels
- Recommendations or next steps
"""

        # CRITICAL FIX: Create client without async context manager
        # genai.Client doesn't support async context managers
        client = genai.Client(api_key=GEMINI_API_KEY)

        synthesis = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=synthesis_prompt
            )
        )

        st.markdown(synthesis.text)

        return {
            "findings_count": len(all_findings),
            "knowledge_entries": len(self.index.entries),
            "synthesis": synthesis.text
        }

    def _initialize_stateful_components(self):
        """Initialize memory, security, and self-correction components with logging."""
        if self._stateful_components_initialized:
            return

        # Initialize dashboard logger
        if not self.dashboard_logger:
            test_id = f"test5_{self.mode}_{self.coordination_pattern}"
            dataset_info = {
                "size": len(self.test_data),
                "type": "classification" if self.test_data else "research"
            }
            configuration = {
                "mode": self.mode,
                "coordination_pattern": self.coordination_pattern,
                "budget_mode": self.budget.mode,
                "max_turns": self.budget.max_turns
            }

            self.dashboard_logger = DashboardLogger(
                test_id=test_id,
                test_type="orchestrator",
                test_name="Test 5",
                model="gemini-2.5-flash",
                dataset_info=dataset_info,
                configuration=configuration
            )

            # Store in session state for persistence
            st.session_state['test5_dashboard_logger'] = self.dashboard_logger

        # Initialize memory manager
        self.memory_manager = MemoryManager(logger=self.dashboard_logger)

        # Set purpose based on goal
        self.memory_manager.update_block(
            block_name="Purpose",
            new_content=f"My current goal is: {self.goal}",
            turn=0,
            trigger="initialization",
            change_summary="Set initial purpose from user goal"
        )

        # Override Tool Guidelines with demo-specific policy if available
        if 'demo_memory_policy' in st.session_state:
            self.memory_manager.update_block(
                block_name="Tool Guidelines",
                new_content=st.session_state['demo_memory_policy'],
                turn=0,
                trigger="DEMO_INITIALIZATION",
                change_summary=f"Loaded {st.session_state.get('demo_scenario', 'demo')} scenario policy"
            )
            st.info(f"🧠 Memory initialized with {st.session_state.get('demo_scenario', 'demo').upper()} policy")

        # Initialize security audit agent
        self.security_agent = SecurityAuditAgent(logger=self.dashboard_logger)

        # Initialize self-correction manager
        self.self_correction_manager = SelfCorrectionManager(
            memory_manager=self.memory_manager,
            logger=self.dashboard_logger
        )

        self._stateful_components_initialized = True

        # Log initial memory snapshot
        self.memory_manager.get_snapshot(turn=0)

    async def run(self):
        """Main orchestrator with mode-specific execution strategies and coordination patterns."""

        test_name = "Test 5"

        # Initialize stateful components (memory, security, self-correction)
        self._initialize_stateful_components()

        # Emit orchestrator start event
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="start",
                agent_id="orchestrator",
                agent_name="Unified Orchestrator",
                agent_type="orchestrator",
                mode=self.mode,
                coordination_pattern=self.coordination_pattern,
                goal=self.goal[:100]  # Truncate for metadata
            )

        st.info(f"🎯 Mode: **{self.mode.upper()}** | 🤝 Pattern: **{self.coordination_pattern.upper()}**")

        try:
            # Route based on coordination pattern
            if self.coordination_pattern == AgentCoordinationPattern.SOLO.value:
                # Solo execution (existing implementations)
                result = await self._run_solo()

            elif self.coordination_pattern == AgentCoordinationPattern.SUBAGENT.value:
                # Subagent orchestration (hierarchical delegation)
                result = await self.run_with_subagents()

            elif self.coordination_pattern == AgentCoordinationPattern.MULTI_AGENT.value:
                # Multi-agent collaboration (peer consensus)
                result = await self.run_with_multi_agent()

            elif self.coordination_pattern == AgentCoordinationPattern.LEAF_SCAFFOLD.value:
                # Leaf agent scaffold (hierarchical multi-agent with supervisor)
                # Get selected agent types from session state or use defaults
                selected_agent_types = st.session_state.get('selected_agent_types',
                    ["web_researcher", "code_executor", "content_generator"])
                result = await self.run_with_leaf_scaffold(selected_agent_types)

            else:
                st.error(f"Unknown coordination pattern: {self.coordination_pattern}")
                result = {}

            # Emit completion event
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id="orchestrator",
                    agent_name="Unified Orchestrator",
                    agent_type="orchestrator",
                    result=str(result)[:200]  # Truncate
                )

            # Finalize dashboard logger with summary metrics
            if self.dashboard_logger:
                summary_metrics = {
                    "final_score": result.get("final_score", 0.0) if isinstance(result, dict) else 0.0,
                    "total_turns": len(self.turn_history),
                    "best_turn": self.best_turn,
                    "total_cost": self.budget.spent_cost,
                    "total_tokens": self.budget.spent_tokens,
                    "mode": self.mode,
                    "coordination_pattern": self.coordination_pattern,
                    "memory_stats": self.memory_manager.stats if self.memory_manager else {},
                    "rethink_count": len(self.self_correction_manager.rethink_history) if self.self_correction_manager else 0
                }

                self.dashboard_logger.finalize_run(summary_metrics)

                st.success(f"✅ Run logs saved to: `{self.dashboard_logger.run_dir}`")

            return result

        except Exception as e:
            # Emit error event
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="error",
                    agent_id="orchestrator",
                    agent_name="Unified Orchestrator",
                    agent_type="orchestrator",
                    error=str(e)
                )

            # Finalize logger even on error (mark as incomplete)
            if self.dashboard_logger:
                try:
                    self.dashboard_logger.finalize_run({
                        "status": "error",
                        "error": str(e),
                        "total_turns": len(self.turn_history)
                    })
                except:
                    pass  # Don't fail on logging errors

            raise

    async def _run_solo(self):
        """Execute in solo mode (single agent, existing implementations)."""

        # Route to appropriate execution strategy based on mode
        if self.mode == "inference":
            # Inference → Prompt Optimization
            result = await self.run_inference_prompt_optimization()

        elif self.mode == "analysis":
            # Analysis → Code Generation
            result = await self.run_analysis_code_generation()

        elif self.mode == "research":
            # Research → Hybrid (decompose first, then decide per subtask)
            result = await self.run_research_hybrid()
        else:
            st.error(f"Unknown mode: {self.mode}")
            result = {}

        return result

    # ============================================
    # COORDINATION PATTERN: Subagent Orchestration
    # ============================================

    async def run_with_subagents(self):
        """
        Hierarchical delegation with specialized subagents.

        Workflow:
        1. Decomposer: Break down the goal into subtasks
        2. Generator: Generate solutions for each subtask
        3. Evaluator: Evaluate the solutions
        4. Analyzer: Analyze failures and suggest improvements
        5. Synthesizer: Combine results into final solution
        """

        st.subheader(f"Subagent Orchestration: {self.mode.upper()} Mode")
        test_name = "Test 5"

        all_results = []

        for turn in range(1, self.budget.max_turns + 1):
            st.markdown(f"## Turn {turn}: Subagent Workflow")

            # ===== SUBAGENT 1: Decomposer =====
            decomposer_id = f"decomposer_turn_{turn}"
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=decomposer_id,
                    agent_name="Decomposer",
                    agent_type="subagent"
                )

            st.write("🔍 **Decomposer**: Breaking down goal into subtasks...")
            context = {"turn": turn, "previous_results": all_results}
            subtasks = await self.decompose(self.goal, context)

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=decomposer_id,
                    agent_name="Decomposer",
                    agent_type="subagent",
                    tasks_generated=len(subtasks)
                )

            st.write(f"  → Generated {len(subtasks)} subtasks")

            # ===== SUBAGENT 2: Generator =====
            generator_id = f"generator_turn_{turn}"
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=generator_id,
                    agent_name="Generator",
                    agent_type="subagent"
                )

            st.write("⚙️ **Generator**: Creating solutions...")

            # Generate solutions based on mode
            if self.mode == "inference":
                solution = await self._generate_inference_solution(subtasks)
            elif self.mode == "analysis":
                solution = await self._generate_analysis_solution(subtasks)
            elif self.mode == "research":
                solution = await self._generate_research_solution(subtasks)
            else:
                solution = {"error": "Unknown mode"}

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=generator_id,
                    agent_name="Generator",
                    agent_type="subagent",
                    solution_generated=True
                )

            st.write("  → Solution generated")

            # ===== SUBAGENT 3: Evaluator =====
            evaluator_id = f"evaluator_turn_{turn}"
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=evaluator_id,
                    agent_name="Evaluator",
                    agent_type="subagent"
                )

            st.write("📊 **Evaluator**: Assessing solution quality...")
            evaluation = await self._evaluate_solution(solution)

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=evaluator_id,
                    agent_name="Evaluator",
                    agent_type="subagent",
                    accuracy=evaluation.get('accuracy', 0.0),
                    score=evaluation.get('score', 0.0)
                )

            st.write(f"  → Score: {evaluation.get('score', 0.0):.3f}")

            # ===== SUBAGENT 4: Analyzer =====
            analyzer_id = f"analyzer_turn_{turn}"
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=analyzer_id,
                    agent_name="Analyzer",
                    agent_type="subagent"
                )

            st.write("🔬 **Analyzer**: Analyzing failures and suggesting improvements...")
            analysis = await self._analyze_failures(solution, evaluation)

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=analyzer_id,
                    agent_name="Analyzer",
                    agent_type="subagent",
                    failures_found=len(analysis.get('failures', [])),
                    suggestions=len(analysis.get('suggestions', []))
                )

            st.write(f"  → Found {len(analysis.get('failures', []))} issues")

            # Store results
            all_results.append({
                "turn": turn,
                "solution": solution,
                "evaluation": evaluation,
                "analysis": analysis
            })

            # Check convergence
            if evaluation.get('score', 0.0) >= 0.95:
                st.success("✅ High quality solution achieved!")
                break

        # ===== SUBAGENT 5: Synthesizer =====
        synthesizer_id = "synthesizer_final"
        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="start",
                agent_id=synthesizer_id,
                agent_name="Synthesizer",
                agent_type="subagent"
            )

        st.divider()
        st.write("🎯 **Synthesizer**: Combining results into final solution...")
        final_solution = await self._synthesize_results(all_results)

        if self.tracker:
            self.tracker.emit(
                test_name=test_name,
                event_type="complete",
                agent_id=synthesizer_id,
                agent_name="Synthesizer",
                agent_type="subagent",
                final_score=final_solution.get('score', 0.0)
            )

        return final_solution

    # ============================================
    # COORDINATION PATTERN: Multi-Agent Collaboration
    # ============================================

    async def run_with_multi_agent(self):
        """
        Peer collaboration with independent proposals, cross-review, and consensus.

        Workflow:
        1. Propose: Each peer agent independently proposes a solution
        2. Review: Agents cross-review each other's proposals
        3. Synthesize: Build consensus from all proposals and reviews
        4. Evaluate: Jointly evaluate the consensus solution
        """

        st.subheader(f"Multi-Agent Collaboration: {self.mode.upper()} Mode")
        test_name = "Test 5"

        st.write(f"👥 **Peer Agents**: {', '.join(self.peer_agent_roles)}")

        all_consensus = []

        for turn in range(1, self.budget.max_turns + 1):
            st.markdown(f"## Turn {turn}: Multi-Agent Collaboration")

            # ===== ROUND 1: Propose =====
            st.write("### Round 1: Independent Proposals")
            proposals = []

            # Add progress indicator for proposals
            proposal_progress = st.progress(0.0)
            proposal_status = st.empty()

            for idx, role in enumerate(self.peer_agent_roles):
                agent_id = f"agent_{role.replace(' ', '_').lower()}_turn_{turn}_propose"

                proposal_status.text(f"💡 {role}: Generating proposal... ({idx+1}/{len(self.peer_agent_roles)})")

                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="start",
                        agent_id=agent_id,
                        agent_name=f"{role} (Propose)",
                        agent_type="peer_agent"
                    )

                proposal = await self._agent_propose(role, self.goal, all_consensus)
                proposals.append({"role": role, "proposal": proposal})

                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id=agent_id,
                        agent_name=f"{role} (Propose)",
                        agent_type="peer_agent",
                        proposal_length=len(str(proposal))
                    )

                # Update progress
                proposal_progress.progress((idx + 1) / len(self.peer_agent_roles))
                st.write(f"  ✓ **{role}**: Proposal submitted")

            proposal_status.text(f"✅ All {len(self.peer_agent_roles)} proposals received")
            proposal_progress.empty()  # Clear progress bar

            # ===== ROUND 2: Review =====
            st.write("### Round 2: Cross-Review")
            reviews = []

            # Add progress indicator for reviews
            review_progress = st.progress(0.0)
            review_status = st.empty()

            for idx, reviewer_role in enumerate(self.peer_agent_roles):
                agent_id = f"agent_{reviewer_role.replace(' ', '_').lower()}_turn_{turn}_review"

                review_status.text(f"🔍 {reviewer_role}: Reviewing proposals... ({idx+1}/{len(self.peer_agent_roles)})")

                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="start",
                        agent_id=agent_id,
                        agent_name=f"{reviewer_role} (Review)",
                        agent_type="peer_agent"
                    )

                # Review all proposals except own
                other_proposals = [p for p in proposals if p['role'] != reviewer_role]
                review = await self._agent_review(reviewer_role, other_proposals)
                reviews.append({"reviewer": reviewer_role, "review": review})

                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id=agent_id,
                        agent_name=f"{reviewer_role} (Review)",
                        agent_type="peer_agent",
                        proposals_reviewed=len(other_proposals)
                    )

                # Update progress
                review_progress.progress((idx + 1) / len(self.peer_agent_roles))
                st.write(f"  ✓ **{reviewer_role}**: Review completed")

            review_status.text(f"✅ All {len(self.peer_agent_roles)} reviews completed")
            review_progress.empty()  # Clear progress bar

            # ===== ROUND 3: Synthesize =====
            st.write("### Round 3: Consensus Building")
            synthesizer_id = f"synthesizer_turn_{turn}"

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=synthesizer_id,
                    agent_name="Consensus Builder",
                    agent_type="synthesizer"
                )

            st.write("🤝 **Consensus Builder**: Synthesizing proposals and reviews...")
            consensus = await self._build_consensus(proposals, reviews)

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=synthesizer_id,
                    agent_name="Consensus Builder",
                    agent_type="synthesizer",
                    consensus_built=True
                )

            st.write("  → Consensus achieved")

            # ===== ROUND 4: Joint Evaluation =====
            st.write("### Round 4: Joint Evaluation")
            evaluator_id = f"joint_evaluator_turn_{turn}"

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=evaluator_id,
                    agent_name="Joint Evaluator",
                    agent_type="evaluator"
                )

            st.write("📊 **Joint Evaluator**: Assessing consensus solution...")
            evaluation = await self._joint_evaluate(consensus, self.peer_agent_roles)

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=evaluator_id,
                    agent_name="Joint Evaluator",
                    agent_type="evaluator",
                    score=evaluation.get('score', 0.0),
                    agreement=evaluation.get('agreement', 0.0)
                )

            st.write(f"  → Score: {evaluation.get('score', 0.0):.3f}, Agreement: {evaluation.get('agreement', 0.0):.3f}")

            # Store consensus
            all_consensus.append({
                "turn": turn,
                "proposals": proposals,
                "reviews": reviews,
                "consensus": consensus,
                "evaluation": evaluation
            })

            # Check convergence
            if evaluation.get('score', 0.0) >= 0.95 and evaluation.get('agreement', 0.0) >= 0.9:
                st.success("✅ High quality consensus achieved!")
                break

        # Return final consensus
        return {
            "strategy": "multi_agent",
            "peer_roles": self.peer_agent_roles,
            "consensus_history": all_consensus,
            "final_consensus": all_consensus[-1]['consensus'] if all_consensus else {},
            "final_score": all_consensus[-1]['evaluation'].get('score', 0.0) if all_consensus else 0.0
        }

    # ============================================
    # COORDINATION PATTERN: Leaf Agent Scaffold
    # ============================================

    async def run_with_leaf_scaffold(self, selected_agent_types: List[str]):
        """
        Execute using hierarchical leaf agent scaffold.

        Workflow:
        1. Supervisor decomposes task into specialized sub-tasks
        2. Delegates to appropriate leaf agents
        3. Manages dependencies between sub-tasks
        4. Synthesizes final result from all agent outputs

        Args:
            selected_agent_types: List of agent type strings (e.g., ["web_researcher", "code_executor"])
        """

        try:
            st.subheader(f"🌳 Leaf Agent Scaffold: {self.mode.upper()} Mode")
            test_name = "Test 5"

            # Initialize LLM client
            if not GEMINI_API_KEY:
                st.error("❌ Gemini API key required for leaf agent scaffold")
                return {"error": "Missing API key"}

            st.write("🔧 Initializing Gemini LLM client...")
            llm_client = GeminiLLMClient(GEMINI_API_KEY)
            st.success("✅ LLM client initialized")

            # Create leaf agents based on selection
            st.write(f"🔧 Creating leaf agents from selection: {selected_agent_types}")
            leaf_agents = []

            agent_type_map = {
                "web_researcher": (AgentType.WEB_RESEARCHER, WebResearchAgent),
                "code_executor": (AgentType.CODE_EXECUTOR, CodeExecutorAgent),
                "content_generator": (AgentType.CONTENT_GENERATOR, ContentGeneratorAgent),
                "validator": (AgentType.VALIDATOR, ValidatorAgent)
            }

            for agent_type_str in selected_agent_types:
                if agent_type_str in agent_type_map:
                    agent_type, agent_class = agent_type_map[agent_type_str]
                    st.write(f"  • Creating {agent_type_str}...")
                    agent = agent_class(llm_client)
                    leaf_agents.append(agent)
                    st.write(f"    ✅ {agent.name} created")

            if not leaf_agents:
                st.error("❌ No valid leaf agents selected")
                return {"error": "No valid agents"}

            st.success(f"✅ Created {len(leaf_agents)} leaf agents: {', '.join([a.name for a in leaf_agents])}")

            # Create supervisor with custom planner and synthesizer
            st.write("🔧 Creating supervisor agent...")
            supervisor = SupervisorAgent(leaf_agents, memory_manager=self.memory_manager)
            supervisor.task_planner = GeminiTaskPlanner(
                available_agents=[a.agent_type for a in leaf_agents],
                llm_client=llm_client
            )
            supervisor.result_synthesizer = GeminiResultSynthesizer(llm_client)
            st.success("✅ Supervisor agent created with policy-based memory")

            # Track supervisor start
            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id="supervisor",
                    agent_name="Supervisor Agent",
                    agent_type="supervisor"
                )

            # ===== STEP 1: Task Planning =====
            st.markdown("### 📋 Task Planning & Decomposition")

            planning_status = st.empty()
            planning_status.text("🧠 Supervisor analyzing task and decomposing into sub-tasks...")

            try:
                sub_tasks = await supervisor.task_planner.decompose(self.goal, self.mode)
                planning_status.empty()

                st.success(f"✅ Decomposed into {len(sub_tasks)} specialized sub-tasks")

                # Display sub-tasks
                with st.expander("📝 View Sub-Task Breakdown", expanded=True):
                    for idx, task in enumerate(sub_tasks, 1):
                        agent_emoji = {
                            AgentType.WEB_RESEARCHER: "🔍",
                            AgentType.CODE_EXECUTOR: "💻",
                            AgentType.CONTENT_GENERATOR: "✍️",
                            AgentType.VALIDATOR: "✅"
                        }.get(task.agent_type, "🤖")

                        st.markdown(f"**Sub-Task {idx}** {agent_emoji} `{task.agent_type.value}`")
                        st.write(f"  → {task.description}")
                        if task.dependencies:
                            st.caption(f"  Dependencies: {', '.join(task.dependencies)}")
                        st.divider()

            except Exception as e:
                planning_status.empty()
                st.error(f"❌ Task planning failed: {str(e)}")
                return {"error": f"Task planning failed: {str(e)}"}

            st.divider()

            # ===== STEP 2: Execute Sub-Tasks =====
            st.markdown("### 🚀 Executing Sub-Tasks")

            results = []

            # Add overall progress
            overall_progress = st.progress(0.0)
            overall_status = st.empty()

            for idx, sub_task in enumerate(sub_tasks, 1):
                agent_emoji = {
                    AgentType.WEB_RESEARCHER: "🔍",
                    AgentType.CODE_EXECUTOR: "💻",
                    AgentType.CONTENT_GENERATOR: "✍️",
                    AgentType.VALIDATOR: "✅"
                }.get(sub_task.agent_type, "🤖")

                st.markdown(f"#### {agent_emoji} Sub-Task {idx}/{len(sub_tasks)}: {sub_task.agent_type.value}")

                overall_status.text(f"Executing: {sub_task.description}...")

                # Track agent start
                agent_id = f"leaf_agent_{sub_task.agent_type.value}_{sub_task.id}"
                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="start",
                        agent_id=agent_id,
                        agent_name=f"{sub_task.agent_type.value}",
                        agent_type="leaf_agent",
                        sub_task_id=sub_task.id
                    )

                with st.spinner(f"Executing: {sub_task.description}..."):
                    result = await supervisor.execute_single_task(sub_task)

                # Track agent completion
                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id=agent_id,
                        agent_name=f"{sub_task.agent_type.value}",
                        agent_type="leaf_agent",
                        sub_task_id=sub_task.id,
                        success=result.success
                    )

                if result.success:
                    st.success(f"✅ {result.agent_name} completed successfully")
                    with st.expander(f"📄 View Output from {result.agent_name}"):
                        st.write(result.output)
                        if result.metadata:
                            st.caption(f"Metadata: {result.metadata}")
                else:
                    st.error(f"❌ {result.agent_name} failed: {result.error}")

                results.append(result)

                # ===== SELF-CORRECTING RESEARCH PIPELINE: Policy Update Logic =====
                # Process validation results and update memory with policy constraints
                if result.success and sub_task.agent_type == AgentType.VALIDATOR and self.memory_manager:
                    validation_result = result.metadata.get("validation_result")

                    if validation_result:
                        try:
                            # Convert to Pydantic schema for type-safe processing
                            artifact = convert_validation_to_artifact(validation_result)

                            # Display validation summary
                            st.markdown("#### 🔍 Validation Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                verdict_emoji = "✅" if artifact.final_verdict == "Verified" else "⚠️"
                                st.metric("Verdict", f"{verdict_emoji} {artifact.final_verdict}")
                            with col2:
                                confidence_color = "normal" if artifact.confidence_score >= 0.7 else "inverse"
                                st.metric("Confidence", f"{artifact.confidence_score:.2%}",
                                         delta=None if artifact.confidence_score >= 0.7 else "Low confidence",
                                         delta_color=confidence_color)

                            # Show red flags if any
                            if artifact.red_flags:
                                with st.expander(f"⚠️ {len(artifact.red_flags)} Red Flag(s) Detected", expanded=True):
                                    for flag in artifact.red_flags:
                                        risk = flag.get("hallucination_risk", 0.0)
                                        risk_emoji = "🔴" if risk >= 0.8 else "🟡" if risk >= 0.5 else "🟢"
                                        st.markdown(f"{risk_emoji} **Claim:** {flag.get('claim', 'N/A')}")
                                        st.caption(f"Risk: {risk:.1%} | Reason: {flag.get('reason', 'N/A')} | Sources: {flag.get('sources_found', 0)}")
                                        st.divider()

                            # Use existing PolicyUpdater to process validation result
                            policy_updated = PolicyUpdater.process_validation_result(
                                validation_result,
                                self.memory_manager,
                                turn=idx
                            )

                            if policy_updated:
                                st.success(f"🧠 **MEMORY UPDATED!** Added {len(artifact.policy_updates)} constraint(s) to Tool Guidelines.")

                                # Display structured policy updates with severity
                                with st.expander("📋 View New Policy Constraints", expanded=True):
                                    for constraint in artifact.policy_updates:
                                        severity_emoji = {
                                            "CRITICAL": "🔴",
                                            "HIGH": "🟠",
                                            "MEDIUM": "🟡",
                                            "LOW": "🟢"
                                        }.get(constraint.severity, "⚪")

                                        st.markdown(f"{severity_emoji} **[{constraint.severity}]** {constraint.constraint_text}")
                                        st.caption(f"ID: `{constraint.constraint_id}` | Priority: {constraint.priority:.1%}")
                                        st.divider()

                                # Log to dashboard if available
                                if self.dashboard_logger:
                                    self.dashboard_logger.log_rethink_event(
                                        turn=idx,
                                        block_name="Tool Guidelines",
                                        trigger="VALIDATION_FAILURE_POLICY_UPDATE",
                                        change_summary=f"Added {len(artifact.policy_updates)} constraint(s) from validation",
                                        old_content_preview="[See Memory Inspector]",
                                        new_content_preview=f"Added constraints: {', '.join([c.constraint_id for c in artifact.policy_updates[:2]])}..."
                                    )
                            else:
                                # Validation passed or no policy updates needed
                                if artifact.confidence_score >= 0.7:
                                    st.info(f"✅ Validation passed (confidence: {artifact.confidence_score:.2%}). No policy updates needed.")

                        except Exception as e:
                            st.error(f"❌ Policy update failed: {e}")
                            st.exception(e)

                # Update progress
                overall_progress.progress((idx) / len(sub_tasks))

            overall_status.text(f"✅ All {len(sub_tasks)} sub-tasks completed")
            overall_progress.empty()

            st.divider()

            # ===== STEP 3: Synthesize Results =====
            st.markdown("### 🎯 Synthesizing Final Result")

            synthesis_status = st.empty()
            synthesis_status.text("🧠 Supervisor synthesizing results from all agents...")

            try:
                final_result = await supervisor.result_synthesizer.synthesize(
                    self.goal,
                    results
                )
                synthesis_status.empty()

                st.success("✅ Synthesis complete!")

                # Track supervisor completion
                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id="supervisor",
                        agent_name="Supervisor Agent",
                        agent_type="supervisor",
                        total_sub_tasks=len(sub_tasks),
                        successful_tasks=final_result.metadata.get('successful_tasks', 0)
                    )

            except Exception as e:
                synthesis_status.empty()
                st.error(f"❌ Result synthesis failed: {str(e)}")
                return {"error": f"Synthesis failed: {str(e)}"}

            # Display final result
            st.markdown("### 📊 Final Synthesized Result")
            st.markdown(final_result.answer)

            # Show execution metadata
            with st.expander("📈 Execution Metadata"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Agents", len(final_result.contributing_agents))
                with col2:
                    st.metric("Successful Tasks", final_result.metadata.get('successful_tasks', 0))
                with col3:
                    st.metric("Failed Tasks", final_result.metadata.get('failed_tasks', 0))

                st.json(final_result.metadata)

            # Show contributing agents
            with st.expander("👥 Contributing Agents"):
                for agent_name in final_result.contributing_agents:
                    st.write(f"  • {agent_name}")

            return {
                "strategy": "leaf_scaffold",
                "leaf_agents": [a.name for a in leaf_agents],
                "sub_tasks": len(sub_tasks),
                "final_result": final_result.answer,
                "metadata": final_result.metadata
            }

        except Exception as e:
            st.error(f"❌ Error in leaf agent scaffold execution: {str(e)}")
            st.exception(e)
            return {"error": str(e)}

    # ============================================
    # Helper Methods for Subagent Orchestration
    # ============================================

    async def _generate_inference_solution(self, subtasks: List[Task]) -> Dict:
        """Generate solution for inference mode (subagent pattern)."""
        # Delegate to existing inference logic
        best_code, best_accuracy, history = await self.run_inference_mode()
        return {
            "type": "inference",
            "code": best_code,
            "accuracy": best_accuracy,
            "history": history
        }

    async def _generate_analysis_solution(self, subtasks: List[Task]) -> Dict:
        """Generate solution for analysis mode (subagent pattern)."""
        # Delegate to existing analysis logic
        best_code, best_perf, history = await self.run_inference_mode()
        return {
            "type": "analysis",
            "code": best_code,
            "performance": best_perf,
            "history": history
        }

    async def _generate_research_solution(self, subtasks: List[Task]) -> Dict:
        """Generate solution for research mode (subagent pattern)."""
        # Execute research tasks
        results = await asyncio.gather(*[
            self.execute_research_task_smart(task) for task in subtasks
        ])
        return {
            "type": "research",
            "results": results,
            "findings_count": sum(len(r.get('findings', [])) for r in results)
        }

    async def _evaluate_solution(self, solution: Dict) -> Dict:
        """Evaluate solution quality."""
        if solution.get('type') == 'inference':
            return {
                "score": solution.get('accuracy', 0.0),
                "accuracy": solution.get('accuracy', 0.0),
                "metric": "accuracy"
            }
        elif solution.get('type') == 'analysis':
            return {
                "score": solution.get('performance', 0.0),
                "performance": solution.get('performance', 0.0),
                "metric": "performance"
            }
        elif solution.get('type') == 'research':
            findings_count = solution.get('findings_count', 0)
            score = min(findings_count / 15.0, 1.0)  # Normalize to 0-1
            return {
                "score": score,
                "findings_count": findings_count,
                "metric": "completeness"
            }
        else:
            return {"score": 0.0, "metric": "unknown"}

    async def _analyze_failures(self, solution: Dict, evaluation: Dict) -> Dict:
        """Analyze failures and suggest improvements."""
        failures = []
        suggestions = []

        if evaluation.get('score', 0.0) < 0.9:
            failures.append(f"Score below threshold: {evaluation.get('score', 0.0):.3f}")
            suggestions.append("Consider refining the approach or adding more iterations")

        return {
            "failures": failures,
            "suggestions": suggestions,
            "needs_refinement": len(failures) > 0
        }

    async def _synthesize_results(self, all_results: List[Dict]) -> Dict:
        """Synthesize all turn results into final solution."""
        if not all_results:
            return {"score": 0.0, "solution": None}

        # Get best result by score
        best_result = max(all_results, key=lambda r: r['evaluation'].get('score', 0.0))

        return {
            "strategy": "subagent",
            "best_turn": best_result['turn'],
            "solution": best_result['solution'],
            "score": best_result['evaluation'].get('score', 0.0),
            "total_turns": len(all_results)
        }

    # ============================================
    # Helper Methods for Multi-Agent Collaboration
    # ============================================

    async def _agent_propose(self, role: str, goal: str, previous_consensus: List[Dict]) -> Dict:
        """Agent proposes a solution from their perspective with robust error handling."""

        # Build context from previous consensus
        context = ""
        if previous_consensus:
            context = f"\n\nPrevious consensus (Turn {len(previous_consensus)}): {previous_consensus[-1].get('consensus', {})}"

        prompt = f"""You are a {role} agent. Propose a solution for this goal:

GOAL: {goal}

MODE: {self.mode}

Your perspective as {role}:
- Focus on your domain expertise
- Propose concrete, actionable solutions
- Consider trade-offs from your viewpoint
{context}

Return JSON with your proposal:
{{
    "approach": "Your proposed approach",
    "rationale": "Why this approach is good from your perspective",
    "key_points": ["Point 1", "Point 2", "Point 3"]
}}
"""

        # Default fallback response
        fallback_response = {
            "approach": f"Proposal from {role}",
            "rationale": "Unable to generate proposal",
            "key_points": []
        }

        if not GEMINI_API_KEY:
            st.warning(f"⚠️ {role}: No Gemini API key available")
            fallback_response["rationale"] = "API key not set"
            return fallback_response

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            # Track cost
            if st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model="gemini-2.5-flash",
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            # Parse and validate response
            try:
                proposal = json.loads(response.text)

                # Validate required fields
                required_fields = ["approach", "rationale", "key_points"]
                if not all(field in proposal for field in required_fields):
                    st.warning(f"⚠️ {role}: Incomplete proposal, using fallback")
                    # Fill in missing fields
                    for field in required_fields:
                        if field not in proposal:
                            proposal[field] = fallback_response[field]

                # Validate key_points is a list
                if not isinstance(proposal.get("key_points"), list):
                    proposal["key_points"] = []

                return proposal

            except json.JSONDecodeError as e:
                st.error(f"❌ {role}: Failed to parse JSON response: {e}")
                fallback_response["rationale"] = f"JSON parse error: {str(e)[:100]}"
                return fallback_response

        except Exception as e:
            st.error(f"❌ {role}: Proposal generation failed: {e}")
            fallback_response["rationale"] = f"Error: {str(e)[:100]}"
            return fallback_response

    async def _agent_review(self, reviewer_role: str, proposals: List[Dict]) -> Dict:
        """Agent reviews other proposals with robust error handling."""

        # Default fallback response
        fallback_response = {
            "strengths": [],
            "concerns": [],
            "suggestions": []
        }

        try:
            proposals_text = "\n\n".join([
                f"Proposal from {p.get('role', 'Unknown')}:\n{json.dumps(p.get('proposal', {}), indent=2)}"
                for p in proposals if isinstance(p, dict)
            ])
        except Exception as e:
            st.warning(f"⚠️ {reviewer_role}: Failed to format proposals: {e}")
            return fallback_response

        prompt = f"""You are a {reviewer_role} agent. Review these proposals:

{proposals_text}

Provide constructive feedback from your perspective:
- Strengths of each proposal
- Weaknesses or concerns
- Suggestions for improvement

Return JSON:
{{
    "strengths": ["Strength 1", "Strength 2"],
    "concerns": ["Concern 1", "Concern 2"],
    "suggestions": ["Suggestion 1", "Suggestion 2"]
}}
"""

        if not GEMINI_API_KEY:
            st.warning(f"⚠️ {reviewer_role}: No Gemini API key available")
            return fallback_response

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            # Track cost
            if st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model="gemini-2.5-flash",
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            # Parse and validate response
            try:
                review = json.loads(response.text)

                # Validate required fields and ensure they're lists
                for field in ["strengths", "concerns", "suggestions"]:
                    if field not in review or not isinstance(review[field], list):
                        review[field] = []

                return review

            except json.JSONDecodeError as e:
                st.error(f"❌ {reviewer_role}: Failed to parse review JSON: {e}")
                return fallback_response

        except Exception as e:
            st.error(f"❌ {reviewer_role}: Review generation failed: {e}")
            return fallback_response

    async def _build_consensus(self, proposals: List[Dict], reviews: List[Dict]) -> Dict:
        """Build consensus from all proposals and reviews."""

        proposals_text = "\n\n".join([
            f"Proposal from {p['role']}:\n{json.dumps(p['proposal'], indent=2)}"
            for p in proposals
        ])

        reviews_text = "\n\n".join([
            f"Review from {r['reviewer']}:\n{json.dumps(r['review'], indent=2)}"
            for r in reviews
        ])

        prompt = f"""Build a consensus solution by synthesizing these proposals and reviews:

PROPOSALS:
{proposals_text}

REVIEWS:
{reviews_text}

Create a consensus that:
- Incorporates the best ideas from all proposals
- Addresses concerns raised in reviews
- Balances different perspectives
- Is actionable and concrete

Return JSON:
{{
    "consensus_approach": "The agreed-upon approach",
    "key_decisions": ["Decision 1", "Decision 2", "Decision 3"],
    "incorporated_ideas": {{"Agent 1": "Idea from Agent 1", "Agent 2": "Idea from Agent 2"}},
    "addressed_concerns": ["Concern 1 addressed", "Concern 2 addressed"]
}}
"""

        if not GEMINI_API_KEY:
            return {
                "consensus_approach": "Consensus approach",
                "key_decisions": [],
                "incorporated_ideas": {},
                "addressed_concerns": []
            }

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            # Track cost
            if st.session_state.cost_tracker:
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model="gemini-2.5-flash",
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            return json.loads(response.text)
        except Exception as e:
            return {
                "consensus_approach": f"Error: {str(e)}",
                "key_decisions": [],
                "incorporated_ideas": {},
                "addressed_concerns": []
            }

    async def _joint_evaluate(self, consensus: Dict, peer_roles: List[str]) -> Dict:
        """Jointly evaluate the consensus solution."""

        # Get individual evaluations from each peer
        individual_scores = []

        for role in peer_roles:
            prompt = f"""You are a {role} agent. Evaluate this consensus solution:

CONSENSUS:
{json.dumps(consensus, indent=2)}

Rate the solution from your perspective (0.0 to 1.0):
- Quality: How good is the solution?
- Completeness: Does it address all aspects?
- Feasibility: Can it be implemented?

Return JSON:
{{
    "quality": 0.8,
    "completeness": 0.9,
    "feasibility": 0.85,
    "overall": 0.85,
    "comments": "Your evaluation comments"
}}
"""

            if not GEMINI_API_KEY:
                individual_scores.append(0.5)
                continue

            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                )

                # Track cost
                if st.session_state.cost_tracker:
                    st.session_state.cost_tracker.update(
                        provider="Google",
                        model="gemini-2.5-flash",
                        api="generate_content",
                        raw_response_obj=response,
                        pricing_resolver=custom_gemini_price_lookup
                    )

                eval_result = json.loads(response.text)
                individual_scores.append(eval_result.get('overall', 0.5))
            except Exception:
                individual_scores.append(0.5)

        # Calculate agreement (how close are the scores?)
        if len(individual_scores) > 1:
            avg_score = sum(individual_scores) / len(individual_scores)
            variance = sum((s - avg_score) ** 2 for s in individual_scores) / len(individual_scores)
            agreement = max(0.0, 1.0 - variance)  # Higher agreement = lower variance
        else:
            avg_score = individual_scores[0] if individual_scores else 0.0
            agreement = 1.0

        return {
            "score": avg_score,
            "agreement": agreement,
            "individual_scores": individual_scores,
            "metric": "consensus_quality"
        }

    # ============================================
    # MODE 1: Inference (Prompt Optimization)
    # ============================================

    async def run_inference_prompt_optimization(self):
        """Inference mode: Optimize prompts for pattern matching tasks.

        Uses batched testing (map-reduce) because we're testing predictions
        on a dataset repeatedly.
        """

        st.subheader("Inference Mode: Prompt Optimization with Batched Testing")

        # Delegate to existing inference mode implementation (uses map-reduce)
        best_code, best_accuracy, history = await self.run_inference_mode()

        return {
            "strategy": "prompt_optimization",
            "best_prompt": best_code,
            "best_accuracy": best_accuracy,
            "history": history
        }

    # ============================================
    # MODE 2: Analysis (Code Generation - Single Execution)
    # ============================================

    async def run_analysis_code_generation(self):
        """Analysis mode: Generate code for computational tasks.

        Uses SINGLE execution per turn (no batching) because we're generating
        code once and executing it once to get results.
        """

        st.subheader("Analysis Mode: Code Generation (Single Execution)")

        test_name = "Test 5"
        best_code = None
        best_output = None
        best_score = 0.0

        for turn in range(1, self.budget.max_turns + 1):
            if self.budget.mode == "turns":
                self.budget.advance_turn()

            st.markdown(f"## Turn {turn} (Budget: {self.budget.left()*100:.1f}% remaining)")

            turn_id = f"analysis_turn_{turn}"

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=turn_id,
                    agent_name=f"Analysis Turn {turn}",
                    agent_type="main_agent"
                )

            # Build code generation prompt
            context_str = ""
            if self.test_data:
                context_str = f"""
Available dataset:
- Size: {len(self.test_data)} items
- Sample: {json.dumps(self.test_data[:3], indent=2)}
"""

            # Include previous attempt if exists
            previous_context = ""
            if best_code:
                previous_context = f"""
Previous attempt (score: {best_score:.3f}):
```python
{best_code}
```

Previous output:
{best_output[:500] if best_output else 'No output'}

Please improve upon this solution.
"""

            code_prompt = f"""Generate Python code to accomplish this computational task:

GOAL: {self.goal}

{context_str}

{previous_context}

Requirements:
- Complete, executable code
- Use pandas, numpy, scipy, matplotlib as needed
- Print clear results to stdout
- Handle errors gracefully

Return ONLY executable Python code.
"""

            st.write(f"**Turn {turn}**: Generating computational code...")

            try:
                # Single code execution - no batching
                if not GEMINI_API_KEY:
                    st.error("GEMINI_API_KEY required for code generation")
                    break

                client = genai.Client(api_key=GEMINI_API_KEY)

                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=code_prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                        )
                    )
                )

                # Track cost
                if st.session_state.cost_tracker:
                    st.session_state.cost_tracker.update(
                        provider="Google",
                        model="gemini-2.5-flash",
                        api="generate_content",
                        raw_response_obj=response,
                        pricing_resolver=custom_gemini_price_lookup
                    )

                # Extract code and output
                generated_code = None
                execution_output = None

                for part in response.candidates[0].content.parts:
                    if part.executable_code:
                        generated_code = part.executable_code.code
                        st.success(f"Generated {len(generated_code)} chars of code")

                    if part.code_execution_result:
                        execution_output = part.code_execution_result.output

                if not generated_code:
                    st.error("No code generated")
                    continue

                # Show code
                with st.expander("View Generated Code"):
                    st.code(generated_code, language='python')

                # Show output
                if execution_output:
                    st.subheader("Execution Results")
                    st.text(execution_output[:1000])  # Limit display

                    # Score based on successful execution
                    has_error = "error" in execution_output.lower() or "traceback" in execution_output.lower()
                    current_score = 0.0 if has_error else 1.0

                    if current_score > best_score:
                        best_code = generated_code
                        best_output = execution_output
                        best_score = current_score
                        self.best_turn = turn
                        st.success(f"✅ New best solution (score: {best_score:.3f})")
                    else:
                        st.info(f"Score: {current_score:.3f} (best: {best_score:.3f})")

                # Record turn
                self.record_turn(
                    tasks_attempted=1,
                    tasks_verified=1 if current_score > 0 else 0,
                    current_best=current_score
                )

                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id=turn_id,
                        score=current_score,
                        has_code=bool(generated_code)
                    )

                # Stop if we have a successful execution
                if best_score >= 1.0:
                    st.success("✅ Successful execution achieved!")
                    break

                # Check budget
                if self.budget.exhausted():
                    st.warning("Budget exhausted")
                    break

            except Exception as e:
                st.error(f"Code generation failed: {e}")
                if self.tracker:
                    self.tracker.emit(
                        test_name=test_name,
                        event_type="complete",
                        agent_id=turn_id,
                        error=str(e)
                    )
                continue

        return (best_code, best_score, self.turn_history)

    # ============================================
    # MODE 3: Research (Hybrid Strategy)
    # ============================================

    async def run_research_hybrid(self):
        """Research mode: Decompose into subtasks, each uses appropriate strategy."""

        st.subheader("Research Mode: Multi-Strategy Task Execution")

        test_name = "Test 5"
        all_findings = []

        for turn in range(1, self.budget.max_turns + 1):
            turn_id = f"research_turn_{turn}"

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="start",
                    agent_id=turn_id,
                    agent_name=f"Research Turn {turn}",
                    agent_type="research_turn"
                )

            st.markdown(f"## Research Turn {turn}")

            # Decompose into subtasks
            context = {"completed": len(all_findings)}
            tasks = await self.decompose_research(self.goal, context)

            st.write(f"Generated {len(tasks)} research subtasks:")
            for task in tasks:
                st.write(f"  - {task.goal}")

            # Execute tasks with intelligent strategy selection
            results = await asyncio.gather(*[
                self.execute_research_task_smart(task)
                for task in tasks
            ])

            # Store findings
            for result in results:
                if result.get('strategy') == 'code_generation':
                    # Computational analysis result
                    await self.index.store(result, VerificationResult(
                        task_id=result['task_id'],
                        claims=[f"Computed: {result.get('computation_type', 'analysis')}"],
                        evidence={"code": result.get('code', ''), "output": result.get('output', '')},
                        verdict="verified",
                        confidence=1.0,
                        outputs=result
                    ))
                    all_findings.append({
                        "type": "computation",
                        "task": result['task_id'],
                        "output": result.get('output', ''),
                        "code": result.get('code', '')
                    })
                else:
                    # Information gathering result
                    await self.index.store(result, VerificationResult(
                        task_id=result['task_id'],
                        claims=result['findings'],
                        evidence={"sources": result['sources']},
                        verdict="verified",
                        confidence=result['confidence'],
                        outputs=result
                    ))
                    all_findings.extend(result['findings'])

            if self.tracker:
                self.tracker.emit(
                    test_name=test_name,
                    event_type="complete",
                    agent_id=turn_id,
                    agent_name=f"Research Turn {turn}",
                    agent_type="research_turn",
                    findings_count=len(all_findings),
                    tasks_completed=len(results)
                )

            # Check if sufficient information gathered
            if len(all_findings) >= 15:
                st.info("Sufficient information gathered")
                break

        # Synthesize final report
        st.divider()
        st.subheader("Research Summary")

        synthesis = await self.synthesize_research_findings(all_findings)

        st.markdown(synthesis)

        return {
            "findings_count": len(all_findings),
            "knowledge_entries": len(self.index.entries),
            "synthesis": synthesis
        }

    async def execute_research_task_smart(self, task: Task) -> Dict:
        """
        Execute research task with intelligent strategy selection.

        Decision logic:
        - Keywords: "compute", "calculate", "analyze data", "statistics" → Code generation
        - Keywords: "search", "find", "research", "what is" → Information gathering
        """

        goal_lower = task.goal.lower()

        # Check if task requires computation
        computational_keywords = [
            'compute', 'calculate', 'analyze data', 'statistics',
            'mean', 'median', 'correlation', 'regression',
            'optimize', 'simulate', 'model'
        ]

        needs_computation = any(kw in goal_lower for kw in computational_keywords)

        if needs_computation:
            st.write(f"🔢 **Computational Task**: {task.goal}")
            return await self.execute_computational_subtask(task)
        else:
            st.write(f"🔍 **Research Task**: {task.goal}")
            return await self.execute_research_task(task)

    async def execute_computational_subtask(self, task: Task) -> Dict:
        """Execute computational subtask using code generation."""

        prompt = f"""Generate Python code to accomplish this computational task:

TASK: {task.goal}

Requirements:
- Use pandas, numpy, scipy, or matplotlib as needed
- Print results to stdout
- Handle errors gracefully

Return ONLY executable Python code.
"""

        if not GEMINI_API_KEY:
            return {
                "task_id": task.id,
                "strategy": "code_generation",
                "error": "GEMINI_API_KEY not set",
                "findings": [],
                "sources": [],
                "confidence": 0.0
            }

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)

            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )
            )

            # Extract code and output
            code = None
            output = None

            for part in response.candidates[0].content.parts:
                if part.executable_code:
                    code = part.executable_code.code
                if part.code_execution_result:
                    output = part.code_execution_result.output

            return {
                "task_id": task.id,
                "strategy": "code_generation",
                "computation_type": "analysis",
                "code": code,
                "output": output,
                "findings": [f"Computation completed: {task.goal}"],
                "sources": ["Gemini Code Execution"],
                "confidence": 1.0
            }

        except Exception as e:
            return {
                "task_id": task.id,
                "strategy": "code_generation",
                "error": str(e),
                "findings": [],
                "sources": [],
                "confidence": 0.0
            }

    async def synthesize_research_findings(self, findings: List[Dict]) -> str:
        """Synthesize research findings into coherent report."""

        prompt = f"""Synthesize these research findings into a coherent report:

ORIGINAL GOAL: {self.goal}

FINDINGS:
{json.dumps(findings, indent=2)}

Create a structured report with:
- Executive summary
- Key findings organized by topic
- Computational results (if any)
- Sources and confidence levels
- Recommendations or next steps
"""

        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY not set"

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)

            synthesis = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            )

            return synthesis.text

        except Exception as e:
            return f"Error synthesizing findings: {str(e)}"

# --- PATCH 4: Structured Summary Helper (Point 4) ---


