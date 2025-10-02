"""
Leaf Agent Scaffold: Hierarchical Multi-Agent System

This module implements a true multi-agent scaffold architecture with:
- Supervisor Agent (orchestrator)
- Specialized Leaf Agents (workers)
- Task Planning & Decomposition
- Result Synthesis

Architecture:
    Supervisor
        ├── WebResearchAgent
        ├── CodeExecutorAgent
        ├── KnowledgeRetrieverAgent
        ├── ContentGeneratorAgent
        └── ValidatorAgent
"""

import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================
# Data Models
# ============================================

class AgentType(Enum):
    """Types of specialized leaf agents."""
    WEB_RESEARCHER = "web_researcher"
    CODE_EXECUTOR = "code_executor"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    CONTENT_GENERATOR = "content_generator"
    VALIDATOR = "validator"
    EDITOR = "editor"


@dataclass
class SubTask:
    """A specialized sub-task for a leaf agent."""
    id: str
    description: str
    agent_type: AgentType
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""


@dataclass
class AgentResult:
    """Result from a leaf agent execution."""
    agent_name: str
    agent_type: AgentType
    sub_task_id: str
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class FinalResult:
    """Final synthesized result from supervisor."""
    answer: str
    contributing_agents: List[str]
    sub_task_results: List[AgentResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================
# Base Classes
# ============================================

class LeafAgent:
    """Base class for specialized leaf agents."""
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        specialization: str,
        tools: List[str],
        context_limit: int = 4000
    ):
        self.name = name
        self.agent_type = agent_type
        self.specialization = specialization
        self.tools = tools
        self.context_limit = context_limit
    
    async def execute(self, sub_task: SubTask, shared_context: Dict[str, Any]) -> AgentResult:
        """
        Execute assigned sub-task.
        
        Args:
            sub_task: The sub-task to execute
            shared_context: Shared context from previous sub-tasks
        
        Returns:
            AgentResult with output and metadata
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def can_handle(self, sub_task: SubTask) -> bool:
        """Check if this agent can handle the sub-task."""
        return sub_task.agent_type == self.agent_type
    
    async def call_llm(self, prompt: str, **kwargs) -> str:
        """Call LLM with agent-specific prompt."""
        # To be implemented by subclasses with specific LLM integration
        raise NotImplementedError("Subclasses must implement call_llm()")


class TaskPlanner:
    """Decomposes complex tasks into specialized sub-tasks."""
    
    def __init__(self, available_agents: List[AgentType]):
        self.available_agents = available_agents
    
    async def decompose(
        self,
        complex_task: str,
        mode: str = "inference"
    ) -> List[SubTask]:
        """
        Break down complex task into specialized sub-tasks.
        
        Args:
            complex_task: The high-level task description
            mode: Execution mode (inference, analysis, research)
        
        Returns:
            List of SubTask objects with dependencies
        """
        # Build agent descriptions
        agent_descriptions = self._get_agent_descriptions()
        
        prompt = f"""You are a task planning specialist for a multi-agent system.

COMPLEX TASK: {complex_task}

MODE: {mode}

AVAILABLE SPECIALIZED AGENTS:
{agent_descriptions}

Your job is to decompose this task into specialized sub-tasks that can be executed by the available agents.

For each sub-task, specify:
1. description: Clear description of what needs to be done
2. agent_type: Which specialized agent should handle it (use exact agent type names)
3. dependencies: List of sub-task IDs that must complete first (use ["1", "2"] format)
4. expected_output: What output is expected from this sub-task

IMPORTANT:
- Create a logical workflow with proper dependencies
- Use specialized agents for their strengths
- Ensure sub-tasks are focused and achievable
- Order sub-tasks by dependencies (independent tasks first)

Return JSON array:
[
  {{
    "id": "1",
    "description": "Sub-task description",
    "agent_type": "web_researcher",
    "dependencies": [],
    "expected_output": "Expected output description"
  }},
  ...
]
"""
        
        # Call LLM to decompose task
        response = await self._call_llm_for_planning(prompt)
        
        # Parse and validate sub-tasks
        sub_tasks = self._parse_sub_tasks(response)
        
        return sub_tasks
    
    def _get_agent_descriptions(self) -> str:
        """Get descriptions of available agents."""
        descriptions = {
            AgentType.WEB_RESEARCHER: "- web_researcher: Gathers information from web sources, searches for recent data and trends",
            AgentType.CODE_EXECUTOR: "- code_executor: Executes Python code for computations, data analysis, and visualizations",
            AgentType.KNOWLEDGE_RETRIEVER: "- knowledge_retriever: Retrieves domain knowledge from knowledge bases and documentation",
            AgentType.CONTENT_GENERATOR: "- content_generator: Writes and formats content, creates structured documents",
            AgentType.VALIDATOR: "- validator: Validates results, checks accuracy and quality",
            AgentType.EDITOR: "- editor: Refines and polishes content for clarity and professionalism"
        }
        
        available_descriptions = [
            descriptions[agent_type]
            for agent_type in self.available_agents
            if agent_type in descriptions
        ]
        
        return "\n".join(available_descriptions)
    
    async def _call_llm_for_planning(self, prompt: str) -> str:
        """Call LLM for task planning."""
        # To be implemented with actual LLM integration
        raise NotImplementedError("Must be implemented with LLM integration")
    
    def _parse_sub_tasks(self, response: str) -> List[SubTask]:
        """Parse LLM response into SubTask objects."""
        try:
            data = json.loads(response)
            
            sub_tasks = []
            for item in data:
                # Convert agent_type string to enum
                agent_type_str = item.get("agent_type", "").lower()
                agent_type = AgentType(agent_type_str)
                
                sub_task = SubTask(
                    id=item["id"],
                    description=item["description"],
                    agent_type=agent_type,
                    dependencies=item.get("dependencies", []),
                    expected_output=item.get("expected_output", "")
                )
                sub_tasks.append(sub_task)
            
            return sub_tasks
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: create simple sequential sub-tasks
            return self._create_fallback_sub_tasks()
    
    def _create_fallback_sub_tasks(self) -> List[SubTask]:
        """Create fallback sub-tasks if parsing fails."""
        return [
            SubTask(
                id="1",
                description="Execute the task",
                agent_type=self.available_agents[0] if self.available_agents else AgentType.CONTENT_GENERATOR,
                dependencies=[],
                expected_output="Task result"
            )
        ]


class ResultSynthesizer:
    """Synthesizes results from multiple leaf agents into final answer."""
    
    async def synthesize(
        self,
        original_task: str,
        results: List[AgentResult]
    ) -> FinalResult:
        """
        Combine partial results into final answer.
        
        Args:
            original_task: The original complex task
            results: List of results from leaf agents
        
        Returns:
            FinalResult with synthesized answer
        """
        # Format results for synthesis
        results_text = self._format_results(results)
        
        prompt = f"""You are a result synthesis specialist.

ORIGINAL TASK: {original_task}

SUB-TASK RESULTS:
{results_text}

Your job is to synthesize these partial results into a coherent final answer.

Guidelines:
- Integrate findings from all agents
- Resolve any conflicts or inconsistencies
- Ensure completeness and accuracy
- Format professionally
- Cite sources where applicable

Return the final synthesized answer."""
        
        # Call LLM for synthesis
        final_answer = await self._call_llm_for_synthesis(prompt)
        
        return FinalResult(
            answer=final_answer,
            contributing_agents=[r.agent_name for r in results],
            sub_task_results=results,
            metadata={
                "total_agents": len(set(r.agent_name for r in results)),
                "successful_tasks": sum(1 for r in results if r.success),
                "failed_tasks": sum(1 for r in results if not r.success)
            }
        )
    
    def _format_results(self, results: List[AgentResult]) -> str:
        """Format agent results for synthesis."""
        formatted = []
        
        for idx, result in enumerate(results, 1):
            status = "✅ Success" if result.success else "❌ Failed"
            
            formatted.append(f"""
Result {idx} - {result.agent_name} ({result.agent_type.value})
Status: {status}
Sub-task ID: {result.sub_task_id}
Output: {result.output}
Metadata: {json.dumps(result.metadata, indent=2)}
""")
        
        return "\n".join(formatted)
    
    async def _call_llm_for_synthesis(self, prompt: str) -> str:
        """Call LLM for result synthesis."""
        # To be implemented with actual LLM integration
        raise NotImplementedError("Must be implemented with LLM integration")


class PolicyUpdater:
    """
    Helper class to process validator output and update memory with policy constraints.
    """

    @staticmethod
    def process_validation_result(validation_result: Dict[str, Any], memory_manager, turn: int) -> bool:
        """
        Process validation result and update memory if needed.

        Args:
            validation_result: Structured validation output from ValidatorAgent
            memory_manager: MemoryManager instance to update
            turn: Current turn number

        Returns:
            True if policy was updated, False otherwise
        """
        if not memory_manager:
            return False

        confidence_score = validation_result.get("confidence_score", 1.0)
        policy_suggestions = validation_result.get("policy_suggestions", [])

        # Update policy if confidence is low
        if confidence_score < 0.7 and policy_suggestions:
            try:
                # Get current Tool Guidelines
                current_guidelines = memory_manager.core_blocks.get("Tool Guidelines")
                if not current_guidelines:
                    return False

                existing_content = current_guidelines.content

                # Add new constraints
                new_constraints = "\n\n--- FACTUAL CONSTRAINTS (Auto-generated from validation) ---\n"
                for suggestion in policy_suggestions:
                    new_constraints += f"\n{suggestion}"

                updated_content = existing_content + new_constraints

                # Update memory block
                memory_manager.update_block(
                    block_name="Tool Guidelines",
                    new_content=updated_content,
                    turn=turn,
                    trigger="VALIDATION_FAILURE",
                    change_summary=f"Added {len(policy_suggestions)} factual constraint(s) from validation"
                )

                return True

            except Exception as e:
                print(f"Error updating policy: {e}")
                return False

        return False


class SupervisorAgent:
    """
    High-level orchestrator that manages leaf agents.

    Responsibilities:
    1. Task planning and decomposition
    2. Delegation to leaf agents
    3. Dependency management
    4. Result consolidation and synthesis
    5. Policy-based memory updates from validation
    """

    def __init__(self, leaf_agents: List[LeafAgent], memory_manager=None):
        self.leaf_agents = leaf_agents
        self.task_planner = TaskPlanner(
            available_agents=[agent.agent_type for agent in leaf_agents]
        )
        self.result_synthesizer = ResultSynthesizer()
        self.shared_context: Dict[str, Any] = {}
        self.memory_manager = memory_manager
        self.policy_updater = PolicyUpdater()
    
    async def execute(self, complex_task: str, mode: str = "inference") -> FinalResult:
        """
        Main execution flow for hierarchical multi-agent system.
        
        Args:
            complex_task: The high-level task to accomplish
            mode: Execution mode (inference, analysis, research)
        
        Returns:
            FinalResult with synthesized answer
        """
        # Step 1: Decompose task into sub-tasks
        sub_tasks = await self.task_planner.decompose(complex_task, mode)
        
        # Step 2: Execute sub-tasks with dependency management
        results = await self.execute_sub_tasks(sub_tasks)
        
        # Step 3: Synthesize final result
        final_result = await self.result_synthesizer.synthesize(complex_task, results)
        
        return final_result
    
    async def execute_sub_tasks(self, sub_tasks: List[SubTask]) -> List[AgentResult]:
        """
        Execute sub-tasks respecting dependencies.
        
        Uses topological sort to execute tasks in correct order.
        """
        results = []
        completed_tasks = set()
        
        # Build dependency graph
        remaining_tasks = sub_tasks.copy()
        
        while remaining_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = [
                task for task in remaining_tasks
                if all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                break
            
            # Execute ready tasks in parallel
            task_results = await asyncio.gather(*[
                self.execute_single_task(task)
                for task in ready_tasks
            ])
            
            results.extend(task_results)
            
            # Mark tasks as completed
            for task in ready_tasks:
                completed_tasks.add(task.id)
                remaining_tasks.remove(task)
        
        return results
    
    async def execute_single_task(self, sub_task: SubTask) -> AgentResult:
        """Execute a single sub-task by delegating to appropriate leaf agent."""
        # Find agent that can handle this sub-task
        agent = self.find_agent_for_task(sub_task)

        if not agent:
            return AgentResult(
                agent_name="Unknown",
                agent_type=sub_task.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=f"No agent available for type: {sub_task.agent_type}"
            )

        # Inject memory constraints into task if memory manager is available
        if self.memory_manager and hasattr(self.memory_manager, 'core_blocks'):
            tool_guidelines = self.memory_manager.core_blocks.get("Tool Guidelines")
            if tool_guidelines:
                # Add constraints to sub-task description
                constraints = tool_guidelines.content
                sub_task.description = f"""{sub_task.description}

CRITICAL CONSTRAINTS FROM MEMORY:
{constraints}

You MUST respect these constraints in your output."""

        # Execute sub-task
        try:
            result = await agent.execute(sub_task, self.shared_context)

            # Update shared context with result
            self.shared_context[sub_task.id] = result.output

            # Process validation results if this was a ValidatorAgent
            if agent.agent_type == AgentType.VALIDATOR and result.success:
                validation_result = result.metadata.get("validation_result")
                if validation_result and self.memory_manager:
                    # Determine current turn (use number of completed tasks as proxy)
                    current_turn = len(self.shared_context)

                    # Update policy if validation failed
                    policy_updated = self.policy_updater.process_validation_result(
                        validation_result,
                        self.memory_manager,
                        current_turn
                    )

                    if policy_updated:
                        # Add metadata to result
                        result.metadata["policy_updated"] = True
                        result.metadata["constraints_added"] = len(validation_result.get("policy_suggestions", []))

            return result

        except Exception as e:
            return AgentResult(
                agent_name=agent.name,
                agent_type=agent.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=str(e)
            )
    
    def find_agent_for_task(self, sub_task: SubTask) -> Optional[LeafAgent]:
        """Find the best leaf agent for a sub-task."""
        for agent in self.leaf_agents:
            if agent.can_handle(sub_task):
                return agent
        return None


# ============================================
# Specialized Leaf Agent Implementations
# ============================================

class WebResearchAgent(LeafAgent):
    """Specialized agent for web research and information gathering."""

    def __init__(self, llm_client):
        super().__init__(
            name="Web Researcher",
            agent_type=AgentType.WEB_RESEARCHER,
            specialization="Information gathering from web sources",
            tools=["web_search", "url_fetch", "content_extraction"],
            context_limit=4000
        )
        self.llm_client = llm_client

    async def execute(self, sub_task: SubTask, shared_context: Dict[str, Any]) -> AgentResult:
        """Perform web research for the sub-task."""
        try:
            prompt = f"""You are a web research specialist.

SUB-TASK: {sub_task.description}

EXPECTED OUTPUT: {sub_task.expected_output}

CONTEXT FROM PREVIOUS TASKS:
{json.dumps(shared_context, indent=2)}

Your job is to gather relevant information for this sub-task.

Guidelines:
- Focus on recent and authoritative sources
- Provide specific data, statistics, and examples
- Cite sources where possible
- Be concise but comprehensive

Return your research findings in a structured format."""

            output = await self.call_llm(prompt)

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=True,
                output=output,
                metadata={"tool_used": "web_search"}
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=str(e)
            )

    async def call_llm(self, prompt: str) -> str:
        """Call LLM with web research capabilities."""
        # Implementation will use Gemini with grounding/search
        return await self.llm_client.generate(prompt)


class CodeExecutorAgent(LeafAgent):
    """Specialized agent for code execution and computational tasks."""

    def __init__(self, llm_client):
        super().__init__(
            name="Code Executor",
            agent_type=AgentType.CODE_EXECUTOR,
            specialization="Python code execution and data analysis",
            tools=["python_exec", "data_analysis", "visualization"],
            context_limit=4000
        )
        self.llm_client = llm_client

    async def execute(self, sub_task: SubTask, shared_context: Dict[str, Any]) -> AgentResult:
        """Execute computational sub-task."""
        try:
            prompt = f"""You are a code execution specialist.

SUB-TASK: {sub_task.description}

EXPECTED OUTPUT: {sub_task.expected_output}

CONTEXT FROM PREVIOUS TASKS:
{json.dumps(shared_context, indent=2)}

Your job is to write and execute Python code to accomplish this sub-task.

Guidelines:
- Write clean, efficient Python code
- Include comments explaining your approach
- Handle edge cases and errors
- Return results in a structured format

Write Python code to solve this task."""

            output = await self.call_llm_with_code_exec(prompt)

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=True,
                output=output,
                metadata={"tool_used": "code_execution"}
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=str(e)
            )

    async def call_llm_with_code_exec(self, prompt: str) -> str:
        """Call LLM with code execution capabilities."""
        # Implementation will use Gemini Code Execution
        return await self.llm_client.generate_with_code_exec(prompt)

    async def call_llm(self, prompt: str) -> str:
        """Fallback to regular LLM call."""
        return await self.call_llm_with_code_exec(prompt)


class ContentGeneratorAgent(LeafAgent):
    """Specialized agent for content generation and writing."""

    def __init__(self, llm_client):
        super().__init__(
            name="Content Generator",
            agent_type=AgentType.CONTENT_GENERATOR,
            specialization="Content writing and formatting",
            tools=["text_generation", "formatting", "structuring"],
            context_limit=8000
        )
        self.llm_client = llm_client

    async def execute(self, sub_task: SubTask, shared_context: Dict[str, Any]) -> AgentResult:
        """Generate content for the sub-task."""
        try:
            prompt = f"""You are a content generation specialist.

SUB-TASK: {sub_task.description}

EXPECTED OUTPUT: {sub_task.expected_output}

CONTEXT FROM PREVIOUS TASKS:
{json.dumps(shared_context, indent=2)}

Your job is to create well-written, structured content for this sub-task.

Guidelines:
- Use information from previous tasks
- Write clearly and professionally
- Structure content logically
- Format appropriately (markdown, sections, etc.)

Generate the requested content."""

            output = await self.call_llm(prompt)

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=True,
                output=output,
                metadata={"tool_used": "text_generation"}
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=str(e)
            )

    async def call_llm(self, prompt: str) -> str:
        """Call LLM for content generation."""
        return await self.llm_client.generate(prompt)


class ValidatorAgent(LeafAgent):
    """Specialized agent for validation and quality assurance with hallucination detection."""

    def __init__(self, llm_client):
        super().__init__(
            name="Validator",
            agent_type=AgentType.VALIDATOR,
            specialization="Quality assurance, validation, and hallucination detection",
            tools=["fact_checking", "validation", "quality_assessment", "hallucination_detection"],
            context_limit=4000
        )
        self.llm_client = llm_client

    async def execute(self, sub_task: SubTask, shared_context: Dict[str, Any]) -> AgentResult:
        """Validate results from previous tasks with hallucination detection."""
        try:
            # Enhanced prompt for structured validation with hallucination detection
            prompt = f"""You are an expert validation specialist with expertise in detecting AI hallucinations and fabricated information.

SUB-TASK: {sub_task.description}

EXPECTED OUTPUT: {sub_task.expected_output}

CONTEXT FROM PREVIOUS TASKS:
{json.dumps(shared_context, indent=2)}

CRITICAL VALIDATION REQUIREMENTS:

1. **Hallucination Detection:** For each biographical claim (degree, previous employer, job title, credentials):
   - Count distinct, high-quality sources supporting the claim
   - Assign Hallucination Risk Score (0.0-1.0):
     * 0.0-0.3: Verified by 2+ credible sources with citations
     * 0.4-0.7: Single source or weak evidence
     * 0.8-1.0: NO sources OR appears to be "industry boilerplate" (typical CEO profile without citations)

2. **Citation Integrity:** Check if claims have proper source attribution

3. **Factual Consistency:** Verify claims don't contradict each other

4. **Red Flag Detection:** Flag claims with hallucination risk > 0.7

5. **Policy Suggestions:** Generate explicit constraints to prevent future hallucinations

REQUIRED OUTPUT FORMAT (valid JSON):
{{
  "verdict": "VERIFIED" or "FAILED",
  "confidence_score": 0.0-1.0,
  "red_flags": [
    {{
      "claim": "specific claim text",
      "hallucination_risk": 0.0-1.0,
      "reason": "explanation of why this is flagged",
      "sources_found": number
    }}
  ],
  "policy_suggestions": [
    "CONSTRAINT: specific factual constraint to add to memory",
    "CONSTRAINT: another constraint"
  ],
  "verified_facts": [
    "fact that passed validation"
  ],
  "summary": "brief summary of validation results"
}}

IMPORTANT: Return ONLY valid JSON. Do not include markdown code blocks or explanatory text."""

            output = await self.call_llm(prompt)

            # Parse and validate JSON output
            try:
                # Remove markdown code blocks if present
                cleaned_output = output.strip()
                if cleaned_output.startswith("```"):
                    # Extract JSON from code block
                    lines = cleaned_output.split('\n')
                    json_lines = []
                    in_code_block = False
                    for line in lines:
                        if line.strip().startswith("```"):
                            in_code_block = not in_code_block
                            continue
                        if in_code_block or (not line.strip().startswith("```")):
                            json_lines.append(line)
                    cleaned_output = '\n'.join(json_lines).strip()

                validation_result = json.loads(cleaned_output)

                # Ensure required fields exist
                if "verdict" not in validation_result:
                    validation_result["verdict"] = "FAILED"
                if "confidence_score" not in validation_result:
                    validation_result["confidence_score"] = 0.5
                if "red_flags" not in validation_result:
                    validation_result["red_flags"] = []
                if "policy_suggestions" not in validation_result:
                    validation_result["policy_suggestions"] = []

            except json.JSONDecodeError as e:
                # Fallback: create structured output from text
                validation_result = {
                    "verdict": "FAILED",
                    "confidence_score": 0.3,
                    "red_flags": [{
                        "claim": "JSON parsing failed",
                        "hallucination_risk": 1.0,
                        "reason": f"Validator output was not valid JSON: {str(e)}",
                        "sources_found": 0
                    }],
                    "policy_suggestions": ["CONSTRAINT: Validator must return valid JSON"],
                    "verified_facts": [],
                    "summary": f"Validation failed due to JSON parsing error. Raw output: {output[:200]}"
                }

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=True,
                output=json.dumps(validation_result, indent=2),
                metadata={
                    "tool_used": "hallucination_detection",
                    "validation_result": validation_result,
                    "verdict": validation_result.get("verdict", "UNKNOWN"),
                    "confidence_score": validation_result.get("confidence_score", 0.0)
                }
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                sub_task_id=sub_task.id,
                success=False,
                output=None,
                error=str(e)
            )

    async def call_llm(self, prompt: str) -> str:
        """Call LLM for validation."""
        return await self.llm_client.generate(prompt)

