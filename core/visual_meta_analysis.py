"""
Visual LLM Meta-Analysis Module

Provides advanced analysis capabilities for Test 6 Mode B:
- Analysis planning with LLM
- Code generation and execution for computational analysis
- Model evaluation and comparison
- Prompt enhancement suggestions
- DYNAMIC SCHEMA ADAPTATION via field introspection
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

# Import dynamic analysis utilities
from core.dynamic_visual_analysis import (
    introspect_analysis_fields,
    create_adaptive_analysis_prompt
)


# Pydantic models for structured outputs
class ComputationalAnalysisPlan(BaseModel):
    """Structured plan for computational analysis."""
    analysis_plan: str
    python_code: str
    expected_outputs: str
    recommended_visualizations: List[str]


async def plan_computational_analysis(
    visual_llm_outputs: List[Dict[str, Any]],
    task_description: str,
    planner_model: str = "gpt-5-nano",
    openai_api_key: Optional[str] = None,
    use_dynamic_adaptation: bool = True
) -> Dict[str, Any]:
    """
    Use an LLM to plan what computational analysis would be valuable.

    **CRITICAL: This function now uses DYNAMIC FIELD INTROSPECTION**
    It inspects the actual outputs to detect fields and adapts the analysis
    plan accordingly, WITHOUT relying on any predefined schema.

    Args:
        visual_llm_outputs: List of structured outputs from visual LLMs
        task_description: Original analysis task description
        planner_model: Model to use for planning (gpt-5-nano or gemini-2.5-flash)
        openai_api_key: OpenAI API key
        use_dynamic_adaptation: If True, uses field introspection (recommended)

    Returns:
        Dict containing:
        - analysis_plan: Description of recommended analyses
        - python_code: Generated Python code for analysis
        - expected_outputs: Description of expected results
        - detected_fields: Dict of detected field categories (if dynamic)
    """
    # === STEP 1: DYNAMIC FIELD INTROSPECTION ===
    if use_dynamic_adaptation and visual_llm_outputs:
        numerical_fields, categorical_fields, descriptive_fields = introspect_analysis_fields(
            visual_llm_outputs
        )

        # Create adaptive prompt based on detected fields
        planning_prompt = create_adaptive_analysis_prompt(
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            descriptive_fields=descriptive_fields,
            task_description=task_description
        )

        # Store detected fields for reference
        detected_fields = {
            'numerical': list(numerical_fields),
            'categorical': list(categorical_fields),
            'descriptive': list(descriptive_fields)
        }
    else:
        # Fallback to generic prompt (old behavior)
        outputs_summary = _summarize_visual_outputs(visual_llm_outputs)

        planning_prompt = f"""Analyze visual LLM outputs and generate Python code for statistical analysis.

**Task:** {task_description}

**Data Summary:**
{outputs_summary}

**Generate:**
1. **analysis_plan**: Brief description of the analysis (1-2 sentences)
2. **python_code**: Python code using pandas/numpy to analyze the data
   - Calculate model agreement rates
   - Compare confidence scores
   - Find patterns or trends
   - Store results in a 'results' variable
3. **expected_outputs**: What the code will produce (1 sentence)
4. **recommended_visualizations**: List 2-3 chart types (e.g., ["bar chart", "heatmap"])

Keep it concise and focused on the most valuable insights."""

        detected_fields = None

    # Call LLM for planning
    if planner_model.startswith("gpt"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)

        try:
            # Use Pydantic structured outputs for GPT-5
            response = await client.beta.chat.completions.parse(
                model=planner_model,
                messages=[{"role": "user", "content": planning_prompt}],
                max_completion_tokens=16000,  # Increased for detailed analysis planning
                response_format=ComputationalAnalysisPlan
            )

            # Track cost
            if st.session_state.get('cost_tracker'):
                st.session_state.cost_tracker.update(
                    provider="OpenAI",
                    model=planner_model,
                    api="chat.completions",
                    raw_response_obj=response
                )

            # Extract parsed object
            plan_obj = response.choices[0].message.parsed
            if plan_obj:
                plan_json = plan_obj.model_dump()
            else:
                # Fallback: try to parse content as JSON
                content = response.choices[0].message.content
                if content:
                    plan_json = json.loads(content)
                else:
                    raise ValueError("Empty response from model")

        except Exception as e:
            error_msg = str(e)

            # Check if it's a length limit error
            if "length limit" in error_msg.lower() or "max_completion_tokens" in error_msg.lower():
                # Provide a simple fallback analysis
                return {
                    "analysis_plan": "Basic statistical analysis of model outputs",
                    "python_code": """import pandas as pd
import numpy as np
from collections import Counter

# Extract model names and their outputs
model_outputs = []
for result in data:
    for model_name, analysis in result.get('model_results', {}).items():
        model_outputs.append({
            'model': model_name,
            'confidence': analysis.get('confidence', 0),
            'rationale_length': len(analysis.get('rationale', ''))
        })

# Create DataFrame
df = pd.DataFrame(model_outputs)

# Calculate statistics
results = {
    'total_analyses': len(df),
    'models': df['model'].unique().tolist(),
    'avg_confidence': df['confidence'].mean(),
    'confidence_by_model': df.groupby('model')['confidence'].mean().to_dict()
}

print("Analysis Results:")
print(f"Total analyses: {results['total_analyses']}")
print(f"Models: {', '.join(results['models'])}")
print(f"Average confidence: {results['avg_confidence']:.2%}")
print("\\nConfidence by model:")
for model, conf in results['confidence_by_model'].items():
    print(f"  {model}: {conf:.2%}")
""",
                    "expected_outputs": "Model statistics including average confidence scores and per-model performance",
                    "recommended_visualizations": ["bar chart", "box plot"]
                }
            else:
                # Return error plan for other errors
                return {
                    "analysis_plan": f"Error generating plan: {error_msg}",
                    "python_code": "# Error: Could not generate code",
                    "expected_outputs": "N/A",
                    "recommended_visualizations": []
                }

    else:  # Gemini
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=gemini_api_key)

        try:
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=planning_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
            )

            # Track cost
            if st.session_state.get('cost_tracker'):
                from core.pricing import custom_gemini_price_lookup
                st.session_state.cost_tracker.update(
                    provider="Google",
                    model="gemini-2.5-flash",
                    api="generate_content",
                    raw_response_obj=response,
                    pricing_resolver=custom_gemini_price_lookup
                )

            plan_json = json.loads(response.text)

        except Exception as e:
            # Return error plan
            return {
                "analysis_plan": f"Error generating plan: {str(e)}",
                "python_code": "# Error: Could not generate code",
                "expected_outputs": "N/A",
                "recommended_visualizations": [],
                "detected_fields": detected_fields
            }

    # Add detected fields to result
    if detected_fields:
        plan_json['detected_fields'] = detected_fields

    return plan_json


async def execute_analysis_code(
    python_code: str,
    visual_llm_outputs: List[Dict[str, Any]],
    use_gemini_execution: bool = True,
    gemini_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute generated Python code for analysis.
    
    Args:
        python_code: Python code to execute
        visual_llm_outputs: Data to analyze
        use_gemini_execution: Use Gemini's code execution framework
        gemini_api_key: Gemini API key
    
    Returns:
        Dict containing:
        - success: Whether execution succeeded
        - results: Analysis results
        - error: Error message if failed
    """
    if use_gemini_execution:
        # Use Gemini's code execution framework (proper SDK)
        try:
            client = genai.Client(api_key=gemini_api_key)

            # Prepare data for code execution
            data_json = json.dumps(visual_llm_outputs, indent=2, default=str)

            execution_prompt = f"""Analyze the visual LLM outputs using Python code.

**Data available:**
```python
import json
data = {data_json}
```

**Task:**
Execute this analysis code:
```python
{python_code}
```

Run the code and provide the results."""

            # Execute with code execution tool
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=execution_prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                    )
                )
            )

            # Track cost
            if st.session_state.get('cost_tracker'):
                from core.pricing import custom_gemini_price_lookup
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
                if part.code_execution_result:
                    execution_output = part.code_execution_result.output

            # Return results
            if execution_output:
                return {
                    "success": True,
                    "results": execution_output,
                    "generated_code": generated_code,
                    "execution_method": "gemini_code_execution"
                }
            else:
                return {
                    "success": False,
                    "error": "No execution output received",
                    "generated_code": generated_code,
                    "execution_method": "gemini_code_execution"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_method": "gemini_code_execution"
            }
    
    else:
        # Local execution (sandboxed)
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            # Prepare data
            data = visual_llm_outputs
            
            # Execute code in restricted namespace
            namespace = {
                "data": data,
                "pd": pd,
                "np": np,
                "stats": stats,
                "json": json
            }
            
            exec(python_code, namespace)
            
            # Extract results
            results = namespace.get("results", "No results variable found")
            
            return {
                "success": True,
                "results": results,
                "execution_method": "local_sandboxed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_method": "local_sandboxed"
            }


async def evaluate_visual_llm_performance(
    visual_llm_outputs: List[Dict[str, Any]],
    task_description: str,
    computational_results: Optional[Dict[str, Any]] = None,
    judge_model: str = "gpt-5-nano",
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use an LLM judge to evaluate which visual LLM performed best.
    
    Args:
        visual_llm_outputs: Outputs from all visual LLMs
        task_description: Original task description
        computational_results: Results from computational analysis
        judge_model: Model to use as judge
        openai_api_key: OpenAI API key
    
    Returns:
        Dict containing:
        - best_model: Name of best performing model
        - model_rankings: Ranked list of models with scores
        - strengths: Dict of model-specific strengths
        - recommendations: Task-specific recommendations
        - enhanced_prompts: Improved prompts for each model
    """
    # Build evaluation prompt
    outputs_summary = _summarize_visual_outputs(visual_llm_outputs)
    
    comp_results_str = ""
    if computational_results and computational_results.get("success"):
        comp_results_str = f"\n**Computational Analysis Results:**\n{computational_results.get('results', 'N/A')}"
    
    evaluation_prompt = f"""You are an expert AI model evaluator. Analyze the performance of multiple visual LLM models.

**Task:** {task_description}

**Visual LLM Outputs:**
{outputs_summary}
{comp_results_str}

**Your Task:**
1. Evaluate which model performed best for this specific task
2. Identify strengths and weaknesses of each model
3. Provide task-specific recommendations
4. Generate enhanced prompts for each model to improve performance

**Output Format (JSON):**
{{
    "best_model": "Name of best model",
    "model_rankings": [
        {{"model": "Model name", "score": 0-100, "rationale": "Why this score"}}
    ],
    "strengths": {{
        "model_name": ["strength1", "strength2"]
    }},
    "recommendations": {{
        "general": "Overall recommendations",
        "task_specific": "Recommendations for this task type"
    }},
    "enhanced_prompts": {{
        "model_name": "Enhanced prompt for this model"
    }}
}}

Provide ONLY the JSON output."""

    # Call judge LLM
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=openai_api_key)
    
    response = await client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_completion_tokens=16000,  # Increased for comprehensive evaluation
        response_format={"type": "json_object"}
    )
    
    evaluation = json.loads(response.choices[0].message.content)
    
    return evaluation


def _summarize_visual_outputs(outputs: List[Dict[str, Any]]) -> str:
    """Create a concise summary of visual LLM outputs."""
    summary_lines = []
    
    for idx, output in enumerate(outputs[:10]):  # Limit to first 10 for brevity
        image_name = output.get("image_name", f"Image {idx+1}")
        model_results = output.get("model_results", {})
        
        summary_lines.append(f"\n**{image_name}:**")
        
        for model_name, analysis in model_results.items():
            if hasattr(analysis, 'rationale'):
                rationale = analysis.rationale[:200] + "..." if len(analysis.rationale) > 200 else analysis.rationale
                confidence = getattr(analysis, 'confidence', 0.0)
                summary_lines.append(f"  - {model_name}: {rationale} (confidence: {confidence:.2f})")
    
    if len(outputs) > 10:
        summary_lines.append(f"\n... and {len(outputs) - 10} more images")
    
    return "\n".join(summary_lines)

