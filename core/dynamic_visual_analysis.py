"""
Dynamic Visual LLM Analysis Module

Provides flexible schema-free analysis that adapts to any visual task:
- No hardcoded Pydantic schemas
- Dynamic field detection via introspection
- Adaptive analysis planning based on detected fields
- Works for VR avatars, general images, medical imaging, etc.
"""

import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict


class DynamicVisualLLMAnalysis:
    """
    Flexible visual LLM analysis result that adapts to any task.
    
    Unlike the hardcoded VisualLLMAnalysis, this class:
    - Accepts ANY fields from the LLM response
    - Automatically categorizes fields as numerical, categorical, or descriptive
    - Provides introspection methods for analysis planning
    - No schema validation - pure flexibility
    """
    
    def __init__(self, model_name: str, raw_response: str, **fields):
        """
        Initialize with model name and arbitrary fields.
        
        Args:
            model_name: Name of the visual LLM model
            raw_response: Raw JSON response from the model
            **fields: Any fields returned by the model
        """
        self.model_name = model_name
        self.raw_response = raw_response
        self.timestamp = datetime.now().isoformat()
        
        # Store all fields dynamically
        self._fields = fields
        
        # Categorize fields automatically
        self._numerical_fields = set()
        self._categorical_fields = set()
        self._list_fields = set()
        self._descriptive_fields = set()
        
        self._categorize_fields()
    
    def _categorize_fields(self):
        """Automatically categorize fields based on their type and name."""
        for key, value in self._fields.items():
            # Skip metadata fields
            if key in ['model_name', 'raw_response', 'timestamp']:
                continue

            # A. Numerical Fields (ratings, counts, scores, confidence)
            if self._is_numerical_field(key, value):
                self._numerical_fields.add(key)

            # B. List/Array Fields (detected items, artifacts, themes)
            elif isinstance(value, list):
                self._list_fields.add(key)
                self._categorical_fields.add(key)

            # C. Long String Fields (descriptions, rationales, notes)
            elif isinstance(value, str) and len(value) >= 100:
                self._descriptive_fields.add(key)

            # D. Short String Fields (classifications, categories)
            elif isinstance(value, str):
                self._categorical_fields.add(key)
    
    def _is_numerical_field(self, key: str, value: Any) -> bool:
        """Determine if a field is numerical."""
        # Check type
        if isinstance(value, (int, float)):
            return True
        
        # Check name patterns
        numerical_patterns = [
            'rating', 'score', 'confidence', 'count', 'number',
            'quantity', 'percentage', 'probability', 'level'
        ]
        
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in numerical_patterns)
    
    def get_field(self, key: str, default=None):
        """Get a field value."""
        return self._fields.get(key, default)
    
    def get_all_fields(self) -> Dict[str, Any]:
        """Get all fields as a dictionary."""
        return self._fields.copy()
    
    def get_numerical_fields(self) -> Dict[str, float]:
        """Get all numerical fields."""
        return {k: self._fields[k] for k in self._numerical_fields if k in self._fields}
    
    def get_categorical_fields(self) -> Dict[str, Any]:
        """Get all categorical fields."""
        return {k: self._fields[k] for k in self._categorical_fields if k in self._fields}
    
    def get_list_fields(self) -> Dict[str, List]:
        """Get all list fields."""
        return {k: self._fields[k] for k in self._list_fields if k in self._fields}
    
    def get_descriptive_fields(self) -> Dict[str, str]:
        """Get all descriptive fields."""
        return {k: self._fields[k] for k in self._descriptive_fields if k in self._fields}
    
    def get_field_categories(self) -> Dict[str, Set[str]]:
        """Get categorization of all fields."""
        return {
            'numerical': self._numerical_fields.copy(),
            'categorical': self._categorical_fields.copy(),
            'list': self._list_fields.copy(),
            'descriptive': self._descriptive_fields.copy()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'raw_response': self.raw_response,
            **self._fields
        }
    
    def __getattr__(self, name: str):
        """Allow attribute-style access to fields."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._fields.get(name)
    
    def __repr__(self):
        field_summary = ', '.join(f"{k}={v}" for k, v in list(self._fields.items())[:3])
        if len(self._fields) > 3:
            field_summary += f", ... ({len(self._fields)} total fields)"
        return f"DynamicVisualLLMAnalysis(model={self.model_name}, {field_summary})"


def parse_dynamic_visual_response(
    raw_response: str,
    model_name: str,
    task_description: Optional[str] = None
) -> DynamicVisualLLMAnalysis:
    """
    Parse any visual LLM response into a dynamic analysis object.
    
    Args:
        raw_response: Raw JSON response from visual LLM
        model_name: Name of the model
        task_description: Optional task description for context
    
    Returns:
        DynamicVisualLLMAnalysis object with all fields
    """
    # Clean response (remove markdown code blocks)
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    # Try to parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        import re
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                # Fallback: create minimal structure
                data = {
                    'error': 'Failed to parse JSON',
                    'raw_text': cleaned[:500]
                }
        else:
            data = {
                'error': 'No JSON found',
                'raw_text': cleaned[:500]
            }
    
    # Create dynamic analysis object
    return DynamicVisualLLMAnalysis(
        model_name=model_name,
        raw_response=raw_response,
        **data
    )


def introspect_analysis_fields(
    analysis_results: List[Dict[str, Any]]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Introspect actual analysis results to detect field types.
    
    This is the CRITICAL function for dynamic adaptation.
    It inspects the actual outputs and categorizes fields WITHOUT
    relying on any predefined schema.
    
    Args:
        analysis_results: List of analysis results from visual LLMs
    
    Returns:
        Tuple of (numerical_fields, categorical_fields, descriptive_fields)
    """
    numerical_fields = set()
    categorical_fields = set()
    descriptive_fields = set()
    
    # Sample from first few results to detect fields
    sample_size = min(3, len(analysis_results))
    
    for result in analysis_results[:sample_size]:
        model_results = result.get('model_results', {})
        
        for model_name, analysis in model_results.items():
            # Handle both DynamicVisualLLMAnalysis and dict
            if isinstance(analysis, DynamicVisualLLMAnalysis):
                categories = analysis.get_field_categories()
                numerical_fields.update(categories['numerical'])
                categorical_fields.update(categories['categorical'])
                descriptive_fields.update(categories['descriptive'])
            
            elif isinstance(analysis, dict):
                # Introspect dictionary fields
                for key, value in analysis.items():
                    # Skip metadata
                    if key in ['model_name', 'raw_response', 'timestamp', 'image_id', 'image_path']:
                        continue

                    # Categorize based on type and name
                    if _is_numerical(key, value):
                        numerical_fields.add(key)
                    elif isinstance(value, list):
                        categorical_fields.add(key)
                    elif isinstance(value, str) and len(value) >= 100:
                        descriptive_fields.add(key)
                    elif isinstance(value, str):
                        categorical_fields.add(key)
    
    return numerical_fields, categorical_fields, descriptive_fields


def _is_numerical(key: str, value: Any) -> bool:
    """Helper to determine if a field is numerical."""
    if isinstance(value, (int, float)):
        return True
    
    numerical_patterns = [
        'rating', 'score', 'confidence', 'count', 'number',
        'quantity', 'percentage', 'probability', 'level', 'index'
    ]
    
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in numerical_patterns)


def create_adaptive_analysis_prompt(
    numerical_fields: Set[str],
    categorical_fields: Set[str],
    descriptive_fields: Set[str],
    task_description: str
) -> str:
    """
    Create an analysis prompt that adapts to detected fields.
    
    This ensures the computational analysis ONLY analyzes fields
    that actually exist in the data.
    
    Args:
        numerical_fields: Detected numerical fields
        categorical_fields: Detected categorical/list fields
        descriptive_fields: Detected descriptive fields
        task_description: Original task description
    
    Returns:
        Adaptive analysis prompt
    """
    prompt = f"""Analyze visual LLM outputs for the following task:

**Task:** {task_description}

**DETECTED FIELDS (analyze ONLY these fields):**

"""
    
    if numerical_fields:
        prompt += f"""**Numerical Fields (for distribution/correlation analysis):**
{', '.join(sorted(numerical_fields))}

**Required Analysis for Numerical Fields:**
- Calculate mean, median, standard deviation for each field
- Identify outliers (values > 2 std dev from mean)
- Calculate correlation between numerical fields (if multiple exist)
- Compare distributions across different models

"""
    
    if categorical_fields:
        prompt += f"""**Categorical/List Fields (for frequency analysis):**
{', '.join(sorted(categorical_fields))}

**Required Analysis for Categorical Fields:**
- Count frequency of unique values/items
- Identify most common categories/items
- Calculate diversity (number of unique values)
- Compare category distributions across models

"""
    
    if descriptive_fields:
        prompt += f"""**Descriptive Fields (for text analysis):**
{', '.join(sorted(descriptive_fields))}

**Required Analysis for Descriptive Fields:**
- Calculate average text length
- Identify common keywords/phrases
- Compare verbosity across models

"""
    
    prompt += """
**CRITICAL REQUIREMENTS:**
1. Analyze ONLY the fields listed above (do NOT assume other fields exist)
2. Handle missing values gracefully (some models may not return all fields)
3. Generate Python code using pandas/numpy
4. Store results in a 'results' dictionary
5. Include model-by-model comparison

**Output Format:**
Return JSON with:
- analysis_plan: Brief description
- python_code: Executable Python code
- expected_outputs: What the code will produce
- recommended_visualizations: List of chart types
"""
    
    return prompt

