"""
Automated cleanup script to remove extracted code from streamlit_test_v5.py

This script removes all code that has been extracted to separate modules:
- Pricing functions (core/pricing.py)
- Pydantic models (core/models.py)
- Model discovery (utils/model_discovery.py)
- Data helpers (utils/data_helpers.py)
- API clients (core/api_clients.py)
- Execution tracker (utils/execution_tracker.py)
"""

import re
from pathlib import Path


def remove_code_blocks(content: str) -> str:
    """Remove extracted code blocks from the main file."""
    
    lines = content.split('\n')
    result_lines = []
    skip_until_line = -1
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip if we're in a block that should be removed
        if i < skip_until_line:
            i += 1
            continue
        
        # Check for pricing functions section (lines ~501-870)
        if i > 500 and i < 900 and ('def _get_default_gemini_models' in line or
                                      'def _parse_gemini_models_from_linkup' in line or
                                      'def custom_gemini_price_lookup' in line or
                                      'def fetch_openai_models_from_linkup' in line or
                                      'def _get_default_openai_models' in line or
                                      'def _parse_openai_models_from_linkup' in line or
                                      'def get_all_available_models' in line or
                                      'def custom_openrouter_price_lookup' in line or
                                      'def _normalize_ollama_root' in line):
            # Find the end of this function
            indent_level = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    # Check if it's a new function or class
                    if next_line.strip().startswith('def ') or next_line.strip().startswith('class ') or next_line.strip().startswith('#'):
                        break
                j += 1
            skip_until_line = j
            result_lines.append(f"# {line.strip()} - moved to core/pricing.py or utils/model_discovery.py\n")
            i = j
            continue
        
        # Check for Pydantic models (lines ~880-992)
        if i > 870 and i < 1000 and ('class Classification(BaseModel)' in line or
                                       'class ClassificationWithConf' in line or
                                       'class SyntheticDataItem' in line or
                                       'class ToolCallSequenceItem' in line or
                                       'class PruningDataItem' in line or
                                       'class TestSummaryAndRefinement' in line or
                                       'class FactualConstraint' in line or
                                       'class ValidationResultArtifact' in line or
                                       'def convert_validation_to_artifact' in line):
            # Find the end of this class/function
            indent_level = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    if next_line.strip().startswith('def ') or next_line.strip().startswith('class ') or next_line.strip().startswith('#'):
                        break
                j += 1
            skip_until_line = j
            result_lines.append(f"# {line.strip()} - moved to core/models.py\n")
            i = j
            continue
        
        # Check for model discovery UI functions (lines ~993-1173)
        if i > 990 and i < 1200 and ('def fetch_openrouter_models_for_ui' in line or
                                       'def fetch_openai_models' in line or
                                       'def get_third_model_display_name' in line):
            indent_level = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    if next_line.strip().startswith('def ') or next_line.strip().startswith('class ') or next_line.strip().startswith('#'):
                        break
                j += 1
            skip_until_line = j
            result_lines.append(f"# {line.strip()} - moved to utils/model_discovery.py\n")
            i = j
            continue
        
        # Check for data helper functions (lines ~1174-1501)
        if i > 1170 and i < 1550 and ('def _subset_for_run' in line or
                                       'def _style_selected_rows' in line or
                                       'def save_results_df' in line or
                                       'def ensure_dataset_directory' in line or
                                       'def save_dataset_to_file' in line or
                                       'def load_classification_dataset' in line or
                                       'def load_tool_sequence_dataset' in line or
                                       'def _normalize_label' in line or
                                       'def load_context_pruning_dataset' in line or
                                       'def _load_df_from_path' in line or
                                       'async def auto_generate_default_datasets' in line or
                                       'async def check_and_generate_datasets' in line or
                                       'def _allowed_labels' in line or
                                       'async def _retry' in line):
            indent_level = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    if next_line.strip().startswith('def ') or next_line.strip().startswith('async def ') or next_line.strip().startswith('class ') or next_line.strip().startswith('#'):
                        break
                j += 1
            skip_until_line = j
            result_lines.append(f"# {line.strip()} - moved to utils/data_helpers.py or utils/helpers.py\n")
            i = j
            continue
        
        # Check for API client functions (lines ~6680-7483)
        if i > 6600 and i < 7600 and ('async def classify_with_openai' in line or
                                       'async def openai_structured_json' in line or
                                       'async def classify_with_gemini' in line or
                                       'async def classify_with_ollama' in line or
                                       'async def ollama_json' in line or
                                       'async def classify_with_openrouter' in line or
                                       'async def openrouter_json' in line or
                                       'async def generate_text_async' in line or
                                       'async def _classify_df_async' in line or
                                       'def _smarter_weighted_pick_row' in line or
                                       'async def run_judge_flexible' in line or
                                       'async def run_judge_ollama' in line or
                                       'async def run_judge_openai' in line or
                                       'async def run_pruner' in line or
                                       'async def generate_synthetic_data' in line):
            indent_level = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    if next_line.strip().startswith('def ') or next_line.strip().startswith('async def ') or next_line.strip().startswith('class ') or next_line.strip().startswith('#'):
                        break
                j += 1
            skip_until_line = j
            result_lines.append(f"# {line.strip()} - moved to core/api_clients.py\n")
            i = j
            continue
        
        # Keep this line
        result_lines.append(line + '\n')
        i += 1
    
    return ''.join(result_lines)


def main():
    """Main cleanup function."""
    main_file = Path('streamlit_test_v5.py')
    backup_file = Path('streamlit_test_v5.py.backup')
    
    # Create backup
    print(f"Creating backup: {backup_file}")
    content = main_file.read_text(encoding='utf-8')
    backup_file.write_text(content, encoding='utf-8')
    
    # Count original lines
    original_lines = len(content.split('\n'))
    print(f"Original file: {original_lines} lines")
    
    # Remove extracted code
    print("Removing extracted code blocks...")
    cleaned_content = remove_code_blocks(content)
    
    # Count new lines
    new_lines = len(cleaned_content.split('\n'))
    print(f"Cleaned file: {new_lines} lines")
    print(f"Removed: {original_lines - new_lines} lines")
    
    # Write cleaned file
    main_file.write_text(cleaned_content, encoding='utf-8')
    print(f"\n✅ Cleanup complete!")
    print(f"Backup saved to: {backup_file}")
    print(f"Main file updated: {main_file}")


if __name__ == '__main__':
    main()

