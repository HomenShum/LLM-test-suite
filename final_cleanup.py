"""
Final cleanup script to remove all leftover code from streamlit_test_v5.py

This script will:
1. Remove duplicate pricing/model discovery functions (lines ~520-900)
2. Remove duplicate Pydantic model definitions
3. Add missing variable definitions
4. Create backup before making changes
5. Validate syntax after cleanup
"""

import re
import shutil
from datetime import datetime

def create_backup():
    """Create timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"streamlit_test_v5.py.backup_final_{timestamp}"
    shutil.copy("streamlit_test_v5.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    return backup_name

def find_section_boundaries(lines):
    """Find the boundaries of leftover code sections."""
    sections_to_remove = []
    
    # Find leftover pricing/model code
    in_leftover_section = False
    section_start = None
    
    for i, line in enumerate(lines):
        # Start of leftover section (after "All Pydantic model definitions removed")
        if "All Pydantic model definitions removed" in line:
            section_start = i + 1
            in_leftover_section = True
            continue
        
        # End of leftover section (when we hit async def run_live_smoke_test or similar)
        if in_leftover_section and (
            line.strip().startswith("async def run_live_smoke_test") or
            line.strip().startswith("# ============================================================") and "DEMONSTRATION" in lines[i+1] if i+1 < len(lines) else False
        ):
            if section_start is not None:
                sections_to_remove.append((section_start, i))
                print(f"   Found leftover section: lines {section_start+1} to {i}")
            in_leftover_section = False
            section_start = None
    
    return sections_to_remove

def remove_sections(lines, sections):
    """Remove specified line ranges from the file."""
    # Sort sections in reverse order to maintain line numbers
    sections_sorted = sorted(sections, reverse=True)
    
    for start, end in sections_sorted:
        del lines[start:end]
        print(f"   Removed lines {start+1} to {end}")
    
    return lines

def add_missing_definitions(lines):
    """Add missing variable definitions after imports."""
    # Find where to insert (after the comment about Pydantic models)
    insert_idx = None
    for i, line in enumerate(lines):
        if "All Pydantic model definitions removed" in line:
            insert_idx = i + 1
            break
    
    if insert_idx is None:
        print("⚠️  Could not find insertion point for variable definitions")
        return lines
    
    # Variable definitions to add
    definitions = """
# Create model metadata from imported functions
from core.pricing import fetch_gemini_models_from_linkup, fetch_openai_models_from_linkup

GEMINI_MODELS_FULL = fetch_gemini_models_from_linkup()
GEMINI_MODEL_METADATA = {
    model_id: {
        'context': meta.get('context', 'N/A'),
        'input_cost': meta.get('input_cost', 'N/A'),
        'output_cost': meta.get('output_cost', 'N/A')
    }
    for model_id, meta in GEMINI_MODELS_FULL.items()
}

OPENAI_MODELS_FULL = fetch_openai_models_from_linkup()
OPENAI_MODEL_METADATA = {
    model_id: {
        'context': meta.get('context', 'N/A'),
        'input_cost': meta.get('input_cost', 'N/A'),
        'output_cost': meta.get('output_cost', 'N/A')
    }
    for model_id, meta in OPENAI_MODELS_FULL.items()
}

# Get all available models for UI selection
from core.pricing import get_all_available_models
AVAILABLE_MODELS = get_all_available_models()

# Pruner instructions for Test 4
PRUNER_INSTRUCTIONS = \"\"\"
You are an expert AI assistant that analyzes conversational context to plan the next step.
Your goal is to identify the minimum essential context needed to answer the user's `new_question` and decide on the correct `action`.

AVAILABLE CONTEXT KEYS: ["instruction", "summary", "user_messages", "agent_responses", "tool_logs"]

Think step-by-step:
1. **Analyze the `new_question`**: What is the user's core intent?
2. **Review the `context` items**: Determine which of the available keys are directly relevant to the question.
3. **Choose the `action` based on this logic**:
   - `general_answer`: Simple questions that only need conversation summary
   - `kb_lookup`: Knowledge base queries that need instruction and tool logs
   - `tool_call`: Actions requiring full context (instruction, summary, tool logs)

Return your analysis in the specified JSON format.
\"\"\"

"""
    
    # Insert the definitions
    lines.insert(insert_idx, definitions)
    print(f"   Added variable definitions at line {insert_idx+1}")
    
    return lines

def main():
    print("🚀 Starting final cleanup...")
    print("=" * 60)
    
    # Create backup
    backup_file = create_backup()
    
    # Read the file
    with open('streamlit_test_v5.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    print(f"📊 Original file: {original_count} lines")
    
    # Find sections to remove
    print("\n🔍 Finding leftover code sections...")
    sections = find_section_boundaries(lines)
    
    if not sections:
        print("   No leftover sections found. Checking for specific patterns...")
        # Manual fallback - find specific problematic lines
        for i, line in enumerate(lines):
            if "if gemini_models:" in line and i > 500 and i < 1000:
                # Found leftover code - remove from here to next valid section
                for j in range(i, min(i+500, len(lines))):
                    if lines[j].strip().startswith("async def run_live_smoke_test"):
                        sections = [(i, j)]
                        print(f"   Found leftover code: lines {i+1} to {j}")
                        break
                break
    
    # Remove sections
    if sections:
        print("\n🗑️  Removing leftover code...")
        lines = remove_sections(lines, sections)
    else:
        print("\n✓ No sections to remove")
    
    # Add missing definitions
    print("\n📝 Adding missing variable definitions...")
    lines = add_missing_definitions(lines)
    
    # Write the cleaned file
    with open('streamlit_test_v5.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    new_count = len(lines)
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"   Original lines: {original_count}")
    print(f"   New line count: {new_count}")
    print(f"   Lines removed: {original_count - new_count}")
    print(f"   Reduction: {(original_count - new_count) / original_count * 100:.1f}%")
    
    # Validate syntax
    print("\n🔍 Validating syntax...")
    import subprocess
    result = subprocess.run(['python', '-m', 'py_compile', 'streamlit_test_v5.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Syntax validation PASSED!")
    else:
        print("   ❌ Syntax validation FAILED:")
        print(result.stderr)
        print(f"\n⚠️  Restoring from backup: {backup_file}")
        shutil.copy(backup_file, "streamlit_test_v5.py")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Final cleanup complete!")
    print(f"💾 Backup saved as: {backup_file}")
    print(f"📊 Final file size: {new_count} lines")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready for testing!")
    else:
        print("\n❌ Cleanup failed - file restored from backup")

