"""
Aggressive extraction script to reduce streamlit_test_v5.py to under 6,500 lines.

This script will:
1. Extract UnifiedOrchestrator class to core/unified_orchestrator.py
2. Extract test execution functions to core/test_runners.py
3. Remove visualization functions (already extracted)
4. Add necessary imports
5. Create backup before making changes
"""

import re
import shutil
from pathlib import Path

def create_backup():
    """Create timestamped backup."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"streamlit_test_v5.py.backup_{timestamp}"
    shutil.copy("streamlit_test_v5.py", backup_name)
    print(f"✅ Created backup: {backup_name}")
    return backup_name

def extract_class_from_file(lines, class_name, start_line_idx):
    """Extract a complete class definition from lines starting at start_line_idx."""
    class_lines = [lines[start_line_idx]]  # Include the class definition line
    indent_level = len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip())
    
    i = start_line_idx + 1
    while i < len(lines):
        line = lines[i]
        current_indent = len(line) - len(line.lstrip())
        
        # If we hit a non-empty, non-comment line with same or less indentation, we're done
        if line.strip() and not line.strip().startswith('#'):
            if current_indent <= indent_level and not line.strip().startswith(('"""', "'''")):
                # Check if it's a new class or function at module level
                if line.strip().startswith(('class ', 'def ', 'async def ')):
                    break
        
        class_lines.append(line)
        i += 1
    
    return class_lines, i

def extract_function_from_file(lines, func_name, start_line_idx):
    """Extract a complete function definition from lines starting at start_line_idx."""
    func_lines = [lines[start_line_idx]]  # Include the def line
    indent_level = len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip())
    
    i = start_line_idx + 1
    while i < len(lines):
        line = lines[i]
        current_indent = len(line) - len(line.lstrip())
        
        # If we hit a non-empty, non-comment line with same or less indentation, we're done
        if line.strip() and not line.strip().startswith('#'):
            if current_indent <= indent_level:
                # Check if it's a new function or class
                if line.strip().startswith(('def ', 'async def ', 'class ', '@')):
                    break
        
        func_lines.append(line)
        i += 1
    
    return func_lines, i

def main():
    print("🚀 Starting aggressive extraction...")
    print("=" * 60)
    
    # Create backup
    backup_file = create_backup()
    
    # Read the main file
    with open('streamlit_test_v5.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    print(f"📊 Original file: {original_count} lines")
    
    # Track what to extract
    unified_orchestrator_lines = []
    test_runner_lines = []
    visualization_functions = [
        'render_model_comparison_chart',
        'render_organized_results',
        'render_progress_replay',
        'render_agent_gantt_chart',
        'render_test5_gantt_chart',
        'render_universal_gantt_chart',
        'render_task_cards',
        'render_single_task_card',
        'render_live_agent_status',
        'render_agent_task_cards'
    ]
    
    test_runner_functions = [
        'run_classification_flow',
        'run_tool_sequence_test',
        'run_pruning_test'
    ]
    
    # Find and extract UnifiedOrchestrator
    print("\n🔍 Searching for UnifiedOrchestrator class...")
    for i, line in enumerate(lines):
        if re.match(r'^class UnifiedOrchestrator', line):
            print(f"   Found at line {i+1}")
            unified_orchestrator_lines, end_idx = extract_class_from_file(lines, 'UnifiedOrchestrator', i)
            print(f"   Extracted {len(unified_orchestrator_lines)} lines (ends at line {end_idx})")
            break
    
    # Find and extract test runner functions
    print("\n🔍 Searching for test runner functions...")
    for func_name in test_runner_functions:
        for i, line in enumerate(lines):
            if re.match(rf'^(async )?def {func_name}\(', line):
                print(f"   Found {func_name} at line {i+1}")
                func_lines, end_idx = extract_function_from_file(lines, func_name, i)
                test_runner_lines.extend(func_lines)
                print(f"   Extracted {len(func_lines)} lines")
                break
    
    # Mark lines to remove
    lines_to_remove = set()
    
    # Mark UnifiedOrchestrator for removal
    if unified_orchestrator_lines:
        for i, line in enumerate(lines):
            if re.match(r'^class UnifiedOrchestrator', line):
                _, end_idx = extract_class_from_file(lines, 'UnifiedOrchestrator', i)
                for j in range(i, end_idx):
                    lines_to_remove.add(j)
                print(f"\n✓ Marked UnifiedOrchestrator for removal ({end_idx - i} lines)")
                break
    
    # Mark visualization functions for removal
    print("\n🔍 Marking visualization functions for removal...")
    for func_name in visualization_functions:
        for i, line in enumerate(lines):
            if re.match(rf'^def {func_name}\(', line):
                _, end_idx = extract_function_from_file(lines, func_name, i)
                for j in range(i, end_idx):
                    lines_to_remove.add(j)
                print(f"   Marked {func_name} ({end_idx - i} lines)")
                break
    
    # Mark test runner functions for removal
    print("\n🔍 Marking test runner functions for removal...")
    for func_name in test_runner_functions:
        for i, line in enumerate(lines):
            if re.match(rf'^(async )?def {func_name}\(', line):
                _, end_idx = extract_function_from_file(lines, func_name, i)
                for j in range(i, end_idx):
                    lines_to_remove.add(j)
                print(f"   Marked {func_name} ({end_idx - i} lines)")
                break
    
    # Create new file with removed lines
    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"   Original lines: {original_count}")
    print(f"   Lines to remove: {len(lines_to_remove)}")
    print(f"   New line count: {len(new_lines)}")
    print(f"   Reduction: {original_count - len(new_lines)} lines ({(original_count - len(new_lines)) / original_count * 100:.1f}%)")
    
    # Write the cleaned file
    with open('streamlit_test_v5.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ Main file updated!")
    
    # Save extracted content to new files
    if unified_orchestrator_lines:
        print(f"\n📝 Creating core/unified_orchestrator.py...")
        with open('core/unified_orchestrator.py', 'w', encoding='utf-8') as f:
            f.write('"""\nUnifiedOrchestrator class extracted from streamlit_test_v5.py\n"""\n\n')
            f.write('# Add necessary imports here\n')
            f.write('import asyncio\nimport json\nfrom typing import List, Dict, Optional, Any\n')
            f.write('from dataclasses import dataclass\nimport google.generativeai as genai\nimport streamlit as st\n\n')
            f.writelines(unified_orchestrator_lines)
        print(f"   ✅ Created ({len(unified_orchestrator_lines)} lines)")
    
    if test_runner_lines:
        print(f"\n📝 Creating core/test_runners.py...")
        with open('core/test_runners.py', 'w', encoding='utf-8') as f:
            f.write('"""\nTest execution functions extracted from streamlit_test_v5.py\n"""\n\n')
            f.write('# Add necessary imports here\n')
            f.write('import asyncio\nimport streamlit as st\nimport pandas as pd\nfrom typing import Optional\n\n')
            f.writelines(test_runner_lines)
        print(f"   ✅ Created ({len(test_runner_lines)} lines)")
    
    print("\n" + "=" * 60)
    print("🎉 Extraction complete!")
    print(f"💾 Backup saved as: {backup_file}")
    print("\n⚠️  Next steps:")
    print("   1. Add proper imports to core/unified_orchestrator.py")
    print("   2. Add proper imports to core/test_runners.py")
    print("   3. Add imports to streamlit_test_v5.py for extracted modules")
    print("   4. Test the application")

if __name__ == "__main__":
    main()

