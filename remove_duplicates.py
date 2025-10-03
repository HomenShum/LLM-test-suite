"""Remove duplicate code sections from streamlit_test_v5.py"""

import re

# Read the file
with open('streamlit_test_v5.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

print(f"Original file: {len(lines)} lines")

# Find the markers
# 1. Find where the first "# Pydantic models moved to core/models.py" comment is
# 2. Find where the UI code starts (st.set_page_config)

first_pydantic_comment = None
ui_start = None

for i, line in enumerate(lines):
    if "# Pydantic models moved to core/models.py" in line and first_pydantic_comment is None:
        first_pydantic_comment = i
    if line.strip().startswith("st.set_page_config"):
        ui_start = i
        break

print(f"First Pydantic comment at line {first_pydantic_comment + 1}")
print(f"UI starts at line {ui_start + 1}")

if first_pydantic_comment and ui_start:
    # Keep everything before the first Pydantic comment
    # Then skip to the UI code
    new_lines = lines[:first_pydantic_comment + 1]
    
    # Add a separator
    new_lines.append("")
    new_lines.append("# ============================================================")
    new_lines.append("# STREAMLIT UI CODE")
    new_lines.append("# ============================================================")
    new_lines.append("")
    new_lines.append("# ---------- UI ----------")
    
    # Add the UI code
    new_lines.extend(lines[ui_start:])
    
    # Write back
    with open('streamlit_test_v5.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    removed_lines = len(lines) - len(new_lines)
    print(f"\n✅ Removed {removed_lines} lines of duplicate code")
    print(f"New file: {len(new_lines)} lines")
    print(f"Removed lines {first_pydantic_comment + 2} to {ui_start}")
else:
    print("❌ Could not find markers")

