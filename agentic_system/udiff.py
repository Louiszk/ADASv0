# modified from https://github.com/Aider-AI/aider/blob/main/aider/coders/udiff_coder.py
import difflib
from pathlib import Path
import re
import os

no_match_error = """UnifiedDiffNoMatch: hunk failed to apply!

The system does not contain lines that match the diff you provided!
Try again with a smaller, more targeted diff.
DO NOT skip blank lines, comments, docstrings, etc!
The diff needs to apply cleanly to the lines of the current code!
"""

not_unique_error = """UnifiedDiffNotUnique: hunk failed to apply!

The system contains multiple sets of lines that match the diff you provided!
Try again with a smaller, more targeted diff.
The diff needs to apply to a unique set of lines in the file!
"""

class SearchTextNotUnique(Exception):
    """Raised when search text appears multiple times in the content."""
    pass

def find_diffs(content):
    """Find diffs in content, supporting various diff formats."""
    if not content.endswith("\n"):
        content = content + "\n"

    lines = content.splitlines(keepends=True)
    
    # Check for diff blocks in markdown
    if any(line.startswith("```diff") for line in lines):
        return find_diffs_in_markdown(content)
    
    # Check for standard unified diff format with @@ markers
    if any(line.startswith("@@ ") for line in lines):
        return parse_unified_diff(lines)
    
    # Try to parse as simple +/- changes without hunk markers
    if any(line.startswith("+") or line.startswith("-") for line in lines):
        synthetic_hunk = ["@@ -1,1 +1,1 @@\n"] 
        synthetic_hunk.extend([l for l in lines if l.startswith("+") or l.startswith("-") or l.startswith(" ")])
        return [(None, synthetic_hunk)]
    
    return []

def find_diffs_in_markdown(content):
    """Extract diffs from markdown code blocks."""
    lines = content.splitlines(keepends=True)
    line_num = 0
    edits = []
    
    while line_num < len(lines):
        if lines[line_num].startswith("```diff"):
            next_line_num, these_edits = process_fenced_block(lines, line_num + 1)
            edits += these_edits
            line_num = next_line_num
        else:
            line_num += 1
            
    return edits

def parse_unified_diff(lines):
    """Parse lines in standard unified diff format."""
    edits = []
    current_hunk = []
    fname = None
    
    for line in lines:
        if line.startswith("@@ "):
            # Start a new hunk
            if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
                edits.append((fname, current_hunk))
            current_hunk = [line]
        elif line.startswith("--- "):
            # File indicator (old)
            continue
        elif line.startswith("+++ "):
            # File indicator (new)
            fname = line[4:].strip()
            current_hunk = []
        elif current_hunk or line.startswith(" ") or line.startswith("+") or line.startswith("-"):
            current_hunk.append(line)
            
    # Add the last hunk if it exists
    if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
        edits.append((fname, current_hunk))
        
    return edits

def process_fenced_block(lines, start_line_num):
    """Process a fenced code block containing a diff."""
    end_line_num = len(lines)
    
    # Find the end of the markdown block
    for line_num in range(start_line_num, len(lines)):
        if lines[line_num].startswith("```"):
            end_line_num = line_num
            break
    
    # Extract the block content
    block = lines[start_line_num:end_line_num]
    block.append("@@ @@")  # Add a marker to ensure the last hunk is processed
    
    # Process the block as a diff
    edits = []
    fname = None
    hunk = []
    keeper = False
    
    for line in block:
        hunk.append(line)
        
        if len(line) < 2:
            continue
            
        if line.startswith("+++ ") and len(hunk) >= 2 and hunk[-2].startswith("--- "):
            if len(hunk) > 2 and hunk[-3] == "\n":
                hunk = hunk[:-3]
            else:
                hunk = hunk[:-2]
                
            edits.append((fname, hunk))
            hunk = []
            keeper = False
            
            fname = line[4:].strip()
            continue
            
        op = line[0]
        if op in "-+":
            keeper = True
            continue
        if op != "@":
            continue
        if not keeper:
            hunk = []
            continue
            
        hunk = hunk[:-1]  # Remove the @@ line
        edits.append((fname, hunk))
        hunk = []
        keeper = False
    
    return end_line_num + 1, edits

def hunk_to_before_after(hunk, lines=False):
    """Convert a diff hunk to before and after strings or lists of lines."""
    before = []
    after = []
    
    for line in hunk:
        if not isinstance(line, str) or len(line) < 1:
            continue
            
        op = line[0]
        rest = line[1:] if len(line) > 1 else ""
        
        if op == " ":
            before.append(rest)
            after.append(rest)
        elif op == "-":
            before.append(rest)
        elif op == "+":
            after.append(rest)
    
    if lines:
        return before, after
    
    before_text = "".join(before)
    after_text = "".join(after)
    
    return before_text, after_text

def normalize_hunk(hunk):
    """Normalize a hunk by cleaning up whitespace."""
    before, after = hunk_to_before_after(hunk, lines=True)
    
    # Clean up pure whitespace lines
    before = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n")))] for line in before]
    after = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n")))] for line in after]
    
    diff = difflib.unified_diff(before, after, n=max(len(before), len(after)))
    try:
        diff = list(diff)[3:]  # Skip the header lines
        return diff
    except IndexError:
        return hunk  # Return original if normalization fails

def directly_apply_hunk(content, hunk):
    """Try to directly apply a hunk to content with simple search and replace."""
    before_text, after_text = hunk_to_before_after(hunk)
    
    if not before_text:
        return None
    
    # Simple case: exact match
    if before_text in content:
        if content.count(before_text) > 1:
            raise SearchTextNotUnique()
        return content.replace(before_text, after_text)
    
    # Try with whitespace normalization
    normalized_content = re.sub(r'\s+', ' ', content)
    normalized_before = re.sub(r'\s+', ' ', before_text)
    
    if normalized_before in normalized_content:
        # Find the actual match position in original content
        first_line = before_text.splitlines()[0] if before_text.splitlines() else ""
        if first_line and first_line in content:
            start_idx = content.find(first_line)
            if start_idx >= 0:
                # Try to locate the exact block
                potential_end = start_idx + len(before_text)
                if potential_end <= len(content):
                    potential_match = content[start_idx:potential_end]
                    if re.sub(r'\s+', ' ', potential_match) == normalized_before:
                        return content[:start_idx] + after_text + content[potential_end:]
    
    return None

def apply_partial_hunk(content, hunk):
    """Apply a hunk by trying with varying amounts of context."""
    # Split into context sections and changes
    ops = [line[0] if len(line) > 0 else " " for line in hunk]
    ops_str = "".join(ops).replace("-", "x").replace("+", "x")
    
    sections = []
    current_type = " "
    current_section = []
    
    for i, op in enumerate(ops_str):
        if op != current_type:
            if current_section:
                sections.append((current_type, current_section))
            current_type = op
            current_section = []
        current_section.append(hunk[i])
    
    if current_section:
        sections.append((current_type, current_section))
    
    # For each change section, try with varying amounts of context
    content_result = content
    all_applied = True
    
    for i, (section_type, section) in enumerate(sections):
        if section_type != "x":
            continue
            
        # Get preceding and following context
        preceding_ctx = []
        following_ctx = []
        
        if i > 0:
            preceding_ctx = sections[i-1][1]
        if i < len(sections) - 1:
            following_ctx = sections[i+1][1]
        
        # Try with progressively less context
        applied = False
        for p_ctx_amount in range(len(preceding_ctx), -1, -1):
            if applied: break
            
            p_ctx = preceding_ctx[-p_ctx_amount:] if p_ctx_amount else []
            
            for f_ctx_amount in range(len(following_ctx), -1, -1):
                f_ctx = following_ctx[:f_ctx_amount] if f_ctx_amount else []
                
                partial_hunk = p_ctx + section + f_ctx
                try:
                    result = directly_apply_hunk(content_result, partial_hunk)
                    if result:
                        content_result = result
                        applied = True
                        break
                except SearchTextNotUnique:
                    continue
        
        if not applied:
            all_applied = False
            break
    
    return content_result if all_applied else None

def do_replace(fname, content, hunk):
    """Apply a hunk to content with various strategies."""
    before_text, after_text = hunk_to_before_after(hunk)
    
    # Handle new files
    if not os.path.exists(fname) and not before_text.strip():
        Path(fname).touch()
        return after_text
        
    # Handle append case
    if not before_text.strip():
        return content + after_text
    
    try:
        # Try direct application first
        result = directly_apply_hunk(content, hunk)
        if result:
            return result
        
        # Try with normalized hunk
        normalized_hunk = normalize_hunk(hunk)
        if normalized_hunk != hunk:
            result = directly_apply_hunk(content, normalized_hunk)
            if result:
                return result
        
        # Try partial application with context
        result = apply_partial_hunk(content, hunk)
        if result:
            return result
        
        # Last resort: try line-by-line matching
        before_lines, after_lines = hunk_to_before_after(hunk, lines=True)
        if before_lines:
            content_lines = content.splitlines(True)
            for i in range(len(content_lines) - len(before_lines) + 1):
                content_block = ''.join(content_lines[i:i+len(before_lines)])
                if content_block.strip() == ''.join(before_lines).strip():
                    new_content = content_lines[:i] + after_lines + content_lines[i+len(before_lines):]
                    return ''.join(new_content)
    
    except SearchTextNotUnique:
        raise
    except Exception:
        pass
    
    return None