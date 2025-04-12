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

def normalize_line_endings(text):
    """Normalize line endings to LF (\n)."""
    return text.replace('\r\n', '\n').replace('\r', '\n')

def normalize_indentation(line):
    """
    Normalize line indentation to exactly 4 spaces per level.
    """
    if not line or not line.strip():
        return line
        
    leading_spaces = len(line) - len(line.lstrip(' '))
    indent_level = round(leading_spaces / 4)
    
    return ' ' * (indent_level * 4) + line.lstrip(' ')

def find_diffs(content):
    """Find diffs in content"""
    content = normalize_line_endings(content)
    
    if not content.endswith("\n"):
        content = content + "\n"

    lines = content.splitlines(keepends=True)
    
    # Check for @@ @@ markers
    hunks = []
    current_hunk = []
    
    for line in lines:
        if line.startswith("@@ "):
            if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
                hunks.append((None, current_hunk))
            current_hunk = [line]
        elif current_hunk and (line.startswith(" ") or line.startswith("+") or line.startswith("-") or not line.strip()):
            current_hunk.append(line)
    
    # Add the last hunk if it exists
    if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
        hunks.append((None, current_hunk))
    
    return hunks

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
    """Normalize a hunk by cleaning up whitespace and indentation."""
    before, after = hunk_to_before_after(hunk, lines=True)
    
    # Clean up pure whitespace lines
    before = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n")))] for line in before]
    after = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n")))] for line in after]
    
    # Normalize indentation
    before = [normalize_indentation(line) for line in before]
    after = [normalize_indentation(line) for line in after]
    
    diff = difflib.unified_diff(before, after, n=max(len(before), len(after)))
    try:
        diff = list(diff)[3:]  # Skip the header lines
        return diff
    except IndexError:
        return hunk  # Return original if normalization fails

def fuzzy_match_block(content, search_block, threshold=0.95):
    """Find the best match for a block of text within content using fuzzy matching."""
    content_lines = content.splitlines()
    search_lines = search_block.splitlines()
    
    if not search_lines:
        return None, 0
    
    best_match = None
    best_ratio = 0
    
    # Try to find a matching section
    for i in range(len(content_lines) - len(search_lines) + 1):
        content_block = "\n".join(content_lines[i:i + len(search_lines)])
        ratio = difflib.SequenceMatcher(None, content_block, search_block).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = content_block
    
    if best_ratio >= threshold:
        return best_match, best_ratio
    
    return None, best_ratio

def directly_apply_hunk(content, hunk):
    """Try to directly apply a hunk to content with simple search and replace."""
    content = normalize_line_endings(content)
    before_text, after_text = hunk_to_before_after(hunk)
    
    if not before_text:
        return None
    
    # Simple case: exact match
    if before_text in content:
        if content.count(before_text) > 1:
            raise SearchTextNotUnique()
        return content.replace(before_text, after_text)
    
    # Try with normalized whitespace
    content_lines = content.splitlines()
    before_lines = before_text.splitlines()
    
    if not before_lines:
        return None
    
    for i in range(len(content_lines) - len(before_lines) + 1):
        match = True
        for j, before_line in enumerate(before_lines):
            content_line = content_lines[i+j]
            # Ignore whitespace differences only for comparison
            if content_line.strip() != before_line.strip():
                match = False
                break
        
        if match:
            # Found a match - replace it
            content_before = "\n".join(content_lines[:i])
            content_after = "\n".join(content_lines[i+len(before_lines):])
            return content_before + ("\n" if content_before else "") + after_text + ("\n" if content_after else "") + content_after
    
    matched_text, match_ratio = fuzzy_match_block(content, before_text)
    if matched_text and match_ratio >= 0.95:  # Very high threshold for safety
        if content.count(matched_text) > 1:
            raise SearchTextNotUnique()
        return content.replace(matched_text, after_text)
    
    return None

def apply_partial_hunk(content, hunk):
    """Apply a hunk by trying with varying amounts of context, ensuring all parts are applied."""
    content = normalize_line_endings(content)
    
    # Group lines by operation type (context vs changes)
    sections = []
    current_section = []
    current_op_type = None  # " " for context, "x" for changes
    
    for line in hunk:
        if not isinstance(line, str) or len(line) < 1:
            continue
        
        op = line[0]
        op_type = " " if op == " " else "x"  # Group + and - together as changes
        
        if op_type != current_op_type:
            if current_section:
                sections.append((current_op_type, current_section))
            current_section = []
            current_op_type = op_type
        
        current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append((current_op_type, current_section))
    
    # Ensure the last section is a context section (for consistency)
    if sections and sections[-1][0] != " ":
        sections.append((" ", []))
    
    # Try to apply each context-changes-context triplet
    all_done = True
    modified_content = content
    
    for i in range(1, len(sections), 2):
        # Skip if this isn't a changes section
        if i >= len(sections) or sections[i][0] != "x":
            continue
        
        # Get preceding and following context
        preceding_ctx = []
        if i > 0:
            preceding_ctx = sections[i-1][1]
        
        changes = sections[i][1]
        
        following_ctx = []
        if i+1 < len(sections):
            following_ctx = sections[i+1][1]
        
        # Try to apply this section
        section_applied = False
        
        # Try with full context first
        mini_hunk = preceding_ctx + changes + following_ctx
        try:
            result = directly_apply_hunk(modified_content, mini_hunk)
            if result:
                modified_content = result
                section_applied = True
        except SearchTextNotUnique:
            pass
        
        # If full context didn't work, try with reduced context
        if not section_applied:
            for before_size in [len(preceding_ctx), len(preceding_ctx)//2, 1, 0]:
                if section_applied:
                    break
                    
                for after_size in [len(following_ctx), len(following_ctx)//2, 1, 0]:
                    if before_size == 0 and after_size == 0 and len(preceding_ctx) + len(following_ctx) > 0:
                        continue  # Skip if we have context but are trying without any
                    
                    b_ctx = preceding_ctx[-before_size:] if before_size else []
                    a_ctx = following_ctx[:after_size] if after_size else []
                    
                    mini_hunk = b_ctx + changes + a_ctx
                    try:
                        result = directly_apply_hunk(modified_content, mini_hunk)
                        if result:
                            modified_content = result
                            section_applied = True
                            break
                    except SearchTextNotUnique:
                        pass
        
        # If this section couldn't be applied, the whole hunk fails
        if not section_applied:
            all_done = False
            break
    
    # Only return the modified content if all sections were applied
    return modified_content if all_done else None

def do_replace(fname, content, hunk):
    """Apply a hunk to content with various strategies."""
    content = normalize_line_endings(content)
    
    # Normalize indentation in the hunk
    normalized_hunk = []
    for line in hunk:
        if line and len(line) > 1 and line[0] in ' +-':
            op = line[0]
            rest = normalize_indentation(line[1:])
            normalized_hunk.append(op + rest)
        else:
            normalized_hunk.append(line)
    
    before_text, after_text = hunk_to_before_after(normalized_hunk)
    
    # Handle new files
    if not os.path.exists(fname) and not before_text.strip():
        Path(fname).touch()
        return after_text
    
    try:
        # Try direct application first - exact match
        result = directly_apply_hunk(content, normalized_hunk)
        if result:
            return result
        
        # Try with more conservative partial application
        result = apply_partial_hunk(content, normalized_hunk)
        if result:
            return result
        
    except SearchTextNotUnique:
        raise
    except Exception:
        pass
    
    return None