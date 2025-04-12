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
    Normalize line indentation to multiples of 4 spaces.
    """
    if not line or not line.strip():
        return line
        
    # Count leading spaces
    leading_spaces = len(line) - len(line.lstrip(' '))
    
    if leading_spaces == 0 or leading_spaces % 4 == 0:
        return line  # Already a multiple of 4
    
    remainder = leading_spaces % 4
    
    if remainder == 1:  # 1, 5, 9, ...
        normalized_spaces = (leading_spaces // 4) * 4  # Round down to previous multiple of 4
    elif remainder in [2, 3]:  # 2, 3, 6, 7, 10, 11, ...
        normalized_spaces = ((leading_spaces // 4) + 1) * 4  # Round up to next multiple of 4
    
    # Replace the indentation
    return ' ' * normalized_spaces + line.lstrip(' ')

def find_diffs(content):
    """Find diffs in content, supporting various diff formats."""
    content = normalize_line_endings(content)
    
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
    content = normalize_line_endings(content)
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
        
        # Normalize indentation to prevent common errors
        rest = normalize_indentation(rest)
        
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

def fuzzy_match_block(content, search_block, threshold=0.8):
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
    
    # Try fuzzy matching if exact match fails
    matched_text, match_ratio = fuzzy_match_block(content, before_text)
    if matched_text and match_ratio >= 0.9:
        if content.count(matched_text) > 1:
            raise SearchTextNotUnique()
        return content.replace(matched_text, after_text)
    
    return None

def extract_context_sections(hunk):
    """Extract context and change sections from a hunk."""
    sections = []
    current_type = None
    current_section = []
    
    for line in hunk:
        if not line or len(line) < 1:
            continue
            
        line_type = line[0]
        if line_type != current_type:
            if current_section:
                sections.append((current_type, current_section))
            current_type = line_type
            current_section = []
        current_section.append(line)
    
    if current_section:
        sections.append((current_type, current_section))
    
    return sections

def apply_partial_hunk(content, hunk):
    """Apply a hunk by trying with varying amounts of context."""
    content = normalize_line_endings(content)
    
    # Extract sections by type (context vs. changes)
    sections = extract_context_sections(hunk)
    
    # Try to apply each section of changes
    content_result = content
    all_applied = True
    
    for i, (section_type, section) in enumerate(sections):
        if section_type not in "+-":
            continue
            
        # Get preceding and following context
        preceding_ctx = []
        following_ctx = []
        
        if i > 0 and sections[i-1][0] == " ":
            preceding_ctx = sections[i-1][1]
        if i < len(sections) - 1 and sections[i+1][0] == " ":
            following_ctx = sections[i+1][1]
        
        # Create mini-hunk with just this section and its context
        mini_hunk = preceding_ctx + section + following_ctx
        
        try:
            # Try to apply just this mini-hunk
            result = directly_apply_hunk(content_result, mini_hunk)
            if result:
                content_result = result
            else:
                # Try with less context
                for p_ctx_amount in range(len(preceding_ctx), -1, -1):
                    p_ctx = preceding_ctx[-p_ctx_amount:] if p_ctx_amount else []
                    
                    for f_ctx_amount in range(len(following_ctx), -1, -1):
                        f_ctx = following_ctx[:f_ctx_amount] if f_ctx_amount else []
                        partial_hunk = p_ctx + section + f_ctx
                        
                        try:
                            inner_result = directly_apply_hunk(content_result, partial_hunk)
                            if inner_result:
                                content_result = inner_result
                                break
                        except SearchTextNotUnique:
                            continue
                    
                    if content_result != content:  # If something changed
                        break
        except SearchTextNotUnique:
            all_applied = False
    
    return content_result if all_applied or content_result != content else None

def do_replace(fname, content, hunk):
    """Apply a hunk to content with various strategies."""
    content = normalize_line_endings(content)
    
    # Normalize indentation in the hunk before applying
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
        
        # Last resort: try line-by-line matching with fuzzy comparison
        before_lines, after_lines = hunk_to_before_after(hunk, lines=True)
        if before_lines:
            content_lines = content.splitlines(True)
            
            for i in range(len(content_lines) - len(before_lines) + 1):
                content_block = ''.join(content_lines[i:i+len(before_lines)])
                
                # Use fuzzy matching for the block
                similarity = difflib.SequenceMatcher(None, content_block.strip(), ''.join(before_lines).strip()).ratio()
                if similarity >= 0.8:  # 80% similarity threshold
                    new_content = content_lines[:i] + after_lines + content_lines[i+len(before_lines):]
                    return ''.join(new_content)
    
    except SearchTextNotUnique:
        raise
    except Exception:
        pass
    
    return None