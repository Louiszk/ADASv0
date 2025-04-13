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
    pass

def normalize_line_endings(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')

def normalize_indentation(line):
    if not line or not line.strip():
        return line
        
    leading_spaces = len(line) - len(line.lstrip(' '))
    indent_level = round(leading_spaces / 4)
    
    return ' ' * (indent_level * 4) + line.lstrip(' ')

def find_diffs(content):
    content = normalize_line_endings(content)
    
    if not content.endswith("\n"):
        content = content + "\n"

    lines = content.splitlines(keepends=True)
    
    hunks = []
    current_hunk = []
    
    for i, line in enumerate(lines):
        if line.startswith("@@ "):
            if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
                hunks.append((None, current_hunk))
            current_hunk = [line]
        elif current_hunk and (line.startswith(" ") or line.startswith("+") or line.startswith("-") or not line.strip()):
            current_hunk.append(line)
    
    if current_hunk and any(l.startswith("+") or l.startswith("-") for l in current_hunk):
        hunks.append((None, current_hunk))
    
    return hunks

def hunk_to_before_after(hunk, lines=False):
    before = []
    after = []
    
    hunk_lines_to_process = hunk[1:] if hunk and hunk[0].startswith("@@ ") else hunk
    
    for line in hunk_lines_to_process:
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
    before, after = hunk_to_before_after(hunk, lines=True)
    
    before = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n"))):] for line in before]
    after = [line if line.strip() else line[-(len(line) - len(line.rstrip("\r\n"))):] for line in after]
    
    before_normalized = [normalize_indentation(line) for line in before]
    after_normalized = [normalize_indentation(line) for line in after]
    
    diff = difflib.unified_diff(before_normalized, after_normalized, n=max(len(before), len(after)))
    try:
        normalized_hunk_lines = list(diff)[2:]
        return normalized_hunk_lines
    except IndexError:
        return hunk

def fuzzy_match_block(content, search_block, threshold=0.95):
    content_lines = content.splitlines()
    search_lines = search_block.splitlines()
    
    if not search_lines:
        return None, 0
    
    best_match_block = None
    best_ratio = 0
    match_indices = []
    
    for i in range(len(content_lines) - len(search_lines) + 1):
        content_block_lines = content_lines[i:i + len(search_lines)]
        content_block = "\n".join(content_block_lines)
        matcher = difflib.SequenceMatcher(None, content_block, search_block)
        ratio = matcher.ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_block = content_block
            match_indices = [i]
        elif ratio == best_ratio and ratio >= threshold:
            match_indices.append(i)
    
    if best_ratio >= threshold:
        if len(match_indices) > 1:
            raise SearchTextNotUnique("Fuzzy match not unique")
        return best_match_block, best_ratio
    
    return None, best_ratio

def directly_apply_hunk(content, hunk):
    content = normalize_line_endings(content)
    before_text, after_text = hunk_to_before_after(hunk)
    
    if not before_text and not after_text:
        return content
    if not before_text and after_text:
        return None
    
    match_count = content.count(before_text)
    if match_count > 0:
        if match_count > 1:
            raise SearchTextNotUnique("Exact match not unique")
        return content.replace(before_text, after_text)
    
    content_lines = content.splitlines(keepends=True)
    before_lines = before_text.splitlines()
    
    if not before_lines:
        return None
    
    match_indices = []
    for i in range(len(content_lines) - len(before_lines) + 1):
        match = True
        for j, before_line in enumerate(before_lines):
            content_line = content_lines[i+j]
            if content_line.strip() != before_line.strip():
                match = False
                break
        if match:
            match_indices.append(i)
    
    if len(match_indices) == 1:
        i = match_indices[0]
        content_prefix = "".join(content_lines[:i])
        content_suffix = "".join(content_lines[i+len(before_lines):])
        result = content_prefix + after_text + content_suffix
        return result
    elif len(match_indices) > 1:
        raise SearchTextNotUnique("Whitespace-normalized match not unique")
    
    try:
        matched_text, match_ratio = fuzzy_match_block(content, before_text)
        if matched_text:
            return content.replace(matched_text, after_text)
    except SearchTextNotUnique:
        raise
    
    return None

def apply_partial_hunk(content, hunk):
    content = normalize_line_endings(content)
    
    sections = []
    current_section = []
    current_op_type = None
    
    hunk_lines_to_process = hunk[1:] if hunk and hunk[0].startswith("@@ ") else hunk
    
    for line in hunk_lines_to_process:
        if not isinstance(line, str) or len(line) < 1:
            continue
        
        op = line[0]
        op_type = " " if op == " " else "x"
        
        if op_type != current_op_type:
            if current_section:
                sections.append((current_op_type, current_section))
            current_section = []
            current_op_type = op_type
        
        current_section.append(line)
    
    if current_section:
        sections.append((current_op_type, current_section))
    
    all_change_sections_applied = True
    modified_content = content
    change_sections_indices = [i for i, (op_type, _) in enumerate(sections) if op_type == 'x']
    
    if not change_sections_indices:
        return content
    
    applied_change_sections = 0
    for i in change_sections_indices:
        changes = sections[i][1]
        
        preceding_ctx = []
        if i > 0 and sections[i-1][0] == " ":
            preceding_ctx = sections[i-1][1]
        
        following_ctx = []
        if i + 1 < len(sections) and sections[i+1][0] == " ":
            following_ctx = sections[i+1][1]
        
        section_applied_successfully = False
        context_options = [
            (len(preceding_ctx), len(following_ctx)),
            (len(preceding_ctx) // 2, len(following_ctx) // 2),
            (1 if preceding_ctx else 0, 1 if following_ctx else 0),
            (0, 0)
        ]
        unique_context_options = sorted(list(set(context_options)), key=sum, reverse=True)
        if (0,0) in unique_context_options and (len(preceding_ctx) > 0 or len(following_ctx) > 0):
            pass
        elif not (len(preceding_ctx) > 0 or len(following_ctx) > 0):
            unique_context_options = [(0,0)]
        
        for before_size, after_size in unique_context_options:
            b_ctx = preceding_ctx[-before_size:] if before_size > 0 else []
            a_ctx = following_ctx[:after_size] if after_size > 0 else []
            
            mini_hunk = ["@@ @@"] + b_ctx + changes + a_ctx
            
            try:
                result = directly_apply_hunk(modified_content, mini_hunk)
                if result is not None:
                    modified_content = result
                    section_applied_successfully = True
                    applied_change_sections += 1
                    break
            except SearchTextNotUnique:
                continue
            except Exception:
                continue
        
        if not section_applied_successfully:
            all_change_sections_applied = False
            break
    
    if all_change_sections_applied:
        return modified_content
    else:
        return None

def do_replace(fname, content, hunk):
    original_content = content
    content = normalize_line_endings(content)
    
    normalized_hunk = []
    hunk_lines_to_process = hunk
    if hunk and hunk[0].startswith("@@ "):
        normalized_hunk.append(hunk[0])
        hunk_lines_to_process = hunk[1:]
    
    for line in hunk_lines_to_process:
        if line and len(line) > 0 and line[0] in ' +-':
            op = line[0]
            rest = line[1:]
            normalized_rest = normalize_indentation(rest)
            normalized_hunk.append(op + normalized_rest)
        else:
            normalized_hunk.append(line)
    
    before_text, after_text = hunk_to_before_after(normalized_hunk)
    
    if not before_text and not after_text:
        return original_content
    
    if not os.path.exists(fname) and not before_text.strip():
        return after_text
    
    try:
        result = directly_apply_hunk(content, normalized_hunk)
        if result is not None:
            return result
        
        result = apply_partial_hunk(content, normalized_hunk)
        if result is not None:
            return result
        
    except SearchTextNotUnique:
        raise
    
    except Exception:
        pass
    
    return None