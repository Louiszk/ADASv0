# modified from https://github.com/Aider-AI/aider/blob/main/aider/coders/udiff_coder.py
from pathlib import Path

# Import the core standalone functions from udiff_coder
from .udiff_coder import (
    SearchTextNotUnique,
    do_replace,
    hunk_to_before_after,
    normalize_hunk
)

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

other_hunks_applied = (
    "Note: some hunks did apply successfully."
)

def find_diffs(content):
    """
    Parses content containing potentially multiple sequential unified diff hunks, each starting with '@@ '.
    Returns a list of tuples: [(None, hunk_lines), ...].
    """
    if not content.endswith("\n"):
        content = content + "\n"

    lines = content.splitlines(keepends=True)
    edits = []
    line_num = 0

    while line_num < len(lines):
        # Find the start of the next hunk
        while line_num < len(lines) and not lines[line_num].startswith("@@ "):
            line_num += 1

        if line_num >= len(lines):
            break

        start_of_hunk = line_num
        current_hunk_lines = [lines[start_of_hunk]]
        line_num += 1

        while line_num < len(lines) and not lines[line_num].startswith("@@ "):
            current_hunk_lines.append(lines[line_num])
            line_num += 1

        # Add the collected hunk if it actually contains changes
        if any(l.strip().startswith(("+", "-")) for l in current_hunk_lines[1:]):
             edits.append((None, current_hunk_lines))

    # Fallback only if NO '@@ ' hunks were found at all
    if not edits and any(line.startswith(("+", "-")) for line in lines):
         if all(line.startswith((" ", "+", "-")) or not line.strip() for line in lines):
              print("Warning: Treating entire input as a single raw diff hunk (no '@@ ' found).")
              edits.append((None, lines))

    return edits

def apply_unified_diff(diff_text, original_content, target_filename="target_system.py"):
    edits = find_diffs(diff_text)
    if not edits:
        print("No diff hunks found in the provided text.")
        return original_content, 0, 0 # content, applied_count, total_count

    current_content = original_content
    total_hunks = len(edits)
    applied_hunks_count = 0
    skipped_hunks_count = 0
    errors = []
    seen_hunks = set()
    target_file_path = Path(target_filename)

    print(f"Found {total_hunks} potential hunks.")

    for i, (_, raw_hunk) in enumerate(edits):
        if not raw_hunk:
             print(f"Skipping potentially empty hunk {i+1}/{total_hunks}")
             continue

        normalized = normalize_hunk(raw_hunk)
        if not normalized:
            print(f"Skipping empty hunk {i+1}/{total_hunks} after normalization")
            continue

        hunk_key = "".join(normalized)
        if hunk_key in seen_hunks:
            print(f"Skipping duplicate hunk {i+1}/{total_hunks}")
            continue
        seen_hunks.add(hunk_key)

        content_before_hunk = current_content

        try:
            result_content = do_replace(target_file_path, current_content, normalized)

            if result_content is not None and result_content != content_before_hunk:
                current_content = result_content
                applied_hunks_count += 1
                print(f"Successfully applied hunk {i+1}/{total_hunks}")
            elif result_content is None:
                original_hunk_text, _ = hunk_to_before_after(normalized)
                num_lines = len(original_hunk_text.splitlines()) if original_hunk_text else 0
                error_detail = no_match_error.format(path=target_filename, original=original_hunk_text[:200] if original_hunk_text else "N/A", num_lines=num_lines)
                errors.append(f"Hunk {i+1}/{total_hunks} failed: No Match\n{error_detail}")
                print(f"Failed to apply hunk {i+1}/{total_hunks}: No Match")
            else:
                 skipped_hunks_count += 1
                 print(f"Skipped hunk {i+1}/{total_hunks}: No change detected after application.")


        except SearchTextNotUnique:
            original_hunk_text, _ = hunk_to_before_after(normalized)
            num_lines = len(original_hunk_text.splitlines()) if original_hunk_text else 0
            error_detail = not_unique_error.format(path=target_filename, original=original_hunk_text[:200] if original_hunk_text else "N/A", num_lines=num_lines)
            errors.append(f"Hunk {i+1}/{total_hunks} failed: Not Unique\n{error_detail}")
            print(f"Failed to apply hunk {i+1}/{total_hunks}: Not Unique")
        except Exception as e:
             errors.append(f"Hunk {i+1}/{total_hunks} failed: Unexpected Error - {repr(e)}")
             print(f"Failed to apply hunk {i+1}/{total_hunks}: Unexpected Error - {repr(e)}")
             import traceback
             traceback.print_exc()

    if errors:
        error_message = "\n---\n".join(errors)
        if applied_hunks_count > 0 and applied_hunks_count < total_hunks:
            error_message += "\n\n" + other_hunks_applied
        raise ValueError(error_message)

    final_applied = applied_hunks_count
    final_total = total_hunks - skipped_hunks_count

    print(f"Applied {final_applied} hunks, Skipped {skipped_hunks_count} hunks out of {total_hunks} found.")
    return current_content, final_applied, final_total