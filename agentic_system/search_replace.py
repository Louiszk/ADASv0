# trimmed from https://github.com/Aider-AI/aider/blob/main/aider/coders/search_replace.py 16.04.2025

import sys
from pathlib import Path

from diff_match_patch import diff_match_patch
from tqdm import tqdm

class RelativeIndenter:
    """Rewrites text files to have relative indentation, which involves
    reformatting the leading white space on lines.  This format makes
    it easier to search and apply edits to pairs of code blocks which
    may differ significantly in their overall level of indentation.

    It removes leading white space which is shared with the preceding
    line.

    Original:
    ```
            Foo # indented 8
                Bar # indented 4 more than the previous line
                Baz # same indent as the previous line
                Fob # same indent as the previous line
    ```

    Becomes:
    ```
            Foo # indented 8
        Bar # indented 4 more than the previous line
    Baz # same indent as the previous line
    Fob # same indent as the previous line
    ```

    If the current line is *less* indented then the previous line,
    uses a unicode character to indicate outdenting.

    Original
    ```
            Foo
                Bar
                Baz
            Fob # indented 4 less than the previous line
    ```

    Becomes:
    ```
            Foo
        Bar
    Baz
    ←←←←Fob # indented 4 less than the previous line
    ```

    This is a similar original to the last one, but every line has
    been uniformly outdented:
    ```
    Foo
        Bar
        Baz
    Fob # indented 4 less than the previous line
    ```

    It becomes this result, which is very similar to the previous
    result.  Only the white space on the first line differs.  From the
    word Foo onwards, it is identical to the previous result.
    ```
    Foo
        Bar
    Baz
    ←←←←Fob # indented 4 less than the previous line
    ```

    """

    def __init__(self, texts):
        """
        Based on the texts, choose a unicode character that isn't in any of them.
        """

        chars = set()
        for text in texts:
            chars.update(text)

        ARROW = "←"
        if ARROW not in chars:
            self.marker = ARROW
        else:
            self.marker = self.select_unique_marker(chars)

    def select_unique_marker(self, chars):
        for codepoint in range(0x10FFFF, 0x10000, -1):
            marker = chr(codepoint)
            if marker not in chars:
                return marker

        raise ValueError("Could not find a unique marker")

    def make_relative(self, text):
        """
        Transform text to use relative indents.
        """

        if self.marker in text:
            raise ValueError("Text already contains the outdent marker: {self.marker}")

        lines = text.splitlines(keepends=True)

        output = []
        prev_indent = ""
        for line in lines:
            line_without_end = line.rstrip("\n\r")

            len_indent = len(line_without_end) - len(line_without_end.lstrip())
            indent = line[:len_indent]
            change = len_indent - len(prev_indent)
            if change > 0:
                cur_indent = indent[-change:]
            elif change < 0:
                cur_indent = self.marker * -change
            else:
                cur_indent = ""

            out_line = cur_indent + "\n" + line[len_indent:]
            # dump(len_indent, change, out_line)
            # print(out_line)
            output.append(out_line)
            prev_indent = indent

        res = "".join(output)
        return res

    def make_absolute(self, text):
        """
        Transform text from relative back to absolute indents.
        """
        lines = text.splitlines(keepends=True)

        output = []
        prev_indent = ""
        for i in range(0, len(lines), 2):
            dent = lines[i].rstrip("\r\n")
            non_indent = lines[i + 1]

            if dent.startswith(self.marker):
                len_outdent = len(dent)
                cur_indent = prev_indent[:-len_outdent]
            else:
                cur_indent = prev_indent + dent

            if not non_indent.rstrip("\r\n"):
                out_line = non_indent  # don't indent a blank line
            else:
                out_line = cur_indent + non_indent

            output.append(out_line)
            prev_indent = cur_indent

        res = "".join(output)
        if self.marker in res:
            # dump(res)
            raise ValueError("Error transforming text back to absolute indents")

        return res


# The patches are created to change S->R.
# So all the patch offsets are relative to S.
# But O has a lot more content. So all the offsets are very wrong.
#
# But patch_apply() seems to imply that once patch N is located,
# then it adjusts the offset of the next patch.
#
# This is great, because once we sync up after a big gap the nearby
# patches are close to being located right.
# Except when indentation has been changed by GPT.
#
# It would help to use the diff trick to build map_S_offset_to_O_offset().
# Then update all the S offsets in the S->R patches to be O offsets.
# Do we also need to update the R offsets?
#
# What if this gets funky/wrong?
#


example = """Left
Left
    4 in
    4 in
        8 in
    4 in
Left
"""


def relative_indent(texts):
    ri = RelativeIndenter(texts)
    texts = list(map(ri.make_relative, texts))

    return ri, texts


line_padding = 100


def line_pad(text):
    padding = "\n" * line_padding
    return padding + text + padding


def line_unpad(text):
    if set(text[:line_padding] + text[-line_padding:]) != set("\n"):
        return
    return text[line_padding:-line_padding]


def lines_to_chars(lines, mapping):
    new_text = []
    for char in lines:
        new_text.append(mapping[ord(char)])

    new_text = "".join(new_text)
    return new_text

def diff_lines(search_text, replace_text):
    dmp = diff_match_patch()
    dmp.Diff_Timeout = 5
    # dmp.Diff_EditCost = 16
    search_lines, replace_lines, mapping = dmp.diff_linesToChars(search_text, replace_text)

    diff_lines = dmp.diff_main(search_lines, replace_lines, None)
    dmp.diff_cleanupSemantic(diff_lines)
    dmp.diff_cleanupEfficiency(diff_lines)

    diff = list(diff_lines)
    dmp.diff_charsToLines(diff, mapping)
    # dump(diff)

    udiff = []
    for d, lines in diff:
        if d < 0:
            d = "-"
        elif d > 0:
            d = "+"
        else:
            d = " "
        for line in lines.splitlines(keepends=True):
            udiff.append(d + line)

    return udiff


def search_and_replace(texts):
    search_text, replace_text, original_text = texts

    num = original_text.count(search_text)
    # if num > 1:
    #    raise SearchTextNotUnique()
    if num == 0:
        return

    new_text = original_text.replace(search_text, replace_text)

    return new_text


class SearchTextNotUnique(ValueError):
    pass


all_preprocs = [
    # (strip_blank_lines, relative_indent, reverse_lines)
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    # (False, False, True),
    # (True, False, True),
    # (False, True, True),
    # (True, True, True),
]

always_relative_indent = [
    (False, True, False),
    (True, True, False),
    # (False, True, True),
    # (True, True, True),
]

never_relative = [
    (False, False),
    (True, False),
]



def flexible_search_and_replace(texts, strategies):
    """Try a series of search/replace methods, starting from the most
    literal interpretation of search_text. If needed, progress to more
    flexible methods, which can accommodate divergence between
    search_text and original_text and yet still achieve the desired
    edits.
    """

    for strategy, preprocs in strategies:
        for preproc in preprocs:
            res = try_strategy(texts, strategy, preproc)
            if res:
                return res


def reverse_lines(text):
    lines = text.splitlines(keepends=True)
    lines.reverse()
    return "".join(lines)


def try_strategy(texts, strategy, preproc):
    preproc_strip_blank_lines, preproc_relative_indent, preproc_reverse = preproc
    ri = None

    if preproc_strip_blank_lines:
        texts = strip_blank_lines(texts)
    if preproc_relative_indent:
        ri, texts = relative_indent(texts)
    if preproc_reverse:
        texts = list(map(reverse_lines, texts))

    res = strategy(texts)

    if res and preproc_reverse:
        res = reverse_lines(res)

    if res and preproc_relative_indent:
        try:
            res = ri.make_absolute(res)
        except ValueError:
            return

    return res


def strip_blank_lines(texts):
    # strip leading and trailing blank lines
    texts = [text.strip("\n") + "\n" for text in texts]
    return texts