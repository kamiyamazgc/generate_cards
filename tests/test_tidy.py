import builtins
from generate_cards import tidy_markdown_para


def test_sentence_end_newlines():
    text = "一行目です。\n二行目です。"
    expected = "一行目です。\n\n二行目です。"
    assert tidy_markdown_para(text) == expected


def test_collapse_excess_blank_lines():
    text = "start\n\n\n\nmiddle\n\n\n\nend"
    expected = "start\n\nmiddle\n\nend"
    assert tidy_markdown_para(text) == expected
