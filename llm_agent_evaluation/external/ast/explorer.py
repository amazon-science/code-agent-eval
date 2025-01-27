import pathlib
from typing import Iterator

import tree_sitter_python

from tree_sitter import Language, Parser, Node, Tree


PY_LANGUAGE = Language(tree_sitter_python.language())


def build_ast_from_file(path_to_file: pathlib.Path) -> Tree:
    """ Build AST for a .py file. 

    Note: This can easily be extended to files with other extensions later.

    Args:
        path_to_file: Path to source code file.

    Returns:
        Abstract syntax tree.
    """
    with open(path_to_file, 'r', encoding='utf-8') as f:
        code = f.read()
    return build_ast_from_source(code)


def build_ast_from_source(code: str) -> Tree:
    """Build AST for a code snippet.

    Args:
        code: Source code.

    Returns:
        Abstract syntax tree.
    """
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf-8'))
    return tree


def traverse_tree(tree: Tree) -> Iterator[Node]:
    """Traverse the tree and print its contents.

    Reference:
    https://github.com/tree-sitter/py-tree-sitter/blob/master/examples/walk_tree.py

    Args:
        tree: Abstract syntax tree.

    Yield:
        Next node in the abstract syntax tree (pre-order traversal).
    """
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def find_nodes_of_type(tree: Tree, node_type: str) -> Iterator[Node]:
    """Filter nodes in the tree by type.

    Args:
        tree: Abstract syntax tree.
        node_type: Type of node to filter.

    Yield:
        Next node in the abstract syntax tree (pre-order traversal).
    """
    for node in traverse_tree(tree):
        if node.type == node_type:
            yield node
