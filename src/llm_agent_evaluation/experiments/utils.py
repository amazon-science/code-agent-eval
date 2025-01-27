from typing import Dict

from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type,
)


SONNET_3 = 'us.anthropic.claude-3-sonnet-20240229-v1:0'
SONNET_3_5 = 'us.anthropic.claude-3-5-sonnet-20240620-v1:0'
OPUS_3 = 'us.anthropic.claude-3-opus-20240229-v1:0'
ANTHROPIC_SONNET_3_5 = 'claude-3-5-sonnet-20240620'
ANTHROPIC_OPUS_3 = 'claude-3-opus-20240229'



# Only those agents are recorded that were tested for both SWE-Bench,
# and the SWE-Bench_Lite versions.
ALLOWED_AGENTS = [
    '20231010_rag_claude2',
    '20231010_rag_gpt35',
    '20231010_rag_swellama13b',
    '20231010_rag_swellama7b',
    '20240402_rag_claude3opus',
    '20240402_rag_gpt4',
    '20240402_sweagent_claude3opus',
    '20240402_sweagent_gpt4',
    '20240509_amazon-q-developer-agent-20240430-dev',
    '20240615_appmap-navie_gpt4o',
    '20240617_factory_code_droid',
    '20240620_sweagent_claude3.5sonnet',
    '20240820_honeycomb',
    '20240627_abanteai_mentatbot_gpt4o',
    '20240811_gru'
]


TEST_STATUS_TO_LABEL = {
    'PASSED': 'pass',
    'FAILED': 'fail',
    'SKIPPED': 'skip',
    'ERROR': 'error',
}


def levenshtein_distance(s1, s2):
    """Computes Levenshtein distance between two code strings.

    Reference:
    https://stackoverflow.com/questions/2460177/edit-distance-in-python

    Args:
        s1: First code string
        s2: Second code string

    Returns:
        The Levenshtein distance between the two code strings.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def extract_function_name(source: str) -> str:
    """Extract name of a (test) function, given its source.

    Args:
        source: Code string.

    Returns:
        Function name.
    """
    ast_tree = build_ast_from_source(source)
    function_nodes = list(find_nodes_of_type(ast_tree, 'function_definition'))
    if not function_nodes:
        return None

    name_node = min(function_nodes, key=lambda node: node.start_point[0])
    for child in name_node.children:
        if child.type == 'identifier':
            name = child.text.decode('utf-8')
            break
    return name


def get_test_label_swebench(
    test_function_name: str,
    instance_id: str,
    tests_status_mapper: Dict[str, Dict[str, str]],
) -> str:
    """Computes the unit test status for a specific test for a patch
    instance in SWE-Bench.

    Args:
        test_function_name: Unit test name.
        instance_id: Patch instance identifier.
        tests_status_mapper: Maps unit tests to pass/fail status.

    Returns:
        Test status label.
    """
    project_name = '-'.join(instance_id.split('-')[:-1]).replace('__', '/')
    # If there is no generation for a specific instance, its tests will
    # not be present in the logs. Skipping these cases.
    if instance_id not in tests_status_mapper.get(project_name, []):
        return None

    # Sometimes, a test appears multiple times in tests_status_mapper, possibly
    # due to issues with regex parsing. Here, we map to the closest candidate
    # based on edit distance.
    candidates = [
        (name, label)
        for name, label in tests_status_mapper[project_name][instance_id].items()
        if test_function_name in name
    ]

    # Skip, if test status can not be retrieved from logs.
    if not candidates:
        return None

    test_status = min(
        candidates,
        key=lambda candidate: levenshtein_distance(test_function_name, candidate)
    )[1]

    return TEST_STATUS_TO_LABEL[test_status]


def get_test_label_bugsinpy(test_function_name: str) -> str:
    raise NotImplementedError
