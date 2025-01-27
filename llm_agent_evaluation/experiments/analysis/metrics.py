import json
import jedi
import logging
import pickle
import py_compile
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib.pyplot as plt

from plotly import express as px

from scipy.stats import gaussian_kde

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from llm_agent_evaluation.data.patch_utils import Patch
from llm_agent_evaluation.external.ast.explorer import (
    build_ast_from_source,
    find_nodes_of_type
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


BENCHMARK = 'swe-bench_Lite'

CONTEXT_KEY = 'function'

PATH_TO_RESOURCES = '../../resources'

AGENT_NAME = '20240617_factory_code_droid'

MODEL_NAME = 'opus-3'

random.seed(42)


def check_compilation_status(patch: Patch, agent_name: str) -> bool:
    """Check whether all files modified by a patch, post-fix, are compilable.

    Args:
        patch: Input patch.

    Return:
        True, if all modified files are compilable, else False.
    """
    path_to_root = Path(patch.path_to_root)

    for chunk in patch.change_patch.after_chunks:
        filename = chunk.filename
        path = path_to_root / 'snapshots' / agent_name / filename
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError:
            return False

    return True


def plot_equivalence(
    true: List[int],
    pred: List[int],
    x_label: str,
    y_label: str,
    tick_labels: str,
    path_to_save: Path,
) -> None:
    fig = px.scatter(
        x=true,
        y=pred,
        text=tick_labels,
        labels={"x": x_label, "y": y_label},
    ).update_traces(mode="markers", marker={"size": 10})
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=30,
        y1=30,
        line=dict(color="Gray", width=2, dash="dash"),
    )
    fig.write_image(str(path_to_save))


def retrieve_results(
    benchmark: str,
    agent_name: str,
    dimension: str,
    key: str,
    path_to_results: Path,
) -> Dict:
    """
    """
    filename = f'{benchmark}.{agent_name}.{dimension}.{key}.json'
    with open(str(path_to_results / filename), 'r') as f:
        results = json.load(f)

    return results


def extract_predictions_ground_truth_and_confidence(
    results: Dict[str, Dict],
    flatten: bool,
) -> Tuple[List[int], List[int]]:
    """Extract predictions and ground truth from results.

    Args:
        results: Input results.
        flatten: Whether to flatten the results or not.

    Returns:
        A tuple containing two lists:
            - The first list represents the ground truth values (0 or 1).
            - The second list represents the predicted values (0 or 1).
            If `flatten` is True, the lists will contain flattened values
            for all instances. Otherwise, the lists will contain separate
            values for each instance.
    """
    all_true, all_pred, all_conf = [], [], []
    for instance_id, item in results.items():
        item_true, item_pred, item_conf = [], [], []
        for _results in item:
            pred = 1 if _results['pred'] == 'yes' else 0
            true = 1 if _results['true'] == 'pass' else 0

            item_true.append(true)
            item_pred.append(pred)
            item_conf.append(int(_results['confidence']))

        if flatten:
            all_true += item_true
            all_pred += item_pred
            all_conf += item_conf
        else:
            all_true.append(item_true)
            all_pred.append(item_pred)
            all_conf.append(item_conf)

    return all_true, all_pred, all_conf

def extract_predictions_and_ground_truth(
    results: Dict[str, Dict],
    patch_mapper: Dict[str, Patch],
    flatten: bool,
) -> Tuple[List[int], List[int]]:
    """Extract predictions and ground truth from results.

    Args:
        results: Input results.
        patch_mapper: Patch mapper.
        flatten: Whether to flatten the results or not.

    Returns:
        A tuple containing two lists:
            - The first list represents the ground truth values (0 or 1).
            - The second list represents the predicted values (0 or 1).
            If `flatten` is True, the lists will contain flattened values
            for all instances. Otherwise, the lists will contain separate
            values for each instance.
    """
    all_true, all_pred = [], []
    for instance_id, item in results.items():
        patch = patch_mapper[instance_id]

        item_true, item_pred = [], []
        for _results in item:
            pred = 1 if _results['pred'] == 'yes' else 0
            true = 1 if _results['true'] == 'pass' else 0

            item_true.append(true)
            item_pred.append(pred)

        if flatten:
            all_true += item_true
            all_pred += item_pred
        else:
            all_true.append(item_true)
            all_pred.append(item_pred)

    return all_true, all_pred


def plot_test_pass_rate_density(
    true_test_pass_rates: List[float],
    pred_test_pass_rate: List[float],
    benchmark: str,
    output_path: Path,
) -> None:
    """
    """
    data = zip(true_test_pass_rates, pred_test_pass_rate)

    # Unpack the tuples into separate lists
    true_test_pass_rates, regressions = zip(*data)

    # Correcting the density contour plot
    plt.figure(figsize=(14, 6))

    # Density Contour plot
    ax = plt.subplot(1, 2, 2)
    xy = np.vstack([true_test_pass_rates, pred_test_pass_rate])
    density = gaussian_kde(xy)(xy)

    # Sort the points by density for a better visual representation
    idx = density.argsort()
    x, y, z = np.array(true_test_pass_rates)[idx], np.array(pred_test_pass_rate)[idx], density[idx]
    plt.scatter(x, y, c=z, s=50, cmap='viridis')
    plt.colorbar(label='Density')

    # Generating a grid to evaluate model over the space
    xmin, xmax = -5, 105
    ymin, ymax = -5, 105
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(gaussian_kde(xy)(positions).T, X.shape)

    # Contour plot
    plt.contourf(X, Y, Z, levels=10, cmap='viridis', alpha=0.5)

    # Add a dashed X=Y line
    ax.plot([xmin, xmax], [ymin, ymax], 'r--', alpha=0.75, zorder=0)
    plt.title(f'LLM-Predicted vs True Test-Pass Rate for instances in {benchmark.upper()}')
    plt.xlabel('True Test Pass Rate (%)')
    plt.ylabel('LLM-Predicted Test Pass Rate (%)')
    plt.tight_layout()
    plt.savefig(str(output_path))


def plot_test_pass_rate_histogram(
    true_test_pass_rates: List[float],
    pred_test_pass_rates: List[float],
    number_bins: int,
    benchmark: str,
    output_path: Path,
):
    """
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the histograms
    ax.hist(
        [pred_test_pass_rates, true_test_pass_rates],
        bins=number_bins,
        label=['LLM-Predicted', 'True'],
        color=['red', 'green'],
        alpha=0.5,
    )

    # Add labels and title
    ax.set_xlabel('Percentage of Passed Tests in Test Patch (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'LLM-Predicted v/s True Percentage of Passed Passed per Instance in {benchmark.upper()}')
    ax.legend()

    plt.savefig(str(output_path))

    # Show the plot
    plt.show()


def compute_holistic_evaluation_metrics(all_true, all_pred):
    tp = sum([1 for true, pred in zip(all_true, all_pred) if true == pred == 1])
    tn = sum([1 for true, pred in zip(all_true, all_pred) if true == pred == 0])

    logger.info('Predicted resolved rates:')
    logger.info(
        f"  Accuracy: {accuracy_score(all_true, all_pred)*100:.3f}%\t"
        f"  Precision: {precision_score(all_true, all_pred)*100:.3f}%\t"
        f"  Recall: {recall_score(all_true, all_pred)*100:.3f}%\t"
        f"  F1-score: {f1_score(all_true, all_pred)*100:.3f}%\t"
    )

    logger.info((
        f'  Correctly predicted build failures: {tn}/{len(all_true) - sum(all_true)}\t'
        f'  Correctly predicted passing builds: {tp}/{sum(all_true)}'
    ))
    logger.info('### End-of-Evaluation ###\n\n')


def compute_test_centric_evaluation_metrics(
    all_true,
    all_pred,
    all_confidence,
    test_lengths,
    total_tests_per_instance,
    key,
):
    flattened_true = [item for sublist in all_true for item in sublist]
    flattened_pred = [item for sublist in all_pred for item in sublist]
    test_tp = sum([
        1 for true, pred in zip(flattened_true, flattened_pred) 
        if true == pred == 1
    ])
    test_tn = sum([
        1 for true, pred in zip(flattened_true, flattened_pred) 
        if true == pred == 0
    ])
    logger.info('  Predicted Test Status:')
    logger.info(
        f"  Accuracy: {accuracy_score(flattened_true, flattened_pred)*100:.3f}%\t"
        f"  Precision: {precision_score(flattened_true, flattened_pred)*100:.3f}%\t"
        f"  Recall: {recall_score(flattened_true, flattened_pred)*100:.3f}%\t"
        f"  F1-score: {f1_score(flattened_true, flattened_pred)*100:.3f}%\t"
    )

    logger.info((
        f'  Correctly predicted test failures: {test_tn}/{len(flattened_true) - sum(flattened_true)}\t'
        f'  Correctly predicted passing tests: {test_tp}/{sum(flattened_true)}'
    ))

    resolved_true = [0 if 0 in item else 1 for item in all_true]
    resolved_pred = [0 if 0 in item else 1 for item in all_pred]
    resolved_tp = sum([
        1 for true, pred in zip(resolved_true, resolved_pred) 
        if true == pred == 1
    ])
    resolved_tn = sum([
        1 for true, pred in zip(resolved_true, resolved_pred) 
        if true == pred == 0
    ])

    logger.info('  Predicted resolved rates:')
    logger.info(
        (f"  Accuracy: {accuracy_score(resolved_true, resolved_pred)*100:.3f}%\t"
         f"  Precision: {precision_score(resolved_true, resolved_pred)*100:.3f}%\t"
         f"  Recall: {recall_score(resolved_true, resolved_pred)*100:.3f}%\t"
         f"  F1-Score: {f1_score(resolved_true, resolved_pred)*100:.3f}%\t"
        )
    )
    logger.info((
        f'  Correctly predicted build failures: {resolved_tn}/{len(resolved_true) - sum(resolved_true)}\t'
        f'  Correctly predicted passing builds: {resolved_tp}/{sum(resolved_true)}'
    ))
    logger.info(f'### End-of-Evaluation ###\n\n')

    true_test_pass_rates = [
        sum(sublist)*100 / total_tests_per_instance[idx] for idx, sublist in enumerate(all_true)
    ]
    pred_test_pass_rates = [
        sum(sublist)*100 / total_tests_per_instance[idx] for idx, sublist in enumerate(all_pred)
    ]

    # Plot test pass rate density.
    path_to_plots = Path.cwd() / 'figures'
    path_to_plots.mkdir(parents=True, exist_ok=True)

    plot_test_pass_rate_density(
        true_test_pass_rates,
        pred_test_pass_rates,
        BENCHMARK,
        path_to_plots / f'{BENCHMARK}.{AGENT_NAME}.density.{key}.png',
    )

    plot_test_pass_rate_histogram(
        true_test_pass_rates,
        pred_test_pass_rates,
        20,
        BENCHMARK,
        path_to_plots / f'{BENCHMARK}.{AGENT_NAME}.hist.{key}.png',
    )


def compute_patch_level_metrics(all_true, all_pred):
    tp = sum([1 for true, pred in zip(all_true, all_pred) if true == pred == 1])
    tn = sum([1 for true, pred in zip(all_true, all_pred) if true == pred == 0])

    logger.info('Predicted resolved rates:')
    logger.info(
        f"  Accuracy: {accuracy_score(all_true, all_pred)*100:.3f}%\t"
        f"  Precision: {precision_score(all_true, all_pred)*100:.3f}%\t"
        f"  Recall: {recall_score(all_true, all_pred)*100:.3f}%\t"
        f"  F1-score: {f1_score(all_true, all_pred)*100:.3f}%\t"
    )

    logger.info((
        f'  Correctly predicted build failures: {tn}/{len(all_true) - sum(all_true)}\t'
        f'  Correctly predicted passing builds: {tp}/{sum(all_true)}'
    ))
    logger.info(f'### End-of-Evaluation ###\n\n')


if __name__ == '__main__':
    path_to_results = Path(PATH_TO_RESOURCES) / f'results-{MODEL_NAME}'

    patches_filename = f'{BENCHMARK}.{AGENT_NAME}.patches.{CONTEXT_KEY}-context.pkl'
    with open(str(Path(PATH_TO_RESOURCES) / 'cache' / patches_filename), 'rb') as f:
        all_patches = pickle.load(f)

    patch_mapper = {patch.id: patch for patch in all_patches}

    resolved_status_filename = f'{BENCHMARK}.{AGENT_NAME}.resolved-status.json'
    with open(str(Path(PATH_TO_RESOURCES) / 'cache' / resolved_status_filename), 'r') as f:
        resolved_status_mapper = json.load(f)

    tests_status_filename = f'{BENCHMARK}.{AGENT_NAME}.test-status.json'
    with open(str(Path(PATH_TO_RESOURCES) / 'cache' / tests_status_filename), 'r') as f:
        tests_status_mapper = json.load(f)

    ## Random baseline.
    logger.info(
        f'*** Results for random baseline ***'
    )
    all_true = [1 if patch.id in resolved_status_mapper['resolved'] else 0 for patch in all_patches]
    all_pred = [random.choice([1, 0]) for _ in range(len(all_true))]

    logger.info('Predicted resolved rates:')
    logger.info(
        f"  Accuracy: {accuracy_score(all_true, all_pred)*100:.3f}%\t"
        f"  Precision: {precision_score(all_true, all_pred)*100:.3f}%\t"
        f"  Recall: {recall_score(all_true, all_pred)*100:.3f}%\t"
        f"  F1-score: {f1_score(all_true, all_pred)*100:.3f}%\t"
    )
    logger.info(f'### End-of-Evaluation ###\n\n')

    ## Holistic evaluation.
    logger.info(
        f'*** Results for holistic, execution-specific evaluation ***'
    )

    holistic_results = retrieve_results(
        BENCHMARK, AGENT_NAME, 'execution', 'none', path_to_results
    )
    all_true, all_pred = extract_predictions_and_ground_truth(
        holistic_results, patch_mapper, flatten=True
    )
    compute_holistic_evaluation_metrics(all_true, all_pred)

    ## Holistic evaluation, w/ patches.
    logger.info(
        f'*** Results for holistic, execution-specific evaluation with patches ***'
    )

    holistic_results = retrieve_results(
        BENCHMARK, AGENT_NAME, 'execution', 'none-patch', path_to_results
    )
    all_true, all_pred = extract_predictions_and_ground_truth(
        holistic_results, patch_mapper, flatten=True
    )
    compute_holistic_evaluation_metrics(all_true, all_pred)

    ## Test-centric evaluation.
    print()
    logger.info(
        f'*** Results for test-centric, execution-specific evaluation ***'
    )

    test_centric_results = retrieve_results(
        BENCHMARK, AGENT_NAME, 'execution', 'test-centric', path_to_results
    )
    test_lengths = [
            len(item['prompt_inputs']['test'].split('\n')) \
            for result in test_centric_results.values() for item in result
        ]

    all_true, all_pred, all_conf = extract_predictions_ground_truth_and_confidence(
        test_centric_results, flatten=True
    )
    total_tests_per_instance = [
        len(patch_mapper[_id].test_patch.relevant_tests) for _id in list(test_centric_results.keys())
    ]
    compute_test_centric_evaluation_metrics(
        all_true,
        all_pred,
        all_conf,
        test_lengths,
        total_tests_per_instance,
        'test-centric'
    )

    # Test-centric evaluation with patches.
    print()
    logger.info(
        f'*** Results for test-centric, execution-specific evaluation with patches ***'
    )

    test_centric_results = retrieve_results(
        BENCHMARK, AGENT_NAME, 'execution', 'test-centric-patch', path_to_results
    )
    test_lengths = [
            len(item['prompt_inputs']['test'].split('\n')) \
            for result in test_centric_results.values() for item in result
        ]

    all_true, all_pred, all_conf = extract_predictions_ground_truth_and_confidence(
        test_centric_results, flatten=True
    )
    total_tests_per_instance = [
        len(patch_mapper[_id].test_patch.relevant_tests) for _id in list(test_centric_results.keys())
    ]
    compute_test_centric_evaluation_metrics(
        all_true,
        all_pred,
        all_confidence,
        test_lengths,
        total_tests_per_instance,
        'test-centric-patch'
    )

    # Patch equivalence
    for key, context in zip(
        ['+- 3 lines', '+-10 lines', 'function-level'],
        ['none', 'U10', 'function'],
    ):
        logger.info(
            f'*** Results for patch ({key} context), semantics-specific evaluation ***'
        )

        patch_results = retrieve_results(
            BENCHMARK, AGENT_NAME, f'semantics.{context}', 'patch', path_to_results
        )

        all_true, all_pred = extract_predictions_and_ground_truth(
            patch_results, patch_mapper, flatten=True
        )
        compute_patch_level_metrics(all_true, all_pred)


    ## Reference-Free Evaluation without hints
    for key, context in zip(
       ['+- 3 lines', '+-10 lines', 'function-level'],
       ['none', 'U10', 'function'],
    ):
        logger.info(
            f'*** Results for patch ({key} context), reference-free evaluation ***'
        )

        patch_results = retrieve_results(
            BENCHMARK, AGENT_NAME, f'ref-free.{context}', 'patch', path_to_results
        )

        all_true, all_pred = extract_predictions_and_ground_truth(
            patch_results, patch_mapper, flatten=True
        )
        compute_patch_level_metrics(all_true, all_pred)


    ## Reference-Free Evaluation with hints
    for key, context in zip(
       ['+- 3 lines', '+-10 lines', 'function-level'],
       ['none', 'U10', 'function'],
    ):
        logger.info(
            f'*** Results for patch ({key} context), reference-free evaluation with hints ***'
        )

        patch_results = retrieve_results(
            BENCHMARK, AGENT_NAME, f'ref-free.{context}', 'patch.hints', path_to_results
        )

        all_true, all_pred = extract_predictions_and_ground_truth(
            patch_results, patch_mapper, flatten=True
        )
        compute_patch_level_metrics(all_true, all_pred)
