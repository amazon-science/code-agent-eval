import json
import logging
import pickle
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import express as px

from scipy.stats import gaussian_kde

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from llm_agent_evaluation.data.patch_utils import Patch
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)

random.seed(42)

BENCHMARK = 'swe-bench_Lite'

PATH_TO_RESOURCES = '../../resources'

AGENT_NAME = '20240617_factory_code_droid'

MODEL_NAME = 'opus-3'


def extract_predictions_and_ground_truth(
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

    if flatten:
        return np.array(all_true), np.array(all_pred), np.array(all_conf)
    else:
        return all_true, all_pred, all_conf


def plot_test_confidence_maps(
    lengths,
    confidence_scores,
    outcomes,
    output_path,
):
    count_correct, count_incorrect, count_total = 0, 0, 0
    for x, y, z in zip(lengths, confidence_scores, outcomes):
        if x >=100:
            count_total += 1
            if z == 1:
                count_correct += 1
            elif z == 0:
                count_incorrect += 1
    print(f'Number of instances with test lengths >=100: {count_total}. Of these, correct {count_correct}, incorrect {count_incorrect}')

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.set_title('Verbalized Confidence Measures v/s Test Lengths')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Test Length')

    # Create two separate scatter plots for each outcome
    scatter1 = ax1.scatter(
        [c for c, o in zip(confidence_scores, outcomes) if o == 0],
        [l for l, o in zip(lengths, outcomes) if o == 0],
        marker='x', color='red', label='Incorrect Prediction'
    )
    scatter2 = ax1.scatter(
        [c for c, o in zip(confidence_scores, outcomes) if o == 1],
        [l for l, o in zip(lengths, outcomes) if o == 1],
        marker='o', color='darkgreen', label='Correct Prediction'
    )

    # Add a legend
    ax1.legend(handles=[scatter1, scatter2])

    # Violin plots of length and confidence, split by outcome
    sns.violinplot(x=outcomes, y=lengths, ax=ax2, split=False)
#    sns.violinplot(x=outcomes, y=confidence_scores, ax=ax2, split=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Incorrect', 'Correct'])
    ax2.set_title('Test Length v/s Outcome')
    ax2.set_xlabel('Outcome')
    ax2.set_ylabel('Test Length')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.savefig(output_path)

    # Show the plot
    plt.show()


def plot_test_confidence_maps_v2(
    lengths,
    confidence_scores,
    outcomes,
    output_path,
):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Density plot for correct predictions
    sns.kdeplot(
        x=[c for c, o in zip(confidence_scores, outcomes) if o == 1],
        y=[l for l, o in zip(lengths, outcomes) if o == 1],
        ax=ax1, fill=True, cmap="Greens", thresh=True,
        label="Correct Prediction"
    )
    ax1.set_title('Density Plot for Correct Predictions')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Test Length')

    # Density plot for incorrect predictions
    sns.kdeplot(
        x=[c for c, o in zip(confidence_scores, outcomes) if o == 0],
        y=[l for l, o in zip(lengths, outcomes) if o == 0],
        ax=ax2, fill=True, cmap="Reds", thresh=True,
        label="Incorrect Prediction"
    )
    ax2.set_title('Density Plot for Incorrect Predictions')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Test Length')

    # Add legends
    ax1.legend()
    ax2.legend()

    plt.savefig(output_path)

    # Show the plot
    plt.show()


def plot_confidence(
    true_classes,
    predictions,
    confidence_scores,
    test_lengths,
    output_path,
):
    correct_mask = predictions == true_classes
    incorrect_mask = ~correct_mask

    # Set up the plot
    fig = plt.figure(figsize=(12, 6))

    # Plot distribution for correct predictions
    sns.kdeplot(
        confidence_scores[correct_mask],
        fill=True,
        color='green',
        label="Correct Predictions",
        alpha=0.5, clip=(0, 100),
        linestyle=':',
    )

    # Plot distribution for incorrect predictions
    sns.kdeplot(
        confidence_scores[incorrect_mask],
        fill=True,
        color='red',
        label="Incorrect Predictions",
        alpha=0.5,
        clip=(0, 100),
        linestyle='--',
    )

    # Customize the plot
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.title("Distribution of Verbalized Confidence Measures")
    plt.legend()

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.show()

def plot_confidence_predictions(
    predictions,
    confidence_scores,
    output_path
):
    # Create a histogram with confidence scores on the x-axis
    bins = np.linspace(0, 100, 11)  # 11 bins from 0 to 1
    # correct_counts, _, _ = plt.hist(
    #     [score for score, pred in zip(confidence_scores, predictions) if pred == 1],
    #     bins=bins,
    #     alpha=0.5,
    #     label='Correct',
    #     color='green'
    # )
    incorrect_counts, _, _ = plt.hist(
        [score for score, pred in zip(confidence_scores, predictions) if pred == 0],
        bins=bins,
        alpha=0.5,
        label='Incorrect',
        color='red'
    )

    # Stack the bars
    # plt.bar(bins[:-1], correct_counts, width=0.09, alpha=0.5, color='green', label='Correct')
    # plt.bar(bins[:-1], incorrect_counts, width=0.09, alpha=0.5, color='red', label='Incorrect', bottom=correct_counts)

    # Set labels and title
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Histogram of Confidence Scores')
    plt.legend()

    plt.savefig(output_path)
    # Show the plot
    plt.show()

def apply_threshold(confidence_scores, predictions, lengths, threshold):
    new_predictions = []
    for conf, pred, length in zip(confidence_scores, predictions, lengths):
        if conf >= threshold:
            new_predictions.append(pred)
        else:
            if length >= 50:
                new_predictions.append(0)
            else:
                new_predictions.append(pred)
    return new_predictions    


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn * 100 / (tn + fp)
    print((f"\tAccuracy: {accuracy:.2f}%\tPrecision: {precision:.2f}%\t"
           f"\tRecall: {recall:.2f}%\tF1 Score: {f1:.2f}%\tTrue Negative Rate: {tnr:.2f}%"))


def plot_confidence_and_evaluate(dimension, key):
    filename = f'{BENCHMARK}.{AGENT_NAME}.{dimension}.{key}'
    with open(str(path_to_results / f'{filename}.json'), 'r') as f:
        results = json.load(f)

    all_true, all_pred, confidence_scores = extract_predictions_and_ground_truth(
        results, flatten=True
    )

    evaluate_model(all_true, all_pred)

    test_lengths = None
    if key.startswith('test-centric'):
        test_lengths = [
            len(item['prompt_inputs']['test'].split('\n')) \
            for result in results.values() for item in result
        ]
        print(f'Total pass: {sum(all_true)}\tTotal fail: {len(all_true) - sum(all_true)}')
        # Evaluate model after applying confidence threshold
        all_pred = apply_threshold(confidence_scores, all_pred, test_lengths, 65)
        print('***   Applying threshold   ***')
        evaluate_model(all_true, all_pred)
        print()

        outcomes = [1.0 if pred == all_true[idx] else 0.0 for idx, pred in enumerate(all_pred)]
        plot_test_confidence_maps(
            test_lengths,
            confidence_scores,
            outcomes,
            path_to_figures / f'{filename}.correlations.png',
        )

        new_all_pred_list, all_true_list = [], []
        for result in results.values():
            _pred_list, _true_list = [], []
            for item in result:
                length = len(item['prompt_inputs']['test'].split('\n'))
                conf = item['confidence']
                pred = 1 if item['pred'] == 'yes' else 0
                true = 1 if item['true'] == 'pass' else 0
                if length >= 50 and conf <= 65:
                    _pred_list.append(0)
                else:
                    _pred_list.append(pred)
                _true_list.append(true)
            all_true_list.append(_true_list)
            new_all_pred_list.append(_pred_list)

        logger.info('...    Macro-evaluation    ...')
        true = [0 if 0 in true_list else 1 for true_list in all_true_list]
        pred = [0 if 0 in pred_list else 1 for pred_list in new_all_pred_list]
        evaluate_model(true, pred)


    # Evaluate model with original predictions
    plot_confidence(
        all_true, all_pred, confidence_scores, test_lengths, path_to_figures / f'{filename}.png'
    )

    plot_confidence_predictions(
        all_pred, confidence_scores, path_to_figures / f'{filename}.preds.png'
    )

    # evaluate_model(all_true, all_pred)
    # if key.startswith('test-centric'):
    #     # Evaluate model after applying confidence threshold
    #     all_pred = apply_threshold(confidence_scores, all_pred, 65)
    #     print('***   Applying threshold   ***')
    #     evaluate_model(all_true, all_pred)
    #     print()



if __name__ == '__main__':
    path_to_results = Path(PATH_TO_RESOURCES) / f'results-{MODEL_NAME}'
    path_to_figures = Path.cwd() / 'figures' / MODEL_NAME
    path_to_figures.mkdir(exist_ok=True, parents=True)

    for evaluation_type in [
        'random',
        'edit',
        'ref-free',
        'patch-equivalence',
        'holistic',
        'holistic-patch',
        'test-centric',
        'test-centric-patch',
    ]:
        ## Random baseline:
        if evaluation_type == 'random':
            logger.info(f'*** Results for random baseline ***')
            filename = f'{BENCHMARK}.{AGENT_NAME}.execution.test-centric-patch'
            with open(str(path_to_results / f'{filename}.json'), 'r') as f:
                results = json.load(f)

            all_true, _, _ = extract_predictions_and_ground_truth(
                results, flatten=False
            )
            flattened_true = [item for sublist in all_true for item in sublist]

            weight_1 = sum(flattened_true) / len(flattened_true)
            weight_0 = 1 - weight_1

            all_pred = [
                random.choices([0, 1], weights=[weight_0, weight_1], k=len(item)) \
                for item in all_true
            ]
            flattened_pred = [item for sublist in all_pred for item in sublist]

            logger.info('...    Micro-evaluation    ...')
            evaluate_model(flattened_true, flattened_pred)

            true = [0 if 0 in _true else 1 for _true in all_true]
            pred = [0 if 0 in _pred else 1 for _pred in all_pred]
            logger.info('...    Macro-evaluation    ...')
            evaluate_model(true, pred)

        ## Edit distance:
        elif evaluation_type == 'edit':
            filename = f'{BENCHMARK}.{AGENT_NAME}.lexical'
            with open(str(path_to_results / f'{filename}.json'), 'r') as f:
                results = json.load(f)

            scores = [
                item_results['edit_embedding'] for item_results in results.values()
            ]
            true = [
                1 if item_results['build_status'] == 'pass' else 0 \
                for item_results in results.values()
            ]

            threshold, best_threshold = 0.001, None
            best_f1 = 0
            while threshold < 1.0:
                _pred = [1 if score >= threshold else 0 for score in scores]
                acc = accuracy_score(true, _pred) * 100
                f1 = f1_score(true, _pred) * 100
                if f1 > best_f1:
                    best_threshold = threshold
                    best_f1 = f1
                threshold += 0.001

            logger.info(
                f'*** Results for edit distance, with default patches (threshold {best_threshold}) ***'
            )
            pred = [1 if score >= best_threshold else 0 for score in scores]
            evaluate_model(true, pred)

        ## Reference-Free Evaluation
        elif evaluation_type == 'ref-free':
            for key in ['none.patch', 'U10.patch', 'function.patch']:
                logger.info(
                    f'*** Results for reference-free evaluation, with patches and no hints, {key}***'
                )
                plot_confidence_and_evaluate('ref-free', key)

            for key in ['none.patch.hints', 'U10.patch.hints', 'function.patch.hints']:
                logger.info(
                    f'*** Results for reference-free evaluation, with patches and hints, {key}***'
                )
                plot_confidence_and_evaluate('ref-free', key)

        ## Semantics-Specific Evaluation
        elif evaluation_type == 'patch-equivalence':
            for key in ['none.patch', 'U10.patch', 'function.patch']:
                logger.info(
                    f'*** Results for semantics-specific evaluation with patches, {key} ***'
                )
                plot_confidence_and_evaluate('semantics', key)

        ## Execution-Specific Evaluation
        elif evaluation_type == 'holistic':
            logger.info(
                f'*** Results for holistic, execution-specific evaluation ***'
            )
            plot_confidence_and_evaluate('execution', 'none')

        elif evaluation_type == 'holistic-patch':
            logger.info(
                f'*** Results for holistic, execution-specific evaluation with patches ***'
            )
            plot_confidence_and_evaluate('execution', 'none-patch')

        elif evaluation_type == 'test-centric':
            logger.info(
                f'*** Results for test-centric, execution-specific evaluation ***'
            )
            plot_confidence_and_evaluate('execution', 'test-centric')

        elif evaluation_type == 'test-centric-patch':
            logger.info(
                f'*** Results for test-centric, execution-specific evaluation with patches ***'
            )
            plot_confidence_and_evaluate('execution', 'test-centric-patch')
