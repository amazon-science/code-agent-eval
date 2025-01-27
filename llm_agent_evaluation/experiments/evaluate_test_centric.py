import argparse
import itertools
import json
import logging
import random
import subprocess
from functools import partial
from itertools import zip_longest
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as plt

from plotly import express as px

from scipy.stats import gaussian_kde, spearmanr

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils import *

import matplotlib.pyplot as plt

import numpy as np



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


AGENT_NAMES = [
    '20240702_codestory_aide_mixed',
    '20240617_factory_code_droid',
    '20240402_sweagent_gpt4',
    #'20240820_honeycomb',
    '20240811_gru'
]


def plot_build_status_comparison(results):
    data = dict(zip(range(len(AGENT_NAMES)+1), [0 for _ in range(len(AGENT_NAMES)+1)]))
    for instance_id, instance_results in results.items():
        if len(instance_results) < len(AGENT_NAMES):
            continue

        total = sum([1 if x == y else 0 for _, x, y, _, _, _ in instance_results])
        data[total] += 1

    # Calculate the total sum
    total_sum = sum(data.values())

    # Create labels and values lists
    labels = [f"{k}/{len(data)-1}" for k in data.keys()]
    values = [item*100/total_sum for item in data.values()]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot the bar chart
    ax.bar(labels, values)

    # Set the title and axis labels
    ax.set_title("Distribution of Correct Resolved Status Predictions across Agentic Workflows")
    ax.set_xlabel("#-of Matching Pairs of Predicted and True Resolved Statuses")
    ax.set_ylabel("Percentage (%)")

    # Rotate the x-axis labels for better visibility
    plt.xticks()

    filename = f'agent-comparison-test-centric-{args.model_name}-build.png'
    plt.savefig(Path.cwd() / 'analysis/figures' / filename)

    # Display the chart
    plt.show()


def plot_progress_trajectories(all_agents_pass_rates, key):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over the agents
    for agent_name, pass_rates in all_agents_pass_rates.items():
        sorted_pass_rates = sorted(pass_rates)

        # Calculate the frequency for each pass rate
        pass_rate_counts = [pass_rates.count(rate) for rate in sorted_pass_rates]

        # Plot the progress trajectory
        ax.plot(sorted_pass_rates, pass_rate_counts, marker='o', label=agent_name)

    # Set the title and axis labels
    ax.set_title(f"Progress Trajectories for {key} Rates")
    ax.set_xlabel(f"LLM-Predicted {key} Rate (%)")
    ax.set_ylabel("Task Distribution")

    # Add a legend
    ax.legend()

    _key = '-'.join(key.lower().split(' '))
    filename = f'agent-comparison-test-centric-{args.model_name}-{_key}-trajectories.png'
    plt.savefig(Path.cwd() / 'analysis/figures' / filename)

    # Show the plot
    plt.show()


def compare_on_build_status(path_to_assets, benchmark, model_name, key):
    results = {}
    add_aggregate_agent = False

    for agent_name in AGENT_NAMES:
        resolved_status_file = f'{benchmark}.{agent_name}.resolved-status.json'
        with open(str(path_to_assets / 'cache' / resolved_status_file), 'r') as f:
            resolved_status_mapper = json.load(f)

        filename = f'{benchmark}.{agent_name}.execution.test-centric{key}.json'
        path_to_file = Path(path_to_assets / f'results-{model_name}') / filename
        with open(path_to_file, 'r') as f:
            agent_results = json.load(f)

        for instance_id, instance_results in agent_results.items():

            instance_preds, instance_true = [], []
            for item in instance_results:
                length = len(item['prompt_inputs']['test'].split('\n'))
                _pred = 1 if item['pred'] == 'yes' else 0
                if item['confidence'] <=65 and length >= 50:
                    instance_preds.append(0)
                else:
                    instance_preds.append(_pred)
                instance_true.append(1 if item['true'] == 'pass' else 0)

            agg_pred = 0 if 0 in instance_preds else 1
            agg_true = 0 if 0 in instance_true else 1
            
            # The +1 is related to the aggregate agent bellow
            total = len(instance_true) + (1 if add_aggregate_agent else 0)
            if instance_id not in results:
                results[instance_id] = [(agent_name, agg_pred, agg_true, instance_preds, instance_true, total)]
            else:
                results[instance_id].append((agent_name, agg_pred, agg_true, instance_preds, instance_true, total))

    if add_aggregate_agent:
        for instance_id, agent_perf_list in results.items():

            top_agent_name, top_agg_pred, top_agg_true, top_instance_preds, top_instance_true, total = max(agent_perf_list, key=lambda x: sum(x[3]))
            results[instance_id].append(('aggregate_agent', top_agg_pred, top_agg_true, top_instance_preds, top_instance_true, total))

    return results


def compare_trajectories(path_to_assets, benchmark, model_name, key):
    results = {}
    for agent_name in AGENT_NAMES:
        filename = f'{benchmark}.{agent_name}.execution.test-centric{key}.json'
        path_to_file = Path(path_to_assets / f'results-{model_name}') / filename
        with open(path_to_file, 'r') as f:
            agent_results = json.load(f)

        for instance_id, instance_results in agent_results.items():
            preds = [1 if item['pred'] == 'yes' else 0 for item in instance_results]
            preds = []
            for item in instance_results:
                length = len(item['prompt_inputs']['test'].split('\n'))
                _pred = 1 if item['pred'] == 'yes' else 0
                if item['confidence'] <=65 and length >= 50:
                    preds.append(0)
                else:
                    preds.append(_pred)
            true = [1 if item['true'] == 'pass' else 0 for item in instance_results]
            pred_progress = sum(preds) / len(preds)
            true_progress = sum(true) / len(true)
            if instance_id in results:
                results[instance_id].append((agent_name, pred_progress, true_progress))
            else:
                results[instance_id] = [(agent_name, pred_progress, true_progress)]

    filtered_results = {k: v for k, v in results.items() if len(v) == len(AGENT_NAMES)}
    for agent_name in AGENT_NAMES:
        plot_name = f'{benchmark}.{agent_name}.density.test-centric{key}.png'
        output_path = Path.cwd() / 'analysis/figures' / plot_name

        true_pass_rates = [
            [item for item in sublist if item[0] == agent_name][0][2]*100 for sublist in filtered_results.values()
        ]
        pred_pass_rates = [
            [item for item in sublist if item[0] == agent_name][0][1]*100 for sublist in filtered_results.values()
        ]

        plot_test_pass_rate_density(
            true_pass_rates,
            pred_pass_rates,
            benchmark,
            output_path
        )

    # Plot with all agents
    plot_name = f'{benchmark}.all_agents.density.test-centric{key}.png'
    output_path = Path.cwd() / 'analysis/figures' / plot_name

    true_pass_rates = []
    pred_pass_rates = []

    for agent_name in AGENT_NAMES:
        true_pass_rates.extend([
            [item for item in sublist if item[0] == agent_name][0][2]*100 for sublist in filtered_results.values()
        ])
        pred_pass_rates.extend([
            [item for item in sublist if item[0] == agent_name][0][1]*100 for sublist in filtered_results.values()
        ])

    plot_test_pass_rate_density(
        true_pass_rates,
        pred_pass_rates,
        benchmark,
        output_path
    )

    all_pred_pass_rates = {}
    for iid, instance_results in filtered_results.items():
        for _results in instance_results:
            if _results[0] not in all_pred_pass_rates:
                all_pred_pass_rates[_results[0]] = [_results[1]]
            else:
                all_pred_pass_rates[_results[0]].append(_results[1])
    plot_progress_trajectories(all_pred_pass_rates, key='Test Pass')

    return results



def compare_rankings(results, edit_distances):
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig, ax1 = plt.subplots(figsize=(10,7))

    pred_pass_rates, true_pass_rates = [], []
    base_progress = []
    for iid, item in results.items():
        if len(item) != len(AGENT_NAMES): continue
        pred_pass_rates.append([sum(agent_result[3])/agent_result[-1] for agent_result in item])
        true_pass_rates.append([sum(agent_result[4])/agent_result[-1] for agent_result in item])
        base_progress.append([
            edit_distances[agent_result[0]][iid]['edit_embedding'] \
            for agent_result in item])

    # Plot spearman ranking coefficents
    spearmanr_coeffs = [
        spearmanr(item[0], item[1]).statistic for item in zip(
            [np.argsort(pred).tolist() for pred in pred_pass_rates],
            [np.argsort(true).tolist() for true in true_pass_rates],
        )
    ]
    base_spearmanr_coeffs = [
        spearmanr(item[0], item[1]).statistic for item in zip(
            [np.argsort(pred).tolist() for pred in base_progress],
            [np.argsort(true).tolist() for true in true_pass_rates],
        )
    ]
    sns.kdeplot(spearmanr_coeffs, ax=ax1, fill=True, label='LLM-Predicted Task Progress Rankings')
    sns.kdeplot(base_spearmanr_coeffs, ax=ax1, color='red', linestyle = '--', label='Edit Distance-Based Task Progress Rankings')
    ax1.set_xlim(-1, 1)
    ax1.legend()

    ax1.set_xlabel('Spearman\'s Ranking Coefficient')
    ax1.set_ylabel('Density')
    ax1.set_title('Spearman\'s Ranking Coefficient Distribution for LLM-Predicted v/s Actual Task Progress Rankings')

    # ax2.violinplot([spearmanr_coeffs, base_spearmanr_coeffs])
    # ax2.set_xticks([1, 2])
    # ax2.set_xticklabels(['LLM-Predicted', 'Edit Distance-Based'])
    # ax2.set_xlabel('Task Progress Rankings')
    # ax2.set_ylabel('Spearman\'s Ranking Coefficient')
    # ax2.set_title('Spearman\'s Ranking Coefficient Distribution for Task Progress Rankings')

    plt.savefig(Path.cwd() / 'analysis/figures' / f'rankings-density.png')
    plt.show()


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


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn * 100 / (tn + fp)
    print((f"\tAccuracy: {accuracy:.2f}%\tPrecision: {precision:.2f}%\t"
           f"\tRecall: {recall:.2f}%\tF1 Score: {f1:.2f}%\tTrue Negative Rate: {tnr:.2f}%"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-Based Execution-Free Evalutation of Agent Workflows')

    ## Pipeline arguments
    parser.add_argument('--path_to_assets', type=str, default='../resources',
                        help="Path to dataset resources.")
    parser.add_argument('--type', type=str, default='Lite',
                        choices=['Lite', 'test'], help='SWE-Bench variant')
    parser.add_argument('--filter', action='store_true',
                        help="Whether to filter functions and tests in patch based on dependencies")
    parser.add_argument('--model_name', default='opus-3',
                        choices=['opus-3', 'sonnet-3', 'sonnet-3-5'], help='Claude model key')

    args = parser.parse_args()
    path_to_assets = Path(args.path_to_assets).resolve()

    benchmark = f'swe-bench_{args.type}' if args.type == 'Lite' else args.benchmark
    from llm_agent_evaluation.data.swe_bench.collect import SWEBenchPatchCollector
    PatchCollectorCls = SWEBenchPatchCollector
    patch_collector = PatchCollectorCls(path_to_assets, args.type)

    logger.info(
        'Comparing test-centric proxies for diff patches with function/class-level context.'
    )

    build_results_patch = compare_on_build_status(path_to_assets, benchmark, args.model_name, '-patch')
    plot_build_status_comparison(build_results_patch)

    filtered_results = {
        agent_name: {'macro-pred': [], 'macro-true': [], 'micro-pred': [], 'micro-true': []}
        for agent_name in AGENT_NAMES
    }
    logger.info(
        f'Skipping {len([True for result in build_results_patch.values() if len(result) != len(AGENT_NAMES)])}/{len(build_results_patch)} instances')
    for result in build_results_patch.values():
        if len(result) != len(AGENT_NAMES):
            continue
        for item in result:
            filtered_results[item[0]]['macro-pred'].append(item[1])
            filtered_results[item[0]]['macro-true'].append(item[2])
            filtered_results[item[0]]['micro-pred'] += item[3]
            filtered_results[item[0]]['micro-true'] += item[4]

    for agent_name, _results in filtered_results.items():
        logger.info(f'...    Micro-Results for {agent_name}    ....')
        compute_metrics(_results['micro-true'], _results['micro-pred'])
        logger.info(f'...    Macro-Results for {agent_name}    ....')
        compute_metrics(_results['macro-true'], _results['macro-pred'])

    #build_results_holistic = compare_on_build_status(path_to_assets, benchmark, args.model_name, '')

    # path_to_results = Path(args.path_to_assets) / f'results-{args.model_name}'
    # edit_results = {
    #     agent_name: json.load(open(str(path_to_results / f'{benchmark}.{agent_name}.lexical.json'), 'r')) \
    #     for agent_name in AGENT_NAMES
    # }

    # compare_rankings(build_results_patch, edit_results)
    rank_results = compare_trajectories(path_to_assets, benchmark, args.model_name, '-patch')
