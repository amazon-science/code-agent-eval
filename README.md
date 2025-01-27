# LLMAgentEvaluation

## Getting Started
This section describes the prerequisites, and contains instructions to get the project up and running.

### Setup 

#### A. Project Environment
If ``conda`` is already installed, update to the latest version with ``conda update conda``, and skip steps 1 - 3:
  1. Download the latest, appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine (tested with ``conda 23.11.0``).
  2. Install  it by running the `conda_install.sh` file, with the command:
     ```bash
     $ bash conda_install.sh
     ```
  3. Add `conda` to bash profile:
     ```bash
     $ source ~/.bashrc
     ```
  4. Navigate to ``LLMAgentEvaluation`` (top-level directory) and create a conda virtual environment with the included `environment.yml` file using the following command:
     
     ```bash
     $ conda env create -f environment.yml
     ```

     To test successful installation, make sure ``llmeval`` appears in the list of conda environments returned with ``conda env list``.
  5. Activate the virtual environment with the following command:
     
     ```bash
     $ conda activate llmeval
     ```
  6. Install ``llm_agent_evaluation``:
     
     ```bash
     $ pip install .
     ```       

#### B. Prerequisites
Create an `.env` file in the top-level directory containing `configure.py`, with the following key-value pairs:
     
     ```bash
      AWS_ACCESS_KEY_ID=
      AWS_SECRET_ACCESS_KEY=
      AWS_SESSION_TOKEN=
     ```

#### C. Data Resources 

1. Setting up BugsInPy/SWE-Bench datasets: `configure.py` helps set up both BugsInPy and SWE-Bench` (both 'Lite' and 'test' variants) benchmark datasets.
    * BugsInPy: `$ python configure.py --benchmark BugsInPy`
    * SWE-Bench: `$ python configure.py --benchmark swe-bench --type {Lite|test}`

   This creates `base` and `gold` repository snapshots in `resources/<benchmark>/projects/<project-name>/instances/<instance-id>/snapshots/`.

2. Perturbation-based datasets: One way to get failure patches is to perturb on the gold patches by
    * Removing a random hunk from the gold patch
    * (TODO) Perturbing on the gold patch with an LLM

   To do so, first, navigate to `workflows` and run `perturbation.py` as per the sample commands. This creates `perturb-removal` or `perturb-llm` repository snapshot (after applying the perturbed patch) in `resources/<benchmark>/projects/<project-name>/instances/<instance-id>/snapshots/`, and creates the corresponding `<benchmark>.patches.perturb-{removal|llm}.{none|function}-context.pkl` files in `resources/cache`.

3. Agentic workflows:
    * **[Disclaimer]** This is currently set up to only work with SWE-Bench.
    * To set up a specific agent (allowed, only if evaluated on both `lite` and `test` SWE-Bench variants, https://github.com/swe-bench/experiments/tree/main/evaluation/), navigate to `workflows` and run `agentic_patches.py` as per the sample commands. This creates a repository snapshot for the agent in `resources/<benchmark>/projects/<project-name>/instances/<instance-id>/snapshots/` and caches all individual test case to status (i.e., PASSED, FAILED, or ERROR) maps at `resources/cache/<benchmark>.<agent-name>.test-status.json`.

### Directory Structure

<Add directory structure here>

## Going Ahead

1. Extend `workflows/agentic_patches.py` to BugsInPy.
2. Add utilities to create patches with dependency-context.
3. Build partial ordering based on ordering of change-specific functions in the call graph for the repository.
4. Adding more traditional or LLM-based scorers into `scorers` module.

For 2--3, call graph processing utilities can be added to `external/call_graph` and caller/callee relations for a specific method in any repository can be extracted. This can be structured along the lines of [r2e](https://github.com/r2e-project/r2e/tree/main/r2e/pat/callgraph).
