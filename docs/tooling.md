# Tooling And Commands

## Environment

All UML generation in this audit was done from the `netpyne` conda environment.

## Install Commands Used

```bash
/home/nnovikov/conda_env/netpyne/bin/pip install pylint
/home/nnovikov/miniforge3/bin/mamba install -y -p /home/nnovikov/conda_env/netpyne graphviz
```

## Verification Commands

```bash
/home/nnovikov/conda_env/netpyne/bin/python --version
/home/nnovikov/conda_env/netpyne/bin/pyreverse --version
/home/nnovikov/conda_env/netpyne/bin/dot -V
```

## Fontconfig Cache Hint

Graphviz worked, but it emitted Fontconfig cache warnings because the env cache directories were not writable from the sandboxed session. To keep reruns quiet, use a writable cache directory:

```bash
export XDG_CACHE_HOME=/tmp/codex-fontcache
mkdir -p "$XDG_CACHE_HOME"
```

## `pyreverse` Commands Used

```bash
XDG_CACHE_HOME=/tmp/codex-fontcache \
PYTHONPATH=/home/nnovikov/repo/model_tuner \
/home/nnovikov/conda_env/netpyne/bin/pyreverse \
  -o dot \
  -p model_tuner_data_proc \
  /home/nnovikov/repo/model_tuner/model_tuner/data_proc

XDG_CACHE_HOME=/tmp/codex-fontcache \
PYTHONPATH=/home/nnovikov/repo/sim_res_analyzer/code \
/home/nnovikov/conda_env/netpyne/bin/pyreverse \
  -o dot \
  -p sim_res_analyzer_core \
  /home/nnovikov/repo/sim_res_analyzer/code/data_keeper.py \
  /home/nnovikov/repo/sim_res_analyzer/code/data_proc.py \
  /home/nnovikov/repo/sim_res_analyzer/code/sim_res_parser.py

XDG_CACHE_HOME=/tmp/codex-fontcache \
PYTHONPATH=/home/nnovikov/repo/batch_osc_analyzer \
/home/nnovikov/conda_env/netpyne/bin/pyreverse \
  -o dot \
  -p batch_osc_core \
  /home/nnovikov/repo/batch_osc_analyzer/batch_analyzer.py \
  /home/nnovikov/repo/batch_osc_analyzer/lfp_analyzer.py
```

## PNG Rendering Commands

```bash
XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng classes_model_tuner_data_proc.dot -o classes_model_tuner_data_proc.png

XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng packages_model_tuner_data_proc.dot -o packages_model_tuner_data_proc.png

XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng classes_sim_res_analyzer_core.dot -o classes_sim_res_analyzer_core.png

XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng packages_sim_res_analyzer_core.dot -o packages_sim_res_analyzer_core.png

XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng classes_batch_osc_core.dot -o classes_batch_osc_core.png

XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng packages_batch_osc_core.dot -o packages_batch_osc_core.png
```

## Custom Diagram Rendering Pattern

```bash
XDG_CACHE_HOME=/tmp/codex-fontcache \
/home/nnovikov/conda_env/netpyne/bin/dot -Tpng source_lineage.dot -o source_lineage.png
```

Repeat the same pattern for:

- `abstraction_layers.dot`
- `a1_ou_tuning_workflow.dot`
- `storage_formats.dot`
- `batch_collection.dot`

## Stop Condition

If `pyreverse` or `dot` stop working in the `netpyne` env after a fresh install into that env, stop and ask the user before continuing. The UML outputs in this audit are intended to come from `pyreverse`, not from a substitute tool.
