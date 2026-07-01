# env directory

This directory records dependency and environment information used by the experiments.

## Files

```text
env/
  requirements.txt               Main dependency entry point for the public package.
  requirements_experiment2.txt    Historical dependency snapshot for Experiment 2.
```

## Usage

Install the main requirements before running the scripts:

```powershell
python -m pip install -r env/requirements.txt
```

Some experiments require local model weights or model downloads that are not included in this repository.
