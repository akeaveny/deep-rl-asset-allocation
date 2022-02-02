# deep-rl-asset-allocation

### This work is baed on the following repos:
- [deep-rl-for-automated-stock-trading-ensemble-strategy-ICAIF-2020](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020) 

### Prerequisites

This repo has been developed with python >= 3.8.0

- Anaconda (or virtual env):
```
conda create -n deep-rl python=3.8
conda activate deep-rl
```

- Requirements
```
pip install -r requirements.txt
```

- Setuptools (to install this as a python package)
```
# building
python setup.py sdist bdist_wheel
pip install -e .
```
```
# remove local build
python setup.py clean --all
```

- Formatting with Yapf (.vscode/settings.json)
```
{
    "python.linting.pylintEnabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintPath": "pylint",
    "editor.formatOnSave": true,
    "python.formatting.provider": "yapf",
    "python.formatting.yapfArgs": [
        "--style={based_on_style: pep8, column_limit: 165, indent_width: 4, allow_multiline_dictionary_keys: true, split_all_comma_separated_values: false}"
    ],
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
    // Custom Colour for Status Bar (project specific)
    "workbench.colorCustomizations": {
        "titleBar.activeBackground": "#9c1c1c"
    }
}
```
