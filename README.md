# Boolean Expression Simplification using Deep Reinforcement Learning

## Overview

This project explores the use of deep reinforcement learning to simplify boolean expressions. The goal is to train an agent that can apply simplification rules to reduce the complexity of a given boolean formula. The project implements three different approaches:

1.  **MLP-based Agent:** A standard Deep Q-Network (DQN) with a Multi-Layer Perceptron (MLP) as the function approximator. This agent uses a feature vector (e.g., operator counts, expression depth) to represent the state.
2.  **Sequence-based Agent (LSTM):** An improved version of the MLP agent that uses a Long Short-Term Memory (LSTM) network to process the boolean expression as a sequence of tokens. This allows the agent to learn from the structure of the expression.
3.  **GNN-based Agent:** A more advanced DQN agent that uses a Graph Neural Network (GNN) to represent the boolean expression as an Abstract Syntax Tree (AST). This is the most powerful approach as it directly learns from the graph structure.

## Installation

The project supports both `pip` and `uv` for package management. Choose one of the following methods to install dependencies and run the project.

### Prerequisites
- Python 3.8+

### Using pip

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TarunNagarajan/boolrl.git
    cd boolrl
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **Note on `torch_geometric`:** The `torch_geometric` library can have specific installation requirements depending on your PyTorch version and CUDA version. If you encounter issues, please refer to the official [torch_geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Using uv (recommended)

`uv` is a fast Python package installer and resolver. If you don't have it installed, follow the [installation guide](https://github.com/astral-sh/uv).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TarunNagarajan/boolrl.git
    cd boolrl
    ```

2.  **Install and sync dependencies with uv:**
    ```bash
    uv sync
    ```

3.  **Run commands in the uv environment:**
    ```bash
    uv run python boolrl/main.py --model_type mlp
    ```

## Training

The `main.py` script is used to train all three agents. You can select the agent to train using the `--model_type` command-line argument.


### MLP Agent

```bash
python boolrl/main.py --model_type mlp
```

### Sequence Agent

```bash
python boolrl/main.py --model_type seq
```

### GNN Agent

```bash
python boolrl/main.py --model_type gnn
```

The script will save periodic checkpoints (e.g., `checkpoint_mlp_e100.pth`) and a final model (`checkpoint_mlp_final.pth`). A training plot will also be saved.

## Testing

After training, you can evaluate the performance of the agents using the test scripts.


### Test MLP Agent

```bash
python boolrl/tests/mlp/test_mlp.py
```

### Test Sequence Agent

```bash
python boolrl/tests/seq/test_seq.py
```

### Test GNN Agent

```bash
python boolrl/tests/gnn/test_gnn.py
```

These scripts will load the corresponding final checkpoint file and run the agent on a number of test episodes, reporting the accuracy.

