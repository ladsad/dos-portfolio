# CodeWhisper

CodeWhisper is an intelligent tool for auto-generating documentation and analyzing code quality.

## Project Structure

- `backend/`: Python FastAPI backend.
- `vscode-extension/`: VS Code extension for IDE integration.
- `Documents/`: Project documentation.

## Features

- **Code Analysis**: Calculates cyclomatic complexity, maintainability index, and detects anomalies.
- **Auto-Documentation**: Generates docstrings using **CodeT5+** fine-tuned with **QLoRA**.
- **Dashboard**: Visualizes project health and metrics (Streamlit MVP).
- **VS Code Extension**: Right-click context menu for real-time documentation generation.

## Model Training

The documentation generation model uses **CodeT5-small** fine-tuned on **CodeXGLUE** (Python/Java) with QLoRA.

### Model Performance

| Metric | Score |
| :--- | :--- |
| **BLEU** | 36.65 |
| **ROUGE-L** | 62.17 |
| **BERTScore** | 0.93 |
