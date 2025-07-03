# SKU Selection Optimization

This project implements the outer loop logic for an SKU selection optimization process based on a given flowchart.

The core selection algorithm is treated as a black box.

## Project Structure

- `main.py`: Main script to run the optimization loop.
- `config.py`: Configuration variables.
- `models.py`: Data models for SKU, SelectionResult, etc.
- `data_store.py`: Handles data persistence (loading/saving from/to a file).
- `sku_optimizer.py`: Placeholder for the black-box SKU optimization algorithm.
- `evaluator.py`: Logic for evaluating selection results (profits, costs, simulation).
- `data/`: Directory for data storage.
  - `storage.json`: The file used to store algorithm state.