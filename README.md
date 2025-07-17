# SKU Selection Optimization

This project implements an SKU selection optimization process. It takes historical order data, scores SKUs, performs a two-stage selection (rough and fine), evaluates the selection based on simulated warehouse efficiency, and decides whether to adopt the new SKU selection.

## Features

- **Data-Driven SKU Scoring**: Scores SKUs based on historical data, considering frequency, co-occurrence, and quantity.
- **Two-Stage Selection**: 
    1. **Rough Selection**: Narrows down the SKU list to a manageable number.
    2. **Fine Selection**: Uses optimization (Gurobi) to find the best combination of SKUs under given constraints.
- **Hyperparameter Tuning**: Utilizes Ray Tune for automated hyperparameter optimization of the selection models.
- **Warehouse Simulation**: Evaluates the profitability and efficiency of a given SKU selection.
- **State Persistence**: Saves and loads the system state, including the current best SKU selection and scores.

## Project Structure

```
├── config/         # Configuration files
│   └── config.py
├── data/           # Data files (not in git)
├── model/          # Core logic
│   ├── pre_process/  # Data preprocessing
│   ├── rough_select/ # Rough selection algorithm
│   ├── fine_select/  # Fine selection algorithm (using Gurobi)
│   ├── evaluator.py  # Evaluates selection results
│   └── sku_optimizer.py # Main optimizer orchestrating the process
├── notebooks/      # Jupyter notebooks for analysis
├── schema/         # Data models (dataclasses)
│   └── models.py
├── store/          # Data persistence logic
│   └── data_store.py
├── utils/          # Utility functions
├── main.py         # Main script to run the optimization loop
├── requirements.txt # Python dependencies
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dlrdaile/smart_sku_selector.git
    cd smart_sku_selector
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Gurobi Installation**:
    This project uses Gurobi for optimization. You need a Gurobi license. Please refer to the [Gurobi documentation](https://www.gurobi.com/documentation/) for installation and license setup.

## Usage

1.  **Configure the project:**
    -   Update the paths and parameters in `config/config.py` to match your environment.

2.  **Run the optimization:**
    -   Execute the main script and provide the path to the new data when prompted.
    ```bash
    python main.py
    ```
    -   The script will then prompt for the path to the new data files:
    ```
    new_data_path: <path_to_your_new_data_directory>
    ```

3.  **Output:**
    -   The optimization results and logs will be printed to the console.
    -   The system state is saved to `data/storage.json`.
    -   Evaluation outputs (like simulation results) are saved in the `output/` directory.