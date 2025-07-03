import time
import random
from typing import List
from store.data_store import DataStore
from schema.models import SystemState, SKU, Order
from model.sku_optimizer import SkuOptimizer
from model.evaluator import Evaluator

def initialize_system_state() -> SystemState:
    """Creates a default system state if none exists."""
    print("Initializing system with sample data...")
    sample_skus = [SKU(id=f'sku_{i:03d}', name=f'Product {i}') for i in range(100)]
    return SystemState(all_skus=sample_skus)

def run_optimization_cycle(state: SystemState, new_orders: List[Order]):
    """Runs one full cycle of the SKU optimization process."""
    optimizer = SkuOptimizer()
    evaluator = Evaluator()

    print("\n--- Running Optimization Cycle ---")
    # 1. SKU Scoring (now with new order data)
    print("1. Scoring SKUs...")
    new_scores = optimizer.score_skus(state.all_skus, state.previous_scores, new_orders)

    # 2. Rough Selection
    print("2. Performing rough selection...")
    rough_selected_skus = optimizer.rough_selection(state.all_skus, new_scores, new_orders)

    # 3. Fine Selection
    print("3. Performing fine selection...")
    new_selection_result = optimizer.fine_selection(rough_selected_skus, new_orders)


    # 4. Evaluation
    print("4. Evaluating new selection...")
    evaluator.simulate_warehouse_efficiency(new_selection_result)
    old_profit = evaluator.calculate_profit(state.previous_selection) if state.previous_selection else 0
    new_profit = evaluator.calculate_profit(new_selection_result)
    changeover_cost = evaluator.calculate_changeover_cost(state.previous_selection, new_selection_result)
    new_selection_result.profit = new_profit

    print(f"   - Old Selection Profit: {old_profit:.2f}")
    print(f"   - New Selection Gross Profit: {new_profit:.2f}")
    print(f"   - Changeover Cost: {changeover_cost:.2f}")

    # 5. Decision
    print("5. Making decision...")
    if evaluator.should_switch(old_profit, new_profit, changeover_cost):
        print("   -> Decision: Switch to the new selection.")
        state.previous_selection = new_selection_result
    else:
        print("   -> Decision: Keep the old selection.")

    # 6. Update and Save State
    print("6. Saving state...")
    state.previous_scores = new_scores
    DataStore.save_state(state)
    print("--- Optimization Cycle Complete ---")

def simulate_new_orders(all_skus: List[SKU]) -> List[Order]:
    """Generates a list of simulated new orders."""
    num_orders = random.randint(1, 5)
    orders = []
    for i in range(num_orders):
        num_items = random.randint(1, 5)
        items = {}
        for _ in range(num_items):
            sku = random.choice(all_skus)
            items[sku.id] = items.get(sku.id, 0) + random.randint(1, 3)
        orders.append(Order(id=f"order_{int(time.time())}_{i}", items=items))
    print(f"\n>>> Detected {len(orders)} new orders.")
    return orders

def main():
    """Main loop to listen for new orders and trigger optimization."""
    state = DataStore.load_state()
    if not state:
        state = initialize_system_state()
        DataStore.save_state(state) # Save initial state

    print("System is running. Waiting for new orders...")
    try:
        while True:

            # Simulate the arrival of new orders
            new_orders = simulate_new_orders(state.all_skus)
            
            # Trigger the optimization cycle
            run_optimization_cycle(state, new_orders)

    except KeyboardInterrupt:
        print("\nSystem shutting down.")

if __name__ == "__main__":
    main()