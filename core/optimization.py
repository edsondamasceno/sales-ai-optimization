import numpy as np

def optimize_price_stock(preds, stock, base_price, cost):

    best_profit = -np.inf
    best_price = base_price
    best_order = 0

    price_range = np.linspace(base_price * 0.7, base_price * 1.3, 15)

    for price in price_range:

        elasticity = -1.3
        adjusted_preds = [p * (price / base_price) ** elasticity for p in preds]

        for order_qty in range(0, int(sum(preds) * 2)):

            inventory = stock + order_qty
            profit = 0

            for demand in adjusted_preds:

                sold = min(inventory, demand)
                lost = max(0, demand - inventory)

                revenue = sold * price
                purchase_cost = order_qty * cost
                holding = inventory * 0.05
                penalty = lost * price * 1.5

                profit += revenue - purchase_cost - holding - penalty

                inventory -= sold

            if profit > best_profit:
                best_profit = profit
                best_price = price
                best_order = order_qty

    return best_price, best_order, best_profit