import numpy as np
import plotly.graph_objects as go


def plot(indicator_name:str, prices:np.ndarray, indicator_values:np.ndarray, profits:list, cash:list, bitcoin:list, total:list, dates:list):
    # Initialize empty lists to store data for plotting
    price_data = list(prices)
    fear_and_greed_data = list(indicator_values)
    profit_data = profits
    cash_balance_data = cash
    bitcoin_amount_data = bitcoin
    total_balance_data = total
    date_data = list(dates)

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each data series
    fig.add_trace(go.Scatter(x=date_data, y=price_data, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=date_data, y=fear_and_greed_data, mode='markers', name=indicator_name, marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=date_data, y=profit_data, mode='lines', name='Profit'))
    fig.add_trace(go.Scatter(x=date_data, y=cash_balance_data, mode='lines', name='Cash Balance'))
    fig.add_trace(go.Scatter(x=date_data, y=bitcoin_amount_data, mode='lines', name='Bitcoin Amount'))
    fig.add_trace(go.Scatter(x=date_data, y=total_balance_data, mode='lines', name='Total Balance'))

    # Update layout with labels
    fig.update_layout(title='Price and Trading Metrics Progression',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified')

    # Show the figure
    fig.show()
