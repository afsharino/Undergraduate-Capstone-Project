import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def subplot(indicator_name, uptrend_data, sideway_data, downtrend_data):
    """
    Plot the data in three subplots for uptrend, sideway, and downtrend using Plotly.

    Args:
        indicator_name (str): Name of the indicator.
        uptrend_data (tuple): Tuple containing prices, indicator values, profits, cash, bitcoin, total, and dates for uptrend.
        sideway_data (tuple): Tuple containing prices, indicator values, profits, cash, bitcoin, total, and dates for sideway.
        downtrend_data (tuple): Tuple containing prices, indicator values, profits, cash, bitcoin, total, and dates for downtrend.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Uptrend Segment", "Sideway Segment", "Downtrend Segment"))
    # fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, subplot_titles=("Uptrend Segment", "Sideway Segment", "Downtrend Segment"))

    # Define a helper function to add traces for each segment
    def add_traces(fig, row, data):
        prices, indicator_values, profits, cash, bitcoin, total, dates = data
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=indicator_values, mode='lines', name=indicator_name), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=profits, mode='lines', name='Profit'), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=cash, mode='lines', name='Cash Balance'), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bitcoin, mode='lines', name='Bitcoin Amount'), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=total, mode='lines', name='Total Balance'), row=row, col=1)

    # Add traces for each segment
    add_traces(fig, 1, uptrend_data)
    add_traces(fig, 2, sideway_data)
    add_traces(fig, 3, downtrend_data)

    # Update layout
    fig.update_layout(title='Price and Trading Metrics Progression',
                      height=900, showlegend=False, 
                      hovermode='x unified')

    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=3, col=1)

    fig.show()