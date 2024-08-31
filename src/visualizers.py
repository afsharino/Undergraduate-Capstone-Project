# Import Libraries
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def plot(indicator_name:str, prices:np.ndarray, indicator_values:np.ndarray, profits:list, cash:list, bitcoin:list, total:list, dates:list, absolute_drawdown:list):
    # Initialize empty lists to store data for plotting
    price_data = list(prices)
    fear_and_greed_data = list(indicator_values)
    profit_data = profits
    cash_balance_data = cash
    bitcoin_amount_data = bitcoin
    total_balance_data = total
    date_data = list(dates)
    absolute_drawdown_data = absolute_drawdown

    # Find Maximum Drawdown and the corresponding date
    max_drawdown_value = max(absolute_drawdown_data)
    max_drawdown_index = absolute_drawdown_data.index(max_drawdown_value)
    max_drawdown_date = date_data[max_drawdown_index]

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each data series
    fig.add_trace(go.Scatter(x=date_data, y=price_data, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=date_data, y=fear_and_greed_data, mode='markers', name=indicator_name, marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=date_data, y=profit_data, mode='lines', name='Profit'))
    fig.add_trace(go.Scatter(x=date_data, y=cash_balance_data, mode='lines', name='Cash Balance'))
    fig.add_trace(go.Scatter(x=date_data, y=bitcoin_amount_data, mode='lines', name='Bitcoin Amount'))
    fig.add_trace(go.Scatter(x=date_data, y=total_balance_data, mode='lines', name='Total Balance'))

     # Add trace for Absolute Drawdown
    fig.add_trace(go.Scatter(x=date_data, y=absolute_drawdown_data, mode='lines', name='Absolute Drawdown'))

    # Mark the Maximum Drawdown with a point and annotation
    fig.add_trace(go.Scatter(x=[max_drawdown_date], y=[max_drawdown_value], name='Max Drawdown', marker=dict(size=12, color='red', symbol='bowtie')))


    # Update layout with labels
    fig.update_layout(title={'text': 'Price and Trading Metrics Progression', 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                      height=800,
                      xaxis_title='Date',
                      yaxis_title='Value',
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=0.74),
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
    fig.update_layout(title={'text': 'Price and Trading Metrics Progression', 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                      height=900,
                      showlegend=True,
                      hovermode='x unified')

    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=3, col=1)

    fig.show()


def plot_fitness(fitness_values, num_generations, label):
    plt.plot(np.arange(1, num_generations+1),fitness_values[:-1], label=label, color='purple')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Fitness Over Generations")

    plt.legend()  # Ensure the legend is displayed
    plt.show()


def compare_models(model_results, model_names, dates):
    """
    Compare multiple models by plotting their results on the same chart.

    Args:
        model_results (list): A list of tuples, where each tuple contains (prices, indicator_values, profits, cash, bitcoin, total).
        model_names (list): A list of model names corresponding to the model results.
        dates (list): A list of dates for the x-axis.
    """
    
    # Create Plotly figure
    fig = go.Figure()

    # Iterate over each model's results and plot them on the same chart
    for i, (prices, indicator_values, profits, cash, bitcoin, total, drawdown) in enumerate(model_results):
        model_name = model_names[i]

        # Add traces for each model's data series
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name=f'{model_name} Price'))
        fig.add_trace(go.Scatter(x=dates, y=indicator_values, mode='lines', name=f'{model_name} Indicator'))
        fig.add_trace(go.Scatter(x=dates, y=profits, mode='lines', name=f'{model_name} Profit'))
        fig.add_trace(go.Scatter(x=dates, y=cash, mode='lines', name=f'{model_name} Cash Balance'))
        fig.add_trace(go.Scatter(x=dates, y=bitcoin, mode='lines', name=f'{model_name} Bitcoin Amount'))
        fig.add_trace(go.Scatter(x=dates, y=total, mode='lines', name=f'{model_name} Total Balance'))
        fig.add_trace(go.Scatter(x=dates, y=drawdown, mode='lines', name=f'{model_name} Absolute Drawdown',))


    # Update layout with labels
    fig.update_layout(
        title={'text': 'Comparison of Trading Strategies', 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
        legend=dict(
            orientation="h",  # Arrange the legend items horizontally
            yanchor="top", 
            y=-0.2,  # Adjust this value to move the legend further down
            xanchor="center", 
            x=0.5,

        ),
        xaxis_title='Date',
        yaxis_title='Value',
        height=800,
        hovermode='x unified',
    )

    # Show the figure
    fig.show()
