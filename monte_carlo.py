'''
MonteCarlo simulation for a singular ticker (by Misha Vashurkin,shumihaaa)
Utilizes GBM (equation1: St+1​=St​×e^(μ−​σ^2/2)Δt+σsqrt(Δt​)×Z)
'''
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo Stock Simulator (GBM)", layout="wide")
st.title("📈 Monte Carlo Stock Price Simulator ")

st.sidebar.header("Simulation Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
num_simulations = st.sidebar.slider("Number of Simulations", 5000, 100000, 20000, step=5000)
num_days = st.sidebar.slider("Days to Simulate into Future", 30, 365, 180)
st.sidebar.markdown("---")

try:

    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found. Please check the ticker or date range.")
    else:
        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"


        data['Daily Return'] = data[price_col].pct_change().dropna()
        mean_return = float(data['Daily Return'].mean())
        std_dev = float(data['Daily Return'].std())
        last_price = float(data[price_col].iloc[-1])

     
        st.sidebar.subheader("Historical Stats")
        st.sidebar.write(f"Mean Daily Return: {mean_return:.5f}")
        st.sidebar.write(f"Standard Deviation: {std_dev:.5f}")
        st.sidebar.write(f"Last Price: ${last_price:.2f}")

        
        st.subheader(f"Historical Prices for {ticker}")
        st.line_chart(data[price_col])

       
        st.subheader("Monte Carlo Simulation Running...")
        np.random.seed(42)
        simulated_prices = np.zeros((num_days, num_simulations))
        simulated_prices[0] = last_price
        dt = 1  

        for t in range(1, num_days):
            random_shocks = np.random.randn(num_simulations)  # Z_t
            simulated_prices[t] = simulated_prices[t - 1] * np.exp(
                (mean_return - 0.5 * std_dev**2) * dt + std_dev * random_shocks
            )

        st.success("✅ Simulation complete!")

       
        fig = go.Figure()
        for i in range(min(300, num_simulations)):
            fig.add_trace(go.Scatter(
                y=simulated_prices[:, i],
                mode="lines",
                line=dict(width=0.8, color="rgba(0,100,200,0.15)"),
                showlegend=False
            ))
        fig.add_trace(go.Scatter(
            y=simulated_prices.mean(axis=1),
            mode="lines",
            line=dict(width=3, color="red"),
            name="Average Simulation"
        ))
        fig.update_layout(
            title=f"Monte Carlo Simulation for {ticker} ({num_simulations:,} runs)",
            xaxis_title="Days into Future",
            yaxis_title="Predicted Price ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

      
        final_prices = simulated_prices[-1]
        expected_price = np.mean(final_prices)
        lower_bound = np.percentile(final_prices, 5)
        upper_bound = np.percentile(final_prices, 95)

        st.subheader("Simulation Results (Confidence Intervals)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Price", f"${expected_price:.2f}")
        col2.metric("5% Confidence", f"${lower_bound:.2f}")
        col3.metric("95% Confidence", f"${upper_bound:.2f}")

        
        hist_fig = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=80)])
        hist_fig.update_layout(
            title=f"Distribution of Final Prices after {num_days} Days",
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            template="plotly_dark"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching ticker data: {e}")
