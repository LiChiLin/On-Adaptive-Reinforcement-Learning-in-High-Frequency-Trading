---

# Adaptive Reinforcement Learning in High-Frequency Trading

**Author**: Chi-Lin Li

## Project Overview

This project presents a novel adaptive reinforcement learning framework applied to high-frequency trading (HFT). We integrate the **Double Q-Network (DQN)** with the **Gated Recurrent Unit (GRU)** for data encoding. The model adapts to changing market conditions through a dynamic sliding window algorithm, using the **Sortino ratio** as a key metric to guide trading decisions. Empirical results show the effectiveness of the proposed approach, particularly in managing turnover and transaction costs while achieving superior risk-adjusted returns.

### Key Features:
- **Adaptive Reinforcement Learning**: Integration of DQN and GRU models for better decision-making in fast-paced trading environments.
- **Sliding Window Algorithm**: Dynamic model updating with a time-varying approach, ensuring adaptability to evolving market data.
- **Risk Management**: Use of the Sortino ratio as the reward function, emphasizing risk-adjusted returns with a focus on downside risk.
- **Empirical Studies**: Tested on high-frequency data of major stocks (AAPL, JPM, and Sony) from September 2023 to October 2023, incorporating realistic transaction costs.

## Methodologies

### 1. Preprocessing
- **Data**: High-frequency 5-second interval data for major stocks.
- **Feature Engineering**: Applied transformations like logarithmic and Box-Cox transformations to stabilize variance and enhance the data’s normality.

### 2. GRU-DQN Model
- **GRU Integration**: Captures temporal dependencies in the data for accurate market prediction.
- **Double Q-Network**: Handles high-dimensional, sequential financial data with experience replay and fixed Q-targets to improve stability and convergence.
- **Sliding Window Algorithm**: Dynamically updates the model with new data, adapting to market shifts and maintaining relevance.

### 3. Reward Function
- **Sortino Ratio**: A risk-adjusted measure focusing on downside risk to optimize trading strategies and enhance returns in volatile environments.

## Empirical Results

### AAPL Performance:
- **Average Return**: 7.51%
- **Sharpe Ratio**: 2.26 (highest)
- **Maximum Drawdown**: 11.19%

Compared with traditional strategies like Buy and Hold, Elastic Net, and machine learning models like XGBoost and LightGBM, the **Adaptive GRU-DQN Model** consistently outperformed in terms of risk-adjusted returns, even with realistic transaction costs factored in.

### JPM and Sony Performance:
- **JPM Sharpe Ratio**: 1.28 (highest)
- **Sony Maximum Drawdown**: 8.60%

The model demonstrated robust performance across different equities, highlighting its adaptability to varying market conditions.

## Conclusion

The adaptive GRU-DQN model proves to be a powerful tool for high-frequency trading, outperforming traditional models in terms of both returns and risk management. By incorporating the Sortino ratio and sliding window techniques, the model can dynamically adapt to market fluctuations, providing superior risk-adjusted returns.

Future work will explore even higher-frequency data (milliseconds) and broaden the range of financial instruments, including global stocks, commodities, and foreign exchange. Additionally, investigating newer reinforcement learning algorithms like **Proximal Policy Optimization (PPO)** and **Trust Region Policy Optimization (TRPO)** can further enhance model stability and decision-making.

## Installation

To run this project, you'll need the following dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

## Usage

1. **Preprocess Data**: Prepare high-frequency stock data, applying necessary transformations.
2. **Train GRU-DQN Model**: Use the provided scripts to train the adaptive reinforcement learning model.
3. **Model Evaluation**: Evaluate the model’s performance using metrics like Sharpe ratio and maximum drawdown.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Chi-Lin Li contributed 100% to this project.

---
