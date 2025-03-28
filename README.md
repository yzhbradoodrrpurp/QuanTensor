# QuanTensor

## About

QuanTensor operates on two aspects, Entity and Action. Entity includes Data and Strategy and Action includes Backtest and Deploy. Entity is utilized by Action to perform the behaviors that you want. To be more precise, **QuanTensor is mainly composed of four parts: Data, Strategy, Backtest, Deploy**, as you can see there are four corresponding directories.

- [Data](Data): This directory takes charge of the relevant data of your concernd ticks which includes the following steps:
  1. Import your API of a certain platform and use it to send requests to that platform to obtain data.
  2. Data is stored in time to be analyzed by your customized strategy and deleted in time to free memory when it's outdated and unwanted.

- [Strategy:](Strategy) This directory is a strategy pool in which you will implement your own strategies to analyze the data thus performing buy and sell and other more complicated actions. To be more specific, it includes steps as follows:
  1. Take in the data which is stored under [Data](Data) and analyze it as how you define the buy and sell signals.
  2. The signals are then utilized to do more downstream tasks such as how many to buy / sell and whether to go all in / cash out.
- [Backtest](Backtest): After you have developed your own strategies, you prefer to run them on the previous data to evaluate their performance rather than directly deploy them. Hence this directory is intended to do backtesting operations on your strategies and evaluate them by a few indices like Return on Investment, Annualized Return, Max Drawdown, Sharpe Ratio, Alpha, Bata, Win rate, Profit Loss Ratio and so on. After you have gained an insight into how your strategies perform, you then decide whether you put them into use.
- [Deploy](Deploy): This is where you officially deploy your high-performance strategies and put them into use. You will import the API of a certain platform as well to execute buy and sell behaviors after the corresponding buy and sell signals are detected by your strategies. Aside from that, the profit / loss is also required to demonstrate in real time.

