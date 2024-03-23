# Design of a cross-border ETF arbitrage strategy model



## Award results

The first prize of the 4th "Greater Bay Area Cup" Guangdong, Hong Kong and Macao Financial Mathematical Modeling Competition (Question A) in 2023, plus the Innovation Award Nomination Award

## Summary of entries

This study designs a set of efficient cross-border exchange-traded fund (ETF) trading strategies and explores cross-market arbitrage opportunities between cross-border ETFs and stock index futures. By constructing different mathematical models and strategies, this research aims to use price differences between different markets to provide scientific decision-making support for arbitrage trading.

### Model Explanation

- **Solution to Problem 1**: Based on the Engle-Granger cointegration test and error correction model, a time series prediction model was established to select the 10 ETFs most suitable for trading.
- **Solution to Problem 2**: Use the LSTM network and the time series prediction model of the fully connected layer, combined with indicators such as discount and premium rates, to select the 10 ETFs most suitable for cross-border arbitrage.
- **Solution to Problem 3**: Introduce market volatility index, establish a cross-market arbitrage model of stock index futures and cross-border ETFs, select the most suitable combination, and determine the weight of the combination.
- **Solution to Question 4**: Measure Hang Seng Technology ETF and cross-market arbitrage model to verify the effectiveness of the arbitrage model.

### Actual measurement report

The Hang Seng Technology ETF (513130) and the combination of different cross-border ETFs and stock index futures were analyzed to verify the applicability and robustness of the model, focusing on the transaction data in November 2023 for actual measurement training.

## Repository content

- `/code`: main code of all core mathematical models, based on Python 3.8 environment.
- `/sec2_process`: Actual measurement report test code, based on Python 3.8 environment.
- `/references`: Documents, competition questions and other reference documents cited during the design process.
- `/Contest_Paper_202311081259.pdf`: Model elaboration document.
- `/Test_Report_Final_202311171856.pdf`: actual test report document.
- `/Promotion_Manuscript.pdf`: defense presentation.

## Follow-up research results

The follow-up research results of this project have been published at the 2024 Guangdong-Hong Kong-Macao Greater Bay Area International Academic Conference on Digital Economy and Artificial Intelligence (DEAI2024). Paper citations will be published in subsequent updates.

## Copyright

- This project is open sourced under the GPL v3.0 license and aims to provide resources for the academic and research community.
- If you use this repository in research, please cite our paper.
