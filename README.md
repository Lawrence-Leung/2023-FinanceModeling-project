# 一种跨境ETF套利策略模型设计

## 获奖结果

2023年第四届“大湾区杯”粤港澳金融数学建模竞赛（A题）一等奖，外加创新奖提名奖

## 参赛作品摘要

本研究设计了一套高效的跨境交易所交易基金（ETF）交易策略，并探索了跨境ETF与股指期货之间的跨市场套利机会。通过构建不同的数学模型和策略，本研究旨在利用不同市场间的价格差异，为套利交易提供科学的决策支持。

### 模型阐述

- **问题一的解决**：基于Engle-Granger协整检验与误差修正模型，建立了时间序列预测模型，筛选出最适合交易的10只ETF。
- **问题二的解决**：利用LSTM网络和全连接层的时间序列预测模型，结合折溢价率等指标，选出最适合进行跨境套利的10只ETF。
- **问题三的解决**：引入市场波动指数，建立股指期货与跨境ETF的跨市场套利模型，选出最适合的组合，并进行组合权重的确定。
- **问题四的解决**：实测恒生科技ETF及跨市场套利模型，验证套利模型的有效性。

### 实测报告

分析了恒生科技ETF（513130）及不同跨境ETF与股指期货的组合，验证模型的适用性与鲁棒性，聚焦于2023年11月的交易数据进行实测训练。

## 仓库内容

- `/code`：模型主要代码，基于Python 3.8环境。
- `/sec2_process`：实测报告测试代码，基于Python 3.8环境。
- `/references`：参与设计过程中所引用的文件、竞赛赛题等参考文献文档。
- `/Contest_Paper_202311081259.pdf`：模型阐述文档。
- `/Test_Report_Final_202311171856.pdf`：实测报告文档。
- `/Promotion_Manuscript.pdf`：答辩演示文稿。

## 后续研究成果

本项目的后续研究成果已发表于2024年粤港澳大湾区数字经济与人工智能国际学术会议(DEAI2024)。论文引用将在后续更新中发布。

## 版权

- 本项目以GPL v3.0协议开源，旨在为学术和研究社区提供资源。
- 如在研究中使用本仓库，请引用我们的论文。
