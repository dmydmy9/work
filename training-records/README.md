# 智训引擎 RC03 证据索引

| 评分项 | 证据 |
| --- | --- |
| 平均性能 | [三包十轮逐轮对照](verified.json)，[B 完整公开复核](verify-b-public/score.json) |
| 最差阶段与恢复 | [C 第 6 轮退化候选与保留父模型](verify-c-dynamic/round-06/record.json)，[A 受控异常回执](post-load-regression.xml) |
| 稳定性 | [A 历史三种子](history/a-paired.json)，[B 种子 7](history/b-bearing-seed7.json)、[种子 2026](history/b-bearing-seed2026.json)，[C 整台发动机留出](history/c-held-engines.json) |
| 资源效率 | [逐轮实际训练、复用、CPU、优化步](verified.json)，[历史 PyTRiO 计量](history/token-usage.json) |
| 方法设计 | [历史可见性对照](history/a-probe-02.json)、[第 4 轮](history/a-probe-04.json)、[第 6 轮](history/a-probe-06.json) |
| 实验分析 | [A 切分登记](history/a-protocol.json)，[B 逐轴承及近失效分析](b-review.md)，[C 工况残差失败](history/c-regime-rejection.json) |
| 安全及工业落地 | [Docker 归档重载](archive-load.json)，[归档对象校验](archive-manifest.json)，[父子检查点链](verified.json) |
| 报告与答辩 | [四页技术报告](technical-report-rc03.pdf)，[演示视频](demo-rc03.mp4) |

逐行预测随各轨迹保存。A 预测含样本 ID。B 和 C 预测按相应公开投影原行序记录，决策记录保存预测或模型摘要。正式平台隐藏集成绩仍待平台评测。

[整轴承与整工况筛选](history/b-city-route-screen.json) · [C 核校准三种子](history/c-kernel-screen.json) · [B 真实采样](verify-b-dynamic/native-usage.json) · [C 真实采样](verify-c-dynamic/native-usage.json)

[C 投影变化后同轮恢复](c-projection-shift/report.json) · [C 同预算的短试验](history/c-five-epoch-pilot.json) · [分组保护独立复核](history/c-guarded-pilot.json)

[RC03 三包真实历史消融](history/rc03-history-ablation.json) · [A 同状态逐轮配对](history/rc03-a-same-state.json) · [A 预登记冻结复核](verify-a-review/report.json)

## 从官方标签复算

在已解压的证据目录运行以下命令。官方数据目录应含 A、B、C 三个包。脚本使用 Python 标准库，以对应公开投影的原顺序读取标签。

```text
python verify_scores.py . /path/to/official-packs
```

复算六条十轮部署轨迹、A 与 B 冻结复核，以及 73 条候选试验的主指标。 [复算脚本](verify_scores.py) · [源码全新构建与逐行对照](source-rebuild.json)

[全部预门控候选预测索引](pre-gate/manifest.json) · [平台镜像状态](platform-status.json)

## SwanLab 本地复训

从官方父检查点在 RC03 镜像中实际续训 A、B、C 各十轮。规划器重放原始上下文逐字匹配的 PyTRiO 4B 回执。此记录给出实际训练步数和逐轮指标。 [复训回执](swanlab-retrain.json)

三个实验已经同步，并通过未登录访问验证。 [SwanLab 公开看板](https://swanlab.cn/@dmydmy9/zhixun-engine-rc03/v1/iowsz6/overview)

## 数据来源与引用

本项目使用组委会提供的 A、B、C 数据投影和官方初始检查点。仓库提供评分脚本、样本 ID 和预测记录，原始数据由组委会或数据提供方下载。

| 数据 | 原始来源与引用 | 数据使用条款 |
| --- | --- | --- |
| A 气体识别 | Alexander Vergara（2012），Gas Sensor Array Drift Dataset，DOI `10.24432/C5RP6W`。 [UCI 数据页](https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset) | UCI 标注 CC BY 4.0。 [许可](https://creativecommons.org/licenses/by/4.0/) |
| B 轴承寿命 | 西安交通大学与 Changxing Sumyoung Technology 发布 XJTU-SY Bearing Datasets。Wang、Lei、Li 和 Li（2020），DOI `10.1109/TR.2018.2882682`。 [作者数据页](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) · [论文](https://doi.org/10.1109/TR.2018.2882682) | 作者公开数据供轴承寿命预测验证，要求引用上述论文。 |
| C 发动机寿命 | NASA Dashlink 发布 CMAPSS Jet Engine Simulated Data。Saxena、Goebel、Simon 和 Eklund（2008），DOI `10.1109/PHM.2008.4711414`。 [NASA 数据页](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) · [论文](https://doi.org/10.1109/PHM.2008.4711414) | NASA 数据目录未指定数据许可证。 |
