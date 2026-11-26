# 智训引擎 Harness RC04（v2 试运行候选）

## v1 故障与修复

v1 平台试运行 `fddf03d2…` 在训练优化阶段显示 `ValueError`。页面最后显示 A 6/10、B 7/10、C 4/10，公开详情未提供首个失败包、轮次或训练栈。

| 用例 | v1 在本地 SDK 触发的错误 | RC04 行为 |
| --- | --- | --- |
| B 候选评分缺少可用 RMSE | `host did not provide finite nonnegative B RMSE` | 保留父模型，记录评分缺失，不估算候选增益 |
| B 拟合轴承的快照不足 | `need at least two fitting bearings with six snapshots` | 跳过轴承迁移配方，以基础续训配方参与宿主评分 |
| C 训练样本 ID 被匿名化 | `invalid C training identity 'opaque-fit-000000'` | 禁用依赖发动机分组的配方，继续按特征续训，保留已知重叠检查 |

用 v1 原归档与 RC04 运行相同三个受控宿主用例，RC04 均完成轮次并返回检查点决策。未处理异常会将包、轮次和异常类型写入训练 stderr。

## 新归档本地验收

重载 Docker 归档所对应的 Linux/amd64 镜像，经赛方 SDK 1.0.4 从官方初始检查点实际续训。每轮的规划上下文逐条与原始 PyTRiO `Qwen/Qwen3.5-4B` 回执匹配，随后重放规划回复。验收逐轮核对父检查点、实训步、评分及最终检查点。三包末轮检查点和常规验证指标与 v1 本地跑法一致。

| 包 | 本地指标 | 官方初始检查点 | 十轮均值 | 第十轮 | 末轮相对改善 | 轮次 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Macro-F1 | 0.699991 | 0.721553 | 0.721553 | +3.08% | 10/10 |
| B | RMSE（分钟） | 245.149294 | 230.566110 | 230.328294 | +6.05% | 10/10 |
| C | RMSE（循环） | 52.868573 | 49.115207 | 49.115207 | +7.10% | 10/10 |

A 的 Macro-F1 越高越好，B、C 的 RMSE 越低越好。B 另用 1,679 条官方公开验证行运行独立预测，RMSE 为 588.949982 分钟，与 v1 同一公开验证投影的逐行预测一致。表中的 B 数值来自 1,062 行的开发验证投影。本页表格为本地评分，平台隐藏集结果由评测平台发布。

[RC04 逐轮回执](https://github.com/dmydmy9/work/blob/rc04/training-records/rc04/verified.json) · [三项故障回归](https://github.com/dmydmy9/work/blob/rc04/training-records/rc04/valueerror-regression.json) · [基础方案与评测说明](https://github.com/dmydmy9/work/blob/rc04/training-records/technical-report-rc03.pdf) · [原版 SwanLab 公开复训](https://swanlab.cn/@dmydmy9/zhixun-engine-rc03/v1/iowsz6/overview)

归档 SHA-256 为 `4deee27e6b54f47cdb79e97dbf85bcf76b90bc94af212f82890bacf3221fe655`。
