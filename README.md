# 智训引擎 RC04

RC04 是平台 v2 的候选版本。新版修复两条 B 路径与一条 C 路径上的 `ValueError`。A、B、C 均已从新归档重载后完成本地十轮续训。

[RC04 技术报告](training-records/rc04/technical-report-rc04.md) · [逐轮验收](training-records/rc04/verified.json) · [镜像归档回执](training-records/rc04/archive-load.json)

## RC03 基础方案与实验记录

以下归档标识和实验记录对应 RC03。当前分支的源码登记对应 RC04。

单一 Linux/amd64 镜像通过赛事 SDK 1.0.4 运行 A 气体分类、B 轴承寿命和 C 发动机寿命。平台绑定模型为 `Qwen/Qwen3.5-4B`。

- [准备赛事构建依赖](source/PREPARE_OFFICIAL.md)
- [源码与构建命令](source/README.md)
- [训练记录和证据索引](training-records/README.md)
- [四页技术报告](training-records/technical-report-rc03.pdf)
- [90 秒演示](training-records/demo-rc03.mp4)

镜像标识为 `sha256:0ebd6a1a6db8b907a26c030eab731c8ed140a64d88e952b3bd6ea61b7a694bde`。上传归档 SHA-256 为 `f17be9fb484da09d6cfdb36ef6e3c06a57d18db9650067964a1f64d71dcc8dde`。构建和试验均沿用赛事发布的基线与官方数据包。

三包已从实际归档重载完成各十轮续训，并完成独立预测。基础模型、公开验证指标、按组误差、受控恢复和本地臂的边界详见证据索引。平台隐藏集成绩待平台评测。

[代码使用说明](LICENSE) · [数据引用](training-records/DATA_SOURCES.md)
