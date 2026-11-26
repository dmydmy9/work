# 准备赛事构建依赖

从 [赛事训练起点](https://rsi-index.115.zone/#baselines) 下载并解压 SDK 1.0.4 和官方 C 包。从仓库根目录调用官方依赖恢复脚本，分别传入 SDK 和 C 包的解压目录。脚本按 `SOURCE_HASHES.json` 核验内容并复制构建所需文件。

```text
python source/restore_official_context.py /path/to/official-sdk /path/to/official-pack-C
```

随后进入 `source/` 按 [构建说明](README.md) 构建。
