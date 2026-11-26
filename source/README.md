# 智训引擎合并镜像源码

从本目录执行下列命令构建 SDK 1.0.4 的单一 linux/amd64 镜像。

```text
docker build --platform linux/amd64 --provenance=false -f Dockerfile.city --build-context contest_sdk=sdk --build-context baselines=baselines --build-context frozen_a=frozen-a -t zhixun-harness:city .
docker run --rm --network none zhixun-harness:city describe
docker save -o harness-city.tar zhixun-harness:city
docker load -i harness-city.tar
```

由赛方挂载训练投影、选定父检查点和此前的轮次状态。赛方使用宿主 RPC 处理模型调用和计量。包 A、B、C 通过 `school_city_submission.py` 分发，产物名称为 `checkpoint.npz`。镜像内预测入口为 `python3 /app/harness.py predict`，逐行读写。

`SOURCE_HASHES.json` 登记文件 SHA-256 和使用这些文件核对的镜像身份。基线实现及 SDK 来自赛方发布包。运行依赖为 Python 3.11、NumPy 1.26.4 和随包 SDK。NumPy 的许可证随 Python 发行包保留。
