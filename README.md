# pairwise-imgkit

## 题面（可直接粘贴）

滤镜服务加滤镜链后出现三类事故：加错核就整条链失效却无法回退、流式处理与整图处理结果不一致、两张图并发跑同一条链时边框像素互相污染。imgkit/core.py 是核心，tests/test_core.py 15 个红。要修的是：空链返回原图副本；失败的 add 不改变链且可回退；apply_stream 与 apply 逐像素一致；并发与串行结果一致；核必须奇数方阵且和为 1，非法输入抛 ValueError。

## 运行方式

```bash
python -m pytest tests -q
```

初始结果应为 `15 failed`（全部因 `NotImplementedError` 失败，无 collection error）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
