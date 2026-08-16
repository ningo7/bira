# BFSRCNN 专用 Chisel 测试

本目录只包含 BFSRCNN 模型层、组合数据流、独立顶层和 trace 回放验证。通用 BIRA 模块测试位于 `src/test/scala/bira/` 根目录。

```bash
cd generators/bira/hw
sbt "testOnly bira.models.bfsrcnn.*"
```
