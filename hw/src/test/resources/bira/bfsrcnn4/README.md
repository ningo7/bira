# BFSRCNN 4×4 快速回归 fixture

这是 BFSRCNN 专用的仓库内置 fixture：输入为 4×4，完整执行程序员手写的14层 C 推理，输出为16×16。当前量化配置中 shrink3 和 expand 使用 W8，head、shrink2 和 final 使用 W16，shrink1 使用 W4。命令由通用 Runtime Planner 自动分配片上地址后生成。它供 `BfsrcnnTraceReplaySpec` 和 BFSRCNN Verilator harness 做快速、免生成回归，不是通用 BIRA 测试数据格式示例。

通用 fixture 二进制协议由软件侧 `simulation` 模块维护；模型的层名、阶段尺寸和黄金输出布局由对应模型 harness 解释。

该目录是受版本控制的冻结测试资源，不是构建输出。软件工具不会自动写入或覆盖这里；更新时应先在 `sw/build/` 生成候选 fixture、完成结果核对，再明确替换并提交本目录。
