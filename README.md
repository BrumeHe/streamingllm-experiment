# 实验：Attention sink 可视化 + StreamingLLM 式 KV 策略验证

模型：Qwen2.5-1.5B-Instruct（bf16，eager attention；fp16 会因激活值上溢产生 NaN，故用 bf16）
长文本：LongBench narrativeqa 第一篇文档（202039 字符）
计算资源: RTX 4060 laptop 8GB
主体部分代码: scripts/
结果: results/