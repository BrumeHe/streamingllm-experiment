# PPL comparison: KV cache strategies

- model: Qwen2.5-1.5B-Instruct (bf16, eager attention)
- text source: LongBench narrativeqa doc (len=202039 chars, via hf-mirror.com data.zip)
- tokens scored: 12276 (first 12288 tokens of the text)
- chunk: 1024, window: 1024, sink tokens: 4

| setting | description | PPL |
|---|---|---|
| full | full KV (baseline) | 4.4726 |
| window | sliding window only | 8.5696 |
| sink_window | sink (first 4) + sliding window | 4.7126 |
