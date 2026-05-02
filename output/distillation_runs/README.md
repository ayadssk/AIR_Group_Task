# Different Distillation runs

- Distilling Qwen2.5-7B into Qwen2.5-0.5B
- Distilling Qwen2.5-1.5B into Qwen2.5-0.5B
- Distilling llama3.2-3B into llama3.2-1B
- Distilling llama3.2-1B into roBERTa-large

All configurations have also been tried with different settings i.e.
- Lora rank and alpha values
- Distillation loss (kd_loss): binary_cross_entropy_with_logits, 
- Batch sizes (2-8) and num epochs (3-5)
- larger model tried with and without quantization (mostly with quant as otherwise they would not fit in memory in jupyterhub)
- with and without **evidence**

## Conclusion
None of the results have even really beat the provided baseline
