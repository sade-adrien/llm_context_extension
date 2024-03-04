## LLM Context Extension

This project is an application of the YaRN method for extending the context window of a LLM. 

We extend the context of Mistral-7B in three steps:
* 1. Continuing the pre-training, with fixed-size 16k-token long examples (from RedPajama-2 dataset)
  2. Continuing the  pre-training, with long-examples with sizes following a 1/x-like distribution - supposed to mimic the real-life distribution (also from RedPajama-2)
  3. Fine-tuning for Q/A on long-context examples (from Long-Alpaca dataset)
