# multiagent-comparison

# Dataset

Download trialgpt's ground truths dataset from here (labelled by clinicians):
https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations/viewer/default/train
- I've renamed this from `train-00000-of-00001.parquet` to `ground_truths.parquet`
- Note that it includes trialgpt's results as well. We use `expert_eligibility` as ground truth only.

The SIGIR 2016 corpus, available at: https://data.csiro.au/collection/csiro:17152

To prepare the data, run:

```
uv run ./data_preprocessor.py
```

# Code

There are three main scripts which can be run as follows:

```
uv run .\main.py --llm openai --no-cache  # baselines
uv run .\reflection.py --llm openai  # generator-reflection pattern
uv run .\specialists.py --llm openai # hierarchical multi-agent pattern
```

We provide the `results/` that were used in the paper.

We also include a results_analysis.ipynb which displays results.
