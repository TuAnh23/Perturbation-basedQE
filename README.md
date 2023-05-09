The repository contains the implementation for the paper: "Perturbation-based QE: An Explainable, Unsupervised Word-level Quality Estimation method for Blackbox Machine Translation"

## Installation
Run the commands in `env_installation.sh`.

## Reproducing experiments
### In-domain evaluation 
Hyperparameter tuning and evaluation on WMT21 QE shared task `en-de` and `en-zh` data
```bash
bash run_whole_pipeline.sh
```
After running the command, the performance in MCC score is at `analyse_output/WMT21_DA_test_{src_lang}2{tgt_lang}/collect_results.txt`

Evaluation on fully unsupervised data, i.e., `en-cs` and `en-ja`:
```bash
bash run_test_only.sh
```

### Out-of-domain evaluation 
Experiments on detecting wrong gender token output from MT systems on WinoMT data
```bash
bash run_gender_bias.sh ${MTmodel}
```

Experiments on detecting wrong WSD token output from MT systems on MuCoW data
```bash
bash run_WSD.sh ${MTmodel}
```

`MTmodel` can be `LLM`, i.e., large language model performing translation by prompting, or `qe_wmt21`, i.e., the in-domain MT models provided in the WMT21 QE shared task
