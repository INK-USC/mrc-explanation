## Source Code for "Teaching Machine Comprehension with Compositional Explanations"

### Parse SQuAD Explanations
```bash
PYTHONPATH='.' python parser/parse_squad_exps.py --verbose --save_ans_func
```

### Hard Match

```bash
PYTHONPATH='.' python parser/match_squad_hard.py --nproc 32 --verbose --save_matched
```