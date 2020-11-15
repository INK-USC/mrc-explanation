### Parser
```bash
python parser/parser.py 
```

### SQuAD reader
```bash
PYTHONPATH='.' python utils/squad_reader.py 
```

### Parse one explanation and validate on itself
```bash
PYTHONPATH='.' python parser/example.py
```

```bash
PYTHONPATH='.' python parser/parse_squad_exps.py --verbose --unit_test
```

### Hard match on dev
```bash
PYTHONPATH='.' python parser/match_squad_hard.py \
    --qa_file ./data/squad/dev_processed_dep.json \
    --out_file_hard ./data/squad_matched/hard_matched_dev.json \
    --out_file_abstain ./data/squad_matched/abstain_dev.json \
    --verbose \
    --nproc 20 \
    --save_matched;
```