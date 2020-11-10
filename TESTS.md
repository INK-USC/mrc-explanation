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