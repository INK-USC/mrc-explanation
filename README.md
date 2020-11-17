## Source Code for "Teaching Machine Comprehension with Compositional Explanations" (Findings of EMNLP 2020)

**TL;DR**: We **collect human explanations** that justifies their answering decisions when doing QA task; We **transform** these explanations **into executable “teacher” programs**; We use programs to **annotate unlabeled QA examples** and train a “student” QA model.

**Project homepage**: [https://github.com/INK-USC/mrc-explanation-project](https://github.com/INK-USC/mrc-explanation-project)


### Configure Environment
```bash
conda create -n mrc-explanation python=3.6.9
conda activate mrc-explanation
pip install torch==1.4.0 allennlp==0.9.0 nltk==3.4.5 pandas==0.25.3
```

Then navigate to nltk source code `nltk/parse/chart.py`, line 685, modify `function parse, change for edge in self.select(start=0, end=self._num_leaves,lhs=root):` to `for edge in self.select(start=0, end=self._num_leaves):`.

### Download Data
Please download pre-processed data and explanations from [here](https://drive.google.com/drive/folders/1Ho8FJrjaByq-6pSYJkajzk78QdriNM-p?usp=sharing). Please put the csv files at `./explanations` and json files at `./data/squad`.

### Example
This code snippet contains a minimal example that explains how an explanation is parsed, and how a constructed program is used to annotate new instances.
```bash
PYTHONPATH='.' python parser/example.py
```

### Parse SQuAD Explanations
```bash
PYTHONPATH='.' python parser/parse_squad_exps.py --verbose --save_ans_func
```

### Hard Match

```bash
PYTHONPATH='.' python parser/match_squad_hard.py --nproc 32 --verbose --save_matched
```