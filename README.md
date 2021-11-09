
---

<div align="center">

# Deep Consensus Finding

<!-- [![SINGA](https://img.shields.io/badge/SINGA-803300?logoColor=white)](https://singa.apache.org/)
![coverage](https://img.shields.io/badge/coverage-25%25-yellowgreen)
![license](https://img.shields.io/badge/license-Apache-green) -->

</div>

## Description

Deep neural network to find the consensus of clustering in DNA storage system

## How to run

Install dependencies

```bash
pip install -r requirements.txt
```

Build dataset
```bash
cd src
python generate_DNA.py
cd ..
```

Run scripts

```bash
python -m src.main
python -m src.inference
```

## Contribution

See [How-to-Contribute](contributing.md)
