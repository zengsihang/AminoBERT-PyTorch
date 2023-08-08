# AminoBERT-PyTorch
Unofficial PyTorch checkpoint of AminoBERT in RGN-2 by transferring weight from original TensorFlow checkpoint.

**Note**: We have checked the consistency of performance on our tasks, but haven't checked it on the benchmarks.

## Usage
- Download checkpoint: https://www.dropbox.com/sh/bp1fggxyxpx4zml/AAAzvY95L4ltUTN8NcEJD64Da?dl=0
- Load checkpoint:
```python
from transformers import BertModel
model = BertModel.from_pretrained("./aminobert")
```
- Load Tokenizer:
```python
from tokenization import FullTokenizer
tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')
```
- Generate embeddings from fasta
Check `get_aminobert_embedding.py` for more detail.

## TODO
- Performance comparison on benchmarks
- Codes for transferring weights 

## Acknowledgement
Some of the codes and TensorFlow checkpoint are obtained from https://github.com/aqlaboratory/rgn2.

## Reference
https://www.nature.com/articles/s41587-022-01432-w.epdf?sharing_token=h5IMCZkXufTQSaMPi1i2hdRgN0jAjWel9jnR3ZoTv0M9YquEWqhqqdSBTuCn4caitD2yaZSXQThLcP0nQBKyAZ1GDYGAyCORyMcRGFWfcyoVobvnb13zyLnPDFP4g6fGWqQpwTnYd8UT1oHDnEcUlZ332ExK7H2Dkes7acsd89E%3D
