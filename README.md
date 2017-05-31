SEQ_PREDICT
===========

Copyright (c) 2016 Michael Bowling <mbowling@ualberta.ca>
See LICENSE for information on using or distributing the code.

Module for compression and sequence prediction using Bayesian prediction models.

Implements the following Bayesian models:
- KT 
- SAD 
- CTW
- CTS
- PTW
- An early version of Forget-Me-Not
- Model combinations: Averaging, Factoring

Most of the implementations are in pure python.  CTW and CTS with integer k-ary
alphabets (i.e., 0, 1, ..., k) using KT estimators has a fast c-based
implementation.

In addition, it includes an arithmetic binary coder.

Using it for Compression
------------------------

Executing the module on the command line invokes a compression tool that allows
you to select one of the implemented models to use with an arithmetic encoder
for compression.

```bash
python -m seq_predict compress FILE.TXT FILE.TXT.compressed -m CTW
```

Using it for Prediction 
-----------------------

The model classes can be imported as below.

```python
from compress.models import PTW
from compress.fast import CTS_KT
```

The models adhere to the following API:
- `predict(symbol, history)`
- `log_predict(symbol, history)`
- `log_predict_seq(symbols, history)`
- `update(symbol, history)`
- `update_seq(symbols, history)`
- `copy()`

The history is optional for models that don't look at history (e.g., KT
estimators).  The `*_seq` functions only make sense if appending symbols are how
the history is updated.
