# ntsne.py # 

a python3 numpy wrapper for `bh_tsne` ([https://github.com/lvdmaaten/bhtsne](https://github.com/lvdmaaten/bhtsne))

## Usage ##
`ntsne` calls [bh_tsne](https://github.com/lvdmaaten/bhtsne) via python's subprocess library.
If `bh_tsne` is not on the system path or in the current working directory, `ntsne` will attempt to clone and compile the `bhtsne` binary under `~/.ntsne`.

`ntsne.best_tsne` will run `bh_tsne` multiple times with the same parameters, returning the t-SNE results with the lowest KL-divergence.

```python
import ntsne
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X_tsne = ntsne.tsne(digits.data, perplexity=30, theta=0.5)

plt.scatter(X_tsne[:,0], X_tsne[:,1], c=digits.target)
plt.show()
```