# Local Graph-based Dictionary Expansion (LGDE)
Python code for the paper "LGDE: Local Graph-based Dictionary Expansion" by Dominik J Schindler, Sneha Jha, Xixuan Zhang, Kilian Buehling, Annett Heft and Mauricio Barahona: http://arxiv.org/abs/2405.07764 

## Installation
Clone the repository and open the folder in your terminal. 

```bash
$ git clone https://github.com/barahona-research-group/LGDE.git
$ cd LGDE/
```

Then, to install the package with ``pip``, execute the following command:

```bash
$ pip install .
```

## Using the code
To use *LGDE* we require a list of seed keywords `seed_dict`, a list of all candidate keywords `word_list` (for example from a domain-specific corpus) and their word embeddings `word_vecs`. We can then initialise a new *LGDE* object and expand the seed dictionary.

```python
from lgde import LGDE

# expand seed dictionary using LGDE method
lgde = LGDE(seed_dict,word_list,word_vecs)
lgde.expand(k=5,t=1)

# the discovered keywords are stored in a new attribute
print(lgde.discovered_dict_)
```

To discover new keywords we first construct a semantic similarity graph using CkNN [1] and then compute the semantic community of each seed keyword using fast local community detection with Severability [2]. The parameter $k$ determines the graph density and $t$ the size of the semantic communities. See documentation for more details and additional functionality for dictionary evaluation and plotting.

## Experiments
Our experiments of *LGDE* applied to a corpus of hate speech-related communication on Reddit and Gab can be found in the `experiments/redgab` directory. Our additional experiment of *LGDE* applied to a corpus of conspiracy-related communication on 4chan can be found in the `experiments/4chan` directory. 

## Cite
Please cite our paper if you use our code or data in your own work:

```
@article{schindlerLGDELocalGraphbased2024,
  author = {Schindler, Dominik J. and Jha, Sneha and Zhang, Xixuan and Buehling, Kilian and Heft, Annett and Barahona, Mauricio},
  title = {LGDE: Local Graph-based Dictionary Expansion},
  publisher = {arXiv},
  year = {2024},
  doi = {10.48550/arXiv.2405.07764},
  url = {http://arxiv.org/abs/2405.07764},
}
```

## References
[1] T. Berry and T. Sauer, 'Consistent manifold representation for topological data analysis', *Foundations of Data Science*, vol. 1, no. 1, p. 1-38, Feb. 2019, doi: 10.3934/fods.2019001.

[2] Y. Yu William, D. Jean-Charles, S. Yaliraki and M. Barahona, 'Severability of mesoscale components and local time scales in dynamical networks', arXiv: 2006.02972, Jun. 2020, doi: 10.48550/arXiv.2006.02972

## Licence
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.