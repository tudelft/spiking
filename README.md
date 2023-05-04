# tudelft/spiking

Collection of templates and examples for implementing spiking neural networks in plain PyTorch and JAX (coming).

There's many great SNN + PyTorch libraries out there (looking at you, [Norse](https://github.com/norse/norse) and [snnTorch](https://github.com/jeshraghian/snntorch)), but their feature-richness makes it more difficult to get a deep understanding of SNNs, while people who already have the understanding often end up modifying the libary to their needs. This repo tries to show that you're usually better off implementing your own spiking networks in PyTorch, because you actually only need a very thin layer of spiking sauce on top of PyTorch! Result: better understanding and more flexibility. Win-win!

Line count of `core/torch`: 477

If you find this repo useful, consider citing it or giving it a star!

```bibtex
@misc{mavlab2023spiking,
  author = {Hagenaars, J and Paredes-Vall\'{e}s, F},
  title = {{tudelft/spiking}},
  year = {2023},
  howpublished = {https://github.com/tudelft/spiking}
}
```