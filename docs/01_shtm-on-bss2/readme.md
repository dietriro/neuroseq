# Sequence Learning with Analog Neuromorphic Multi-Compartment Neurons and On-Chip Structural STDP

## 1 Code & Data

The original version of the code from the publication [[1]](#3-references) is located within the tagged version [shtm-on-bss2_v1.0](https://github.com/dietriro/neuroseq/tree/shtm-on-bss2_v1.0). This version is deprecated, since it still contains all the evaluation data directly with the repository. During the refactoring process, the data was moved to another repository ([neuroseq-evaluation](https://github.com/dietriro/neuroseq-evaluation)) and is now included as a submodule only. While it should generally be possible to replicate the experiments with the most current version of the _neuroseq_ package, we still created a tested version ([shtm-on-bss2_v2.0](https://github.com/dietriro/neuroseq/tree/shtm-on-bss2_v2.0)) after the refactoring process that is provably able to replicate the experiments and plots. You can check this version out using the method described in the main readme and its tag name.

We suggest to stick with the most current version of the repository or the refactored code version of the paper if possible. In case you would still like to run the exact code and experiments from the paper [[1]](#3-references), then we recommend to create a separate folder for the old version to avoid issues with the git-submodules system. In order to download and set up the original code and data run the following command in a terminal:

```bash
# Preferred   - Clone only specific tag (tag_name), e.g. shtm-on-bss2, and only the state at that revision (saves time and space)
git clone --depth 1 --branch shtm-on-bss2_v1.0 git@github.com:dietriro/neuroseq.git
```

Afterwards, you should be able to proceed as described in the main readme file with the evaluations. Just bear in mind that the names in this version are different from the ones in the current version.

## 2 Plots

Currently, the plots can be replicated using the plotting files in the `scripts` (old: `tests`) folder. We plan to add specific scripts to the `plots` folder in order to make the replication of figures easier.

## 3 References

---

[1]&nbsp;&nbsp;&nbsp; R. Dietrich, P. Spilger, E. Muller, J. Schemmel, and A. C. Knoll, “[Sequence Learning with Analog Neuromorphic Multi-Compartment Neurons and On-Chip Structural STDP](doi.org/),” in Lecture Notes of Computer Science, Springer Nature, TBP.

[2]&nbsp;&nbsp;&nbsp; Y. Bouhadjar, D. J. Wouters, M. Diesmann, and T. Tetzlaff, “[Sequence learning, prediction, and replay in networks of spiking neurons](doi.org/10.1371/journal.pcbi.1010233),” PLOS Computational Biology, Jun. 2022.

[3]&nbsp;&nbsp;&nbsp; C. Pehle et al., “[The BrainScaleS-2 Accelerated Neuromorphic System With Hybrid Plasticity](doi.org/10.3389/fnins.2022.795876),” Frontiers in Neuroscience, Feb. 2024.