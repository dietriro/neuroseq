# Sequence Learning with Analog Neuromorphic Multi-Compartment Neurons and On-Chip Structural STDP

## 1 Code & Data

The original version of the code from the publication [[1]](#3-references) is located within the tagged version [shtm-backtracing_v1.0](https://github.com/dietriro/neuroseq/tree/shtm-backtracing_v1.0). While it should generally be possible to recreate the experimental results with the current version of the repository, we recommend using this version in case you would like to replicate the exact results.

You can download the code and data with:

```bash
# Preferred   - Clone only specific tag (tag_name), e.g. shtm-on-bss2, and only the state at that revision (saves time and space)
git clone --recurse-submodules --depth 1 --branch shtm-backtracing_v1.0 git@github.com:dietriro/neuroseq.git
# Only necessary if the submodule loading failed 
cd ./neuroseq/data/evaluation
git checkout shtm-backtracing_v1.0
```

Afterwards, you should be able to proceed as described in the main readme file with the evaluations.


## 2 Plots

The plots from the paper [[1]](#3-references) can be replicated using the respective scripts in the `plots/paper_shtm-backtracing` folder. 

## 3 References

---

[2]&nbsp;&nbsp;&nbsp; R. Dietrich, P. Spilger, E. Müller, J. Schemmel, and A. C. Knoll, Sequence Learning with Analog “[Neuromorphic
Multi-Compartment Neurons and On-Chip Structural STDP]()”, Lecture Notes in Computer Science, TBP Jan. 2025.

[2]&nbsp;&nbsp;&nbsp; Y. Bouhadjar, D. J. Wouters, M. Diesmann, and T. Tetzlaff, “[Sequence learning, prediction, and replay in networks of spiking neurons](doi.org/10.1371/journal.pcbi.1010233),” PLOS Computational Biology, Jun. 2022.

[3]&nbsp;&nbsp;&nbsp; C. Pehle et al., “[The BrainScaleS-2 Accelerated Neuromorphic System With Hybrid Plasticity](doi.org/10.3389/fnins.2022.795876),” Frontiers in Neuroscience, Feb. 2024.