# idrr

Code for **Learning Compact Representations via Intrinsic Dimension Regularization**, published in the Proceedings of Machine Learning Research (PMLR) through the GRaM workshop at ICLR 2026, Proceedings Track. Authors: Kaustubh Bukkapatnam, Laksh Patel, Soham Batra.

The whole thing is one Colab notebook, `IDRR_reproduction.ipynb`. It rebuilds every experiment in the paper from scratch: the method, the seven baselines, the four datasets, Tables 1 to 5, and all the figures. The original experiment code was lost, so this is a clean reimplementation written against the paper's own specification (Section 5 and Appendix C).

## The idea

Networks learn representations in far more dimensions than the task needs. IDRR puts a differentiable penalty on the soft effective rank of a batch of representations (the exponential of the Shannon entropy of the normalised singular values) and pushes it toward a data-driven target, sqrt((k - 1) times the effective rank of the input). The loss is two-sided so the network can't collapse either: a penalty above the target plus a lighter one below it.

The result in the paper is that IDRR combined with dropout matches dropout's accuracy while cutting representation dimensionality by 68 to 81 percent. On MNIST that's 96.5% accuracy at effective rank 12.6 against 96.6% at rank 39.7 for dropout alone.

## Running it

Open the notebook in Colab with a T4 runtime and run all. `RUN_MODE = "smoke"` checks the pipeline in about five minutes on one seed. `RUN_MODE = "full"` uses the paper's protocol: AdamW at 1e-3 with cosine annealing, batch 128, up to 500 epochs with early stopping at patience 15, five seeds (three for CIFAR-10), 10,000 train and 2,000 test examples per dataset. The MLP sweep is about two hours; the CNN sweep (Table 2) is another three to four and is gated by `RUN_CNN`.

Everything is written to `results/` as CSV and PNG. Set `SAVE_TO_DRIVE = True` to keep them across runtime restarts.

## What's implemented

Soft effective rank and the two-sided loss (Eq. 1 and 2), the data-driven target rank (Eq. 3), Standard, Weight Decay, Dropout, Jacobian regularisation (random-projection estimator), IDRR, IDRR-Adaptive with the cosine lambda schedule, and IDRR+Dropout. MLPs with hidden [256, 128] and a 64-d representation ([512, 256, 128] and 128-d for CIFAR-10), and the three-block CNNs from Appendix C. Paired t-tests by seed, PCA projections, singular value spectra, per-epoch dynamics, the lambda and target-rank ablations, and the bottleneck baselines.

## Two things to know

The synthetic dataset generator was lost with the code. The one here is an 8-d manifold embedded in R^100 with 5 classes, tuned to land in the same accuracy regime as the paper (about 70 to 77% for the baselines). Absolute ranks on that dataset can differ from Table 1 while the ordering of methods holds.

The original seeds and subset draws are gone too, so the MNIST, Fashion-MNIST and CIFAR-10 numbers will land close to the published tables rather than digit for digit.

## Citation

Bukkapatnam, K., Patel, L., and Batra, S. Learning Compact Representations via Intrinsic Dimension Regularization. GRaM Workshop at ICLR 2026, Proceedings of Machine Learning Research.

## License

MIT. Copyright 2026 Soham Batra.
