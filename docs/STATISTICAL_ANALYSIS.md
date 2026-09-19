# Statistical analysis

Analysis is performed on paired differences keyed by dataset, architecture, seed and condition. Outputs include mean, median, standard deviation, bootstrap 95% CI, paired t-test, Wilcoxon signed-rank, Shapiro-Wilk diagnostic and Cohen's dz. Bonferroni is the primary multiplicity correction; Holm and Benjamini-Hochberg are sensitivities.

A meaningful result must pass the corrected statistical criterion, the preregistered practical-effect threshold and a non-trivial difference in original units. An optional TOST for Macro-F1 similarity uses the configured margin only after that margin is finalized before official analysis.
