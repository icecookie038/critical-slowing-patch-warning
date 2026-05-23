\# v0.4 Robustness Experiment Log



\## Objective



This stage is designed to summarize experimental results under different random seeds, input modes, and model settings, in order to evaluate the stability and reproducibility of the current early-warning model for critical slowing down.



\## Completed Versions



\- v0.1: Original working version

\- v0.2: Repository structure reorganization

\- v0.3: README and requirements update

\- v0.4: Robustness experiments and result summary



\## Plan for This Stage



1\. Summarize the results of the deep learning model under different random seeds and input modes, including full, image-only, and patch-only settings.

2\. Summarize the results of traditional machine learning baseline models.

3\. Calculate the mean and standard deviation of AUC, AUPRC, ACC, F1, Precision, Recall, and lead time.

4\. Generate summary tables that can be directly used in the Results section of the manuscript.

5\. Compare the full, image-only, and patch-only input modes to evaluate whether patch-level indicators provide stable performance improvements.



\## Key Metrics



\- AUC

\- AUPRC

\- F1 score

\- Precision

\- Recall

\- Mean lead time

\- Median lead time

\- False positives

\- False negatives



\## Preliminary Conclusion



The current model shows stable early-warning performance on simulated spatial SEIR data. Across different random seeds, the AUC is generally maintained around 0.86 to 0.91, and the mean successful warning lead time is approximately 14 time steps. The full-input mode usually achieves stronger overall performance, suggesting that spatial image information and patch-level statistical indicators provide complementary information for early-warning prediction.

