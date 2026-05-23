\# v0.7 Extended Robustness Experiment Plan



\## Objective



The objective of this stage is to strengthen the experimental evidence for the proposed patch-based early-warning framework. The current results show stable performance across three random seeds, but additional robustness experiments are required to support a stronger manuscript.



\## Planned Experiments



\### 1. Additional random seeds



The current experiments used three random seeds. In this stage, the number of seeds will be increased to five:



\- 42

\- 123

\- 2026

\- 3407

\- 7777



This experiment is designed to evaluate whether the model performance remains stable under repeated random initialization and simulation sampling.



\### 2. Prediction horizon sensitivity



The current main setting uses a warning horizon of h = 30. Additional experiments will test different prediction horizons:



\- h = 20

\- h = 30

\- h = 40



This experiment will evaluate how early the model can provide reliable warnings before the critical transition.



\### 3. Observation noise robustness



To make the simulated setting closer to realistic monitoring data, observation noise will be added to the spatial infection fields. Candidate noise levels include:



\- no noise

\- low noise

\- medium noise

\- high noise



This experiment will test whether the model remains effective when the observed spatial signals are imperfect.



\### 4. Missing observation robustness



Real monitoring data may contain missing observations. This stage will test model performance under different missing-data ratios:



\- 0%

\- 10%

\- 20%

\- 30%



\### 5. Parameter-shift generalization



To evaluate whether the model only works under a fixed simulation setting, parameter-shift experiments will be introduced. The model will be trained under one range of SEIR parameters and tested under another range.



This experiment is important for demonstrating generalization beyond a single simulation configuration.



\## Expected Outputs



This stage should generate:



1\. Extended robustness result files.

2\. Summary tables for additional seeds.

3\. Summary tables for horizon sensitivity.

4\. Summary tables for noise robustness.

5\. Summary tables for missing observation robustness.

6\. Updated manuscript notes for the Results and Discussion sections.



\## Manuscript Relevance



The v0.7 experiments will strengthen the manuscript in three ways:



1\. They will show that the model is not dependent on a single random seed.

2\. They will test whether the warning performance is stable under different prediction horizons.

3\. They will make the simulation study more realistic by introducing noise and missing observations.



These experiments are necessary before claiming that the framework is suitable for a stronger SCI journal submission.

