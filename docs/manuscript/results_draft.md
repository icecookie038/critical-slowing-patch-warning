\# Results Draft



\## 3. Results



\### 3.1 Overall predictive performance



The proposed early-warning framework showed stable predictive performance across different random seeds in the simulated spatial SEIR system. Based on three independent random seeds, the deep learning model achieved consistently high discrimination ability, with the AUC values remaining around 0.86--0.91 across different input settings. This indicates that the model was able to distinguish high-risk samples approaching a critical transition from low-risk samples before the transition occurred.



Among the three input settings, the full-input model achieved an average AUC of approximately 0.885 and an average AUPRC of approximately 0.856. The image-only model achieved an average AUC of approximately 0.869 and an average AUPRC of approximately 0.831. The patch-only model achieved an average AUC of approximately 0.886 and an average AUPRC of approximately 0.853. These results suggest that both spatial image information and patch-level statistical indicators contain useful early-warning signals.



In terms of classification performance, the full-input model achieved an average F1 score of approximately 0.773, while the patch-only model achieved a similar average F1 score of approximately 0.775. The image-only model showed slightly lower but still competitive performance. These results indicate that patch-level indicators can provide predictive information comparable to the full model under the current simulation setting.



The successful warning lead time was also stable across different model settings. The mean lead time was approximately 14 time steps for the full, image-only, and patch-only models. This suggests that the model was not only able to classify risk states, but also able to provide warnings before the simulated critical transition occurred.



Table 1 summarizes the robustness comparison across input modes and baseline models.



\### 3.2 Ablation study of input modes



To evaluate the contribution of different input components, we compared three input modes: full input, image-only input, and patch-only input. The full-input mode combines spatial image information and patch-level statistical indicators. The image-only mode uses only spatial image information, while the patch-only mode uses only patch-level statistical features.



The image-only model achieved reasonable predictive performance, indicating that the spatial distribution of infection states contains useful information related to critical slowing down. However, its performance was slightly lower than that of the full-input and patch-only models in terms of AUC, AUPRC, and F1 score.



The patch-only model achieved performance comparable to, and in some metrics slightly better than, the full-input model. This result suggests that patch-level statistical indicators capture important spatial early-warning information. In particular, the strong performance of the patch-only model implies that the spatial organization of infected patches may provide a compact and informative representation of the system state near a critical transition.



Overall, the ablation results support the hypothesis that patch-level indicators are not merely auxiliary features, but contain meaningful early-warning signals for detecting the approach to critical slowing down in spatial epidemic systems.



\### 3.3 Comparison with traditional machine learning baselines



To further evaluate whether the predictive performance was specific to the deep learning architecture, we compared the proposed model with several traditional machine learning baselines using patch-level statistical features. The baseline models included ExtraTrees, Random Forest, Histogram Gradient Boosting, Logistic Regression, and a multilayer perceptron.



Among the baseline models, tree-based methods achieved strong performance. ExtraTrees and Random Forest achieved average AUC values of approximately 0.853 and 0.852, respectively. Histogram Gradient Boosting and Logistic Regression also achieved average AUC values above 0.84. These results indicate that the patch-level statistical features are intrinsically informative and can support early-warning prediction even without a deep learning architecture.



However, the deep learning models generally achieved higher AUC and AUPRC values than most baseline models. In particular, the full-input and patch-only deep models achieved average AUC values of approximately 0.885 and 0.886, respectively. This suggests that the proposed deep learning framework can further improve predictive performance by learning nonlinear patterns from spatial and patch-level representations.



\### 3.4 Robustness across random seeds



The experiments were repeated under three random seeds to evaluate the robustness of the results. Across these seeds, the main performance metrics showed relatively small variation. The AUC values of the deep learning models remained consistently high, and the mean warning lead time remained close to 14 time steps.



This robustness analysis suggests that the observed early-warning performance was not caused by a single favorable random initialization or a specific random simulation split. Instead, the results indicate that the proposed framework can provide stable prediction performance under repeated experimental settings.



\### 3.5 Lead-time analysis



In early-warning tasks, classification accuracy alone is not sufficient. A useful warning model should also provide predictions sufficiently before the critical transition. Therefore, we further evaluated the successful warning lead time for true positive samples.



The mean successful warning lead time was approximately 14 time steps across the main input modes. The median lead time was around 13--14 time steps, with the interquartile range generally ranging from approximately 7 to 21 time steps. These results indicate that the model can provide warnings well before the simulated transition point, rather than only detecting the transition after it has already occurred.



This lead-time result is important because it shows that the proposed method has potential practical value as an early-warning tool. In a real monitoring scenario, such a lead time could provide a time window for intervention before the system shifts into a high-risk state.



\## 4. Preliminary Interpretation



The current results provide three main findings. First, the proposed framework can predict the risk of approaching a critical transition in a simulated spatial SEIR system with stable performance. Second, patch-level statistical indicators contain strong early-warning information and can achieve performance comparable to the full-input model. Third, the prediction performance is robust across random seeds and is supported by both deep learning models and traditional machine learning baselines.



These findings support the feasibility of using patch-based spatial indicators for early warning of critical slowing down. However, the current results are still based on simulated data. To further strengthen the study for journal submission, additional experiments should be conducted under broader parameter settings, different noise levels, and more realistic observation conditions. A real or semi-real case study would further improve the external validity of the proposed method.



\## 5. Next Experimental Steps



The next stage should focus on strengthening the evidence for robustness and practical relevance. Specifically, future experiments should include:



1\. Increasing the number of random seeds from three to at least five.

2\. Testing the model under different SEIR parameter settings.

3\. Adding observational noise, missing observations, and delayed reporting.

4\. Evaluating model generalization under parameter-shift settings.

5\. Searching for real or semi-real spatial epidemic or pest outbreak datasets for external validation.



These additional experiments will help determine whether the current framework can support a stronger manuscript suitable for SCI journal submission.

