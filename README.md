# bindFx
* TF-binding Effect Prediction (bindFx)

### Install Requirements
Using Conda:<br>
`conda create --name <env_name> --file requirements.txt`<br>

Using pip:<br>
`pip install -r requirements.txt`

## Abstract
Somatic mutations can play functional roles within the genome, thereby contributing to the development of complex human diseases like cancer. Most of these somatic mutations predominantly fall into non-coding regions which make up around 98% of the human genome, where they can have a functional influence by disrupting the accord of regulatory interactions between transcription factors (TFs) and DNA. 

In this study, we propose a novel in-silico approach to assess the repercussion of non-coding mutations on TF-DNA interplay. Our methodology hinges on k-mer-based models of TF-binding specificity, trained on high-throughput in vivo ChIP-seq data.  To estimate the optimal parameters of the k-mer regression binding model for each human TF, we employ the stochastic gradient descent (SGD) algorithm. There are various existing DNA-binding models like the conventional ordinary least squares (OLS) method or recently developed other models that can be used to assess TF-binding specificity. 

However, SGD offers several advantages, including faster convergence and the ability to handle large training datasets more efficiently compared to other models. Plus, SGD can be also performed not only on high-throughput in vivo data but also in vitro data from universal protein-binding microarray (uPBM) experiments. Although the scarcity of the statistical assessment on other models exists, a normalized score (t-value) and corresponding p-value for each predicted change in TF-binding can be computed by SGD estimations. Employing our various TF-binding models constructed by the SGD method to somatic mutation catalogues from 21 breast cancer samples, we can unveil the effects of non-coding mutations on TF binding affinity.  Then, we can further investigate which TF bindings are disrupted and which de novo TF bindings are created by augmenting the loss of TFs (loss of function) and the gain of TFs (gain of function) with the corresponding significant p-values
