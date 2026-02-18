# A deep generative framework for joint households and individuals population synthesis

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/75a1bfb8196bf59b4db2b35b8bdaa2bd9a752935d6abe3ec53c7efbd796f22e3.jpg)


Xiao Qian iD, Utkarsh Gangwal, Shangjia Dong∗ iD, Rachel Davidson $\textcircled { \scriptsize { 1 } }$ 

Department of Civil and Environmental Engineering, University of Delaware, Newark, DE 19716, United States 

# H I G H L I G H T S

• Introduces a deep generative framework for end-to-end joint household-individual population generation. 

• Captures household-individual and individual-individual relationships via a novel data restructuring scheme. 

• Ensures synthetic population realism by learning individual attribute distributions and joint variable correlations. 

• Employs transfer learning to synthetic population marginal distributions with census tract data. 

• Generates realistic synthetic populations in Delaware and North Carolina and preserves data privacy. 

# A R T I C L E I N F O

Keywords: 

Variational autoencoder (VAE) 

Transfer learning under distribution shift 

Synthetic population 

Joint individual-household inventory 

# A B S T R A C T

Household- and individual-level sociodemographic data are critical for analyzing human and infrastructure interactions and informing policy. Yet, available data sources are limited: the American Community Survey microdata provides only samples at the public use microdata area, while census tract data contains only marginal distributions without dependencies. To bridge this gap, we develop a deep generative framework, transfer learning with variational autoencoder (TL-VAE), that produces synthetic populations with realistic household–individual and individual–individual relationships while matching tract-level distributions. The methodological contributions include (1) a new data structure for capturing household-individual and individual-individual relationships, (2) a transfer learning process with pre-training and fine-tuning steps to generate households and individuals whose aggregated distributions align with the census tract marginal distributions, and (3) a decoupled binary cross-entropy (D-BCE) loss function enabling distribution shift and out-of-sample records generation. Model results for an application in Delaware, USA, demonstrate the ability to ensure the realism of generated household-individual records and accurately describe population statistics at the census tract level compared to existing methods. Furthermore, testing in North Carolina, USA, yielded promising results, supporting the transferability of our method. 

# 1 . Introduction

Urban planning [1–3], disaster response and emergency management [4,5], household adaptation behavior analysis [6], and healthcare planning [7,8] can all benefit from an accurate population dataset. With ever-increasing attention to equity and environmental justice in decision-making, there is a heightened imperative to conduct householdlevel investigations to capture the heterogeneous behaviors within the community [9]. Central to this effort is a comprehensive and accurate population dataset, serving as the cornerstone for the analysis and mapping of interactions between humans, the built environment, and external factors like disaster disruptions and policy interventions. However, 

due to privacy and other concerns [10,11], access to a complete true population dataset is often restricted and only anonymized samples and aggregated totals are available. The lack of a population dataset hampers a nuanced understanding of interactions between humans and the built environment at large. Therefore, there is a pressing need to create a realistic synthetic population dataset. In this work, we focus on joint household-individual population datasets in which each individual is defined by their values on a set of individual attribute variables (e.g., gender, age), and each household is comprised of one or more of those individuals and is similarly defined by its values on a set of household attribute variables (e.g., household income). 

Email address: sjdong@udel.edu (S. Dong). 


(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/0f78d7a42d375bc0b780e9c4570610f8797e130e5cac71245a088388e5eebc0b.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/df70036ed68da43b50bba254b0eefec3d67ca59fddc8430074042b07a537004d.jpg)



Fig. 1. Comparison of household income distribution. (a) Map of the state of Delaware with two randomly selected census tracts, A and B; and (b) marginal distributions of household income for the state (microdata), census tract A, and census tract B.


The need for a joint household-individual dataset stems from applications where aggregate data is insufficient. In policy analysis, highresolution synthetic populations enable evaluation of household-level impacts of infrastructure or climate adaptation strategies, revealing disproportionate burdens across socioeconomic groups [12]. In disaster response, combining damage assessments with household sociodemographic details supports estimates of shelter demand, targeted evacuations, and prioritization of aid to vulnerable populations [13]. These uses require geographically precise and structurally realistic population data. 

Ideally, a synthetic population dataset should possess the following key features: 

1. Each individual is realistic. That means each individual’s characteristics should match real-world information. For example, there should not be a lot of very high-income 18-year-olds or teenagers who have doctoral degrees. 

2. Each household is realistic. Namely, household variables’ dependencies should mirror those found in actual households. Additionally, the relationships among individuals within a household should align with real-world patterns. For example, households with individuals holding advanced degrees are more likely to have higher incomes, while lower-income families typically possess fewer vehicles. 

3. The overall population is realistic. The synthetic population’s marginal distributions of individual and household variables should match those observed in real populations at the aggregate level. For example, the synthetic population should reflect the correct proportion of wealthy households as indicated by state statistics. This ensures that the synthetic population accurately represents the characteristics of the actual population. 

4. The geographic distribution of the population is realistic. Because population characteristics in different regions vary significantly, the marginal distributions of individual and household variables in the synthetic population should correspond to the ground truth marginal distributions. For example, the distribution of high-income households should match the wealth pattern in real life. 

Data Challenge Extensive efforts have been made to create synthetic population datasets, some even incorporated aspects of the housing unit characteristics [14] or workplace assignments [15]. Public data sources such as the American Community Survey (ACS) and American 

Housing Survey (AHS) are commonly used for synthetic population development. However, the varying scales of the available population data samples and distributions make synthetic population generation a unique challenge. 

The ACS Public Use Microdata Sample, hereafter referred to as microdata, is released annually by the United States Census Bureau and offers detailed records of individual people and housing units [16]. These records cover a wide range of social, economic, housing, and demographic characteristics. With multiple variables for each individual and household, they provide the dependencies among variables. Unfortunately, ACS PUMS is sampled at the PUMA-level (Public Use Microdata Area), distinct and non-overlapping geographic regions that divide each state or equivalent entity into areas with a minimum population of 100,000. It is based on a sample of $1 \%$ for a single year or $5 \%$ for five years. Researchers may also wish to deploy surveys to collect additional attributes that are not captured by microdata. In such cases, using private data to generate a synthetic population must ensure that the privacy of the original human-subject data is preserved. 

The ACS also provides data at the census tract or block group level, hereafter referred to as census tables, but it includes only marginal distributions of selected attributes [17]. Due to the geographic distribution of the population, the marginal distribution of each variable may differ across census tracts and between the census tracts and PUMAlevel marginal distributions in the microdata (e.g., Fig. 1). Ideally, the individual and household variables that describe the synthetic population should exhibit the variable dependence as in the microdata and the marginal distributions for each census tract from the local data. 

Research Gap Synthetic population generation has long relied on optimization-based methods such as Iterative Proportional Fitting (IPF) [18], Gibbs sampling [19], and their variants [20]. These approaches face the curse of dimensionality, with performance degrading as the attribute space grows, and they primarily replicate existing samples rather than synthesizing new heterogeneity beyond the microdata [19]. Recent advances in deep generative modeling, including Variational Autoencoders (VAE) [21,22], Generative Adversarial Networks (GANs) [23,24], and Diffusion Models [25,26], address some of these limitations by generating out-of-sample data with high-dimensional attributes. Nonetheless, conventional generative models remain unsuitable for population synthesis. By design, they are trained to maximize the evidence lower bound (ELBO) or minimize divergence between the learned distribution $p _ { \theta } ( \mathbf { x } )$ and the training distribution $p _ { \mathrm { d a t a } } ( \mathbf { x } )$ , entangling dependency structures with the observed marginals [27,28]. This prevents adaptation to new geographic contexts where dependency 

structures must be preserved but marginals differ (e.g., aligning microdata with census tract tables). Our framework overcomes this limitation by explicitly decoupling dependency learning from marginal alignment. 

Contributions In this research, we introduce a novel deep-learning population synthesis framework with both household and individual characteristics embedded, aiming to include all key features outlined for an ideal synthetic population dataset. The primary technical contributions of this work can be summarized as follows: 

• We propose a table restructuring technique to facilitate the learning of household-individual and individual-individual relationships in microdata (Feature 2), enabling the generation of a synthetic household and individual inventory simultaneously. This data representation streamlines the learning and generation process, overcoming the limitations of the conventional two-step approach of first generating synthetic individuals and then assembling them into households, which fails to capture the relationships between individuals who live in the same household. 

• We present a novel parameter-efficient transfer learning algorithm that enables the adaptation of generative models trained on PUMAlevel microdata to produce synthetic households and individuals at the census tract level while conforming to the target marginal distributions from the ACS census data table (Features 3 & 4). This method preserves the realism of individual households and individual records as depicted in microdata. Beyond population synthesis, the proposed algorithm can also be applied to other learning and generative tasks, particularly those with differing distributions between training and target data. 

• We introduce a new loss function, Decoupled Binary Cross-Entropy (D-BCE), aimed at gauging the realism of synthetic data by quantifying the difference between the synthetic data and real samples (i.e., microdata) (Features 1 & 4). 

# 2 . Related works

Existing literature to generate the synthetic population with both households and individuals can be grouped into four main categories: (i) synthetic reconstruction (SR), (ii) combinatorial optimization (CO), (iii) statistical learning (SL), and (iv) deep generative methods [29,30]. 

# 2.1 . Synthetic reconstruction

Methods in this category typically follow a two-step process: fitting (where non-integer weights are assigned to individuals and households to match the marginal totals) and allocation (where these non-integer weights are converted to integers and individuals are then replicated based on these weights accordingly). A widely used SR technique is Iterative Proportional Fitting (IPF), which involves building a contingency table to match the marginal totals by minimizing discrimination information or relative entropy [31,32]. While the IPF is simple and fast, its original formulation is incapable of generating both household characteristics and individual attributes concurrently. Researchers trying to use IPF to create a joint distribution of households and individuals either fit the household and individual attributes separately or sequentially, resulting in inconsistent fitting [3,33]. To overcome this limitation, researchers have turned to two-layered population generation methods such as hierarchical IPF [34,35] and iterative proportional update (IPU) [36], which group individuals into households while satisfying marginal totals at both levels. Hierarchical IPF or IPU entails iteratively computing weights for individual and household records, with cross-categorization of individual types into different household types [37,38]. Synthetic population generation can also be viewed as a constrained optimization problem. The goal of optimization model formulations is to calculate household weights so that the weighted distribution of various attributes aligns with population distributions. One commonly used optimization model is entropy maximization 

[39–41]. This approach aims to generate a synthetic population that closely aligns with specified marginal distributions by maximizing entropy while adhering to constraints derived from the sample population data [42]. By maximizing entropy, this model introduces diversity and randomness into the synthetic population, effectively safeguarding the privacy of sample population data. Researchers have also explored other optimization-based models, such as generalized ranking [43]. In this approach, the classical ranking ratio method is often employed to calibrate marginal counts in the frequency table by minimizing the discrepancy between initial and newly estimated weights. The majority of methods in the synthetic reconstruction category rely on both sample and marginal data. Once weights are assigned to individual samples during fitting, they remain unchanged, making these methods deterministic [29]. 

# 2.2 . Combinatorial optimization

The methods in this category aim to find an optimal solution from a finite set of objects. First, an initial synthetic population is generated, often randomly or based on some initial heuristic. This population might not yet satisfy the required constraints, such as demographic distributions, income levels, and household sizes. Next, households are drawn from the microdata to identify the best fit. Starting with randomly chosen households, the process is followed by adding, replacing, or swapping a household in the sample. If the replacement increases the fit, the household is kept [44,45]. This process is repeated until either the objective is reached or a fixed number of iterations is reached. However, the possibility of finding the optimal set can become computationally expensive if the size of the finite set is too large [46]. Therefore, researchers have proposed heuristic algorithms to find a near-optimal solution in these scenarios, including simulated annealing [44,47] and genetic algorithms [48–50]. Birkin et al. [51] implemented a genetic algorithm to generate a synthetic population for some regions in the United Kingdom, but found the model’s performance to be poor as the model failed to find enough individuals from ethnic groups constituting the minority population. Similar to synthetic reconstruction, the methods in combinatorial optimization also require both the sample and marginal data and generate a synthetic population by replicating individuals. 

# 2.3 . Statistical learning

The third category of methods involves simulation-based approaches [29]. Unlike the other two categories, the methods in this category focus on learning the joint distribution of the variables of interest from the available microdata [19,30]. These methods avoid replication of samples by estimating probabilities for different combinations, including those not present in the microdata. Markov process-based methods, including the Markov Chain Monte Carlo (MCMC) simulation-based approach and the Hidden Markov Model (HMM) are a couple of widely used statistical learning-based approaches to simulate the synthetic population. The MCMC methods involve constructing the conditional distributions (e.g., income level given a set of predictors such as age and education) from microdata or zonal statistics using some parametric model (e.g., multinomial linear logistic regression). Later, the Gibbs sampler or MCMC leverages this conditional distribution of each attribute to create individuals from the joint distribution [19]. On the other hand, the HMM models the sequence of observable events that depend on internal factors or are generated by Markovian hidden state processes [52]. However, these studies using MCMC and HMM are limited to generating individuals and pay little attention to the hierarchical structure of households [29]. Casati et al. [53] proposed an extension to the method and used a hierarchical MCMC to group individuals into households, generating a two-layered synthetic population while accounting for the household hierarchical structure. Another statistical learning-based method used by researchers to create a two-layered synthetic population is the Bayesian Network (BN). It is a probabilistic graphical model where a set of random variables (nodes) and their 

conditional distributions (edges) are represented in the form of a directed acyclic graph [54,55]. Zhang et al. [56] defined a BN to consist of two main steps: (i) learning the network structure describing the dependence among related variables and (ii) estimating the parameters to learn the conditional distribution. Sun and Erath [57] showed that BN can capture complex dependencies and higher-order interactions within different variables by concisely abstracting the population structure. To further improve upon capturing the strong interdependencies within a household, Sun et al. [30] proposed a multinomial hierarchical mixture model. The proposed framework uses a two-level hierarchical data structure and integrates a multilevel latent class model [58] to capture the interdependencies. The different statistical learning methods discussed above use a joint probability distribution to overcome the lack of heterogeneity, which could not be resolved by synthetic reconstruction and combinatorial optimization. However, a major drawback of the statistical learning methods is that they fail to satisfy the conditional and marginal distributions simultaneously. Therefore, studies suggest using synthetic reconstruction as a post-processing step after generating a suitable representative population sample using statistical learning. For example, Casati et al. [53] and Rahman and Fatmi [54] used generalized ranking to post-process output from MCMC and BN, respectively. 

# 2.4 . Deep generative methods

Recent advances in computer science techniques have allowed researchers to overcome the limitations of traditional methods with the help of Deep Generative modeling techniques. Researchers have categorized these methods as statistical learning due to their ability to learn the joint distributions [29]. Unlike other statistical learning methods, however, deep generative methods do not require post-processing of the generated samples and can easily deal with many attributes. A deep learning approach involves learning a comprehensive representation from sample tables containing detailed information and using a generative neural network to synthesize a generative table. This process enables the creation of new data that aligns with the joint distribution of the sample tables. Researchers have deployed Generative Adversarial Networks (GAN), including Tabular GAN, conditional tabular GAN, and Copula GAN, to create new data to improve the disaggregated records and generate more representative and diverse datasets [59–61]. Moreover, Lederrey et al. [62] proposed using a Directed Acyclic Tabular GAN (DATGAN) that involved integrating expert knowledge. The authors provided the neural networks with a structure of variables, which allowed them to avoid overfitting and remove possible biases. In addition to these methods, researchers also proposed the use of the Variational Autoencoder (VAE) to synthesize synthetic populations [21,63]. VAE uses unsupervised learning to determine the latent variables from the 

training data (encoder) and uses them to generate new data (decoder). Borysov et al. [21] found VAE to be computationally efficient while outperforming the statistical learning-based methods for higher dimensions. However, different generative models proposed by researchers focused on generating individual data and did not incorporate the joint household-individual structure for the synthetic population generated. Aemmer et al. [22] overcame the limitation by using a Conditional-VAE (CVAE) capable of synthesizing household and individual data simultaneously without any need for post-processing and grouping. The proposed method involved using Household CVAE to generate synthetic households and using them alongside the latent variables of the Individual CVAE decoder to enable combining individuals with households. Nonetheless, the model fails to capture the relationship between the individuals living in the same household. Moreover, using two CVAEs increases the computational demand as it involves training two models. 

Unlike the existing deep learning approach for population synthesis, the proposed approach can generate data conforming to marginal distributions outside the training data (microdata), such as the census tract marginal distribution. Moreover, the proposed method integrates households and individuals more flexibly through microdata restructuring, eliminating the need for training multiple models. 

Choi et al. [64] present a survey covering applications of deep generative models in transportation research, such as trajectory generation, population and activity synthesis, and traffic data imputation. Within population synthesis, they emphasize methods that not only replicate the training sample structure but also satisfy externally specified marginals (e.g., census tract constraints). Copula models [65] achieve this by separating joint distributions into marginals and dependence structures. In continuous cases, marginals can be substituted directly, while discrete settings require discrete copulas or checkerboard approximations; in high dimensions, vine copulas are commonly applied to mitigate dimensionality challenges. Although copulas naturally support marginal replacement, they face practical limitations: identifiability and resolution issues in discrete domains and computational burdens in high dimensions [66,67]. Section 5.2.2 provides a detailed theoretical and empirical comparison between our parameter-efficient fine-tuning approach and copula-based baselines. 

# 3 . Population synthesis framework

In this study, we introduce a deep generative population synthesis framework, Transfer Learning with Variational Autoencoder (TL-VAE), as shown in Fig. 2, that leverages the learning of a joint distribution from microdata, ensuring the generation of synthetic households and individuals whose marginal distribution matches that of the target census tract. This framework lays the groundwork for comprehensive data generation 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/696975f311716c5969b75881e82e0504cbd618c4a85ce2ca2a8786af879aad85.jpg)



Fig. 2. The end-to-end deep generative pipeline for synthetic household-individual inventory development.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/9b6e3efe3c09a47d9c986cf68e93f30aa9ad6630e29e7b6e7cea8fac7c376e75.jpg)



Fig. 3. Data restructuring procedure illustration.


processes with distribution shifts. Moreover, its applicability extends beyond population synthesis to other data types, especially those with divergent distributions between training and target data. The method includes three main steps: (1) restructuring the microdata to facilitate the learning of joint household-individual and individual-individual associations, (2) constructing a transfer learning pipeline to allow the deep generative models, such as VAE, to learn joint distributions in microdata and generate synthetic population conforming to distinct marginal distributions, (3) devising a decoupled binary cross-entropy (D-BCE) loss function to enable the creation of new synthetic individuals rather than solely replicating those in the microdata. The following sections provide detailed explanations for each step. 

We assume that the source (microdata) and target (tract) populations share the same multivariate dependence structure (copula) [65,68,69], while allowing their marginals to differ. This is a common assumption when transferring dependency structures across regions with varying demographics, and its scope and limitations are discussed in Section 5.2.2. 

# 3.1 . Microdata restructure

Microdata includes details about both households and individuals. Our objective is to create a synthetic household and individual inventory that can capture (1) the connection between households and persons (i.e., household-individual) and (2) the relationships among individuals within the same household (i.e., individual-individual). To achieve this, we need to learn a high-dimensional joint distribution that captures these relationships. 

Current methodologies commonly introduce conditional variables [22] or domain expertise [62] into the model to capture variable relationships. This is because the data structure in their loss functions cannot represent household-individual and individual-individual relationships. Specifically, existing data structures consolidate households and individuals based on household ID, creating multiple records within one household, such as H1-P1, H1-P2, and H1-P3, as illustrated in Fig. 3. However, this setup leads to these records being processed separately, treating them as independent individual inputs. Consequently, we cannot effectively learn the relationships between individuals within the same household. Therefore, the traditional organization of population datasets, where one household is divided into multiple person records, hinders our ability to capture relationships between individuals within the same household and between individuals and households. 

To overcome the limitations of the existing population data structure that impact the accuracy of population synthesis, we propose a new way of restructuring microdata, wherein individuals belonging to the same household are added into the same row in the population table, such as 

H1-P1-P2-P3. This restructuring enables the model to effectively learn the relationships between households and individuals, as well as among individuals within the same household. 

Fig. 3 illustrates the data restructuring procedure used to handle variable household sizes. The household and person tables in the microdata are first merged by household ID. We then set the household window size, $N _ { \mathrm { w i n d o w } }$ , to the maximum household size observed in the dataset: 15 individuals in our Delaware and North Carolina samples. Each household is expanded to this window size, with smaller households padded by “NA” values to produce a uniform matrix representation $\mathbf { H } \in \mathbb { R } ^ { N _ { \mathrm { w i n d o w } } \times D }$ , where $D$ is the number of encoded attributes. This format enables efficient batch processing in the neural network. After synthetic households are generated in this window-based format, a post-processing step removes “NA” entries to recover the actual household composition, ensuring variable household sizes consistent with the empirical distribution in the PUMS microdata. 

We handle “NA” values using a modified one-hot encoding scheme. For each categorical variable, an additional category is introduced to represent “NA” or non-existence. Thus, for a categorical variable $X$ with $k$ possible values, the encoding function $\phi : X \to \{ 0 , 1 \} ^ { k + 1 }$ is defined as: 

$$
\phi \left(x _ {i}\right) = \left\{ \begin{array}{l l} \mathbf {e} _ {j} & \text {i f} x _ {i} = v _ {j} \text {f o r} j \in \{1, 2, \dots , k \} \\ \mathbf {e} _ {k + 1} & \text {i f} x _ {i} = ^ {\prime \prime} \mathrm {N A} ^ {\prime \prime} \end{array} \right. \tag {1}
$$

where $\mathbf { e } _ { j }$ denotes the $j$ -th standard basis vector in $\mathbb { R } ^ { k + 1 }$ . Thus, a variable with $k$ categories is represented by $k + 1$ binary features, with the additional feature indicating the absence of a valid value (i.e., “NA”). During model training, the VAE decoder outputs probability distributions over all categories, including the “NA” category. For each position ?? in the window and each categorical variable $X _ { j }$ , the decoder produces a probability vector $\mathbf { p } _ { i , j } = [ p _ { i , j , 1 } , p _ { i , j , 2 } , \ldots , p _ { i , j , k } , p _ { i , j , \mathrm { N A } } ] ^ { T }$ 

$$
p _ {i, j, c} = \frac {\exp \left(z _ {i , j , c}\right)}{\sum_ {c ^ {\prime} = 1} ^ {k + 1} \exp \left(z _ {i , j , c ^ {\prime}}\right)} \tag {2}
$$

where $z _ { i , j , c }$ are the logits produced by the decoder. The cross-entropy loss naturally accommodates these multi-class predictions, including the “NA” category: 

$$
\mathcal {L} _ {\mathrm {C E}} = - \sum_ {i = 1} ^ {N _ {\text {w i n d o w}}} \sum_ {j = 1} ^ {D} \sum_ {c = 1} ^ {k _ {j} + 1} y _ {i, j, c} \log \left(p _ {i, j, c}\right) \tag {3}
$$

where $y _ { i , j , c }$ is the one-hot encoded ground truth and $k _ { j }$ is the number of categories for variable $j$ . 

Finally, we arrange all persons within each household into a single row. In our experiments, sorting individuals by features such as age 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/0c67f262a0a0984f74dbe0f1cf8aa9559c2b1e4add593aaf45f3e9af4bef5e60.jpg)



Fig. 4. Parameter-efficient transfer learning with VAE (TL-VAE).


and education improves the model’s ability to capture intra-household relationships. Each row thus represents a complete household, enabling the use of standard record-level loss functions to learn household representations. 

Subsequent studies have validated the benefits of this approach: Sané et al. [70] conducted extensive experiments on a French household travel survey dataset, demonstrating its advantages over conventional restructuring methods. Their research introduced two frameworks that build directly on our data restructuring principles. SVAE-Pop2 implements our window-based restructuring with fixed-size padded inputs, while MVAE-Pop2 adopts a multi-model approach, using dedicated models for different household sizes. 

The experimental results provide clear evidence of the advantages of our data restructuring scheme. Both implementations outperformed conventional methods in capturing household-individual dependencies, achieving $1 5 \mathrm { - } 2 0 ~ \%$ improvement in preserving relationships between household income and individual age distributions compared to traditional two-step methods. In terms of computational efficiency, the unified window-based approach (SVAE-Pop2) required only a single trained model, whereas the multi-model approach (MVAE-Pop2) needed separate models for each household size, increasing computational cost roughly threefold. Importantly, the restructuring scheme preserved realistic associations within households, including income-age correlations, occupational patterns among members, geographic clustering of household types, and intergenerational composition patterns. 

These findings validate that our data restructuring approach enables the generation of synthetic populations that more faithfully capture the complex relationships between household and individual characteristics, which are critical for accurate agent-based modeling and policy simulations. 

# 3.2 . Parameter-efficient transfer learning under distribution shifts

Deep generative methods often entail two primary steps: training and inference. Traditional methods employ a loss function to compute reconstruction errors, establishing a direct mapping between inputs and outputs. While this ensures consistency with the statistical patterns observed in microdata, it also results in the model learning the joint distribution of the microdata, as shown in Fig. 4. Consequently, the inference stage generates data that tends to align with this joint distribution 

rather than matching the targeted marginal distribution at the census tract level. 

Since the trained model can learn the joint distribution of microdata well and generate a realistic synthetic population, intuitively, we would like to retain the trained model’s learning ability that is embedded in the already trained parameters based on microdata and transfer it to generate data that conforms to a different distribution. This concept refers to transfer learning under distribution shifts. Transfer learning enables the generation of data that conforms to the targeted marginal distribution of specific census tracts without compromising its original capacity to produce realistic household and individual records that are consistent with the microdata. We achieve transfer learning by introducing a finetuning step into the traditional training and inference process (Fig. 4). This approach draws inspiration from research on adversarial attacks in generative neural networks [71] and parameter-efficient fine-tuning techniques in large language models [72]. 

TL-VAE procedure. As shown in Fig. 4, the input to the VAE-based population synthesis pipeline is a matrix $\boldsymbol { X } \in \mathbb { R } ^ { N \times D }$ , where $N$ is the number of households in microdata and $D$ is the number of household and individual attributes after one-hot encoding (see Section 4.1). Each row of the input, $X _ { i } ,$ represents a restructured household record con­ taining both household and individual characteristics. While the matrix can be processed in batches for computational efficiency, each record is handled independently by the encoder $E _ { \theta }$ and decoder $D _ { \theta }$ . 

The encoder outputs two vectors, the mean $\pmb { \mu }$ and standard deviation $\pmb { \sigma }$ of the latent space, which are then reparameterized to obtain the latent variable $\mathbf { Z } = \pmb { \mu } + \epsilon \odot \pmb { \sigma } _ { }$ , where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ represents independent standard normal noise, ensuring differentiable sampling and stable gradient flow during training. Following standard VAE design, ?? is modeled as a multivariate normal vector with independent components, implemented via a diagonal covariance matrix. This promotes a disentangled representation and enables smooth interpolation in latent space, which is critical for generative tasks. In this stage (Fig. 4(a)), the latent space ?? is regularized to approximate a normal distribution using KL-Divergence $( D _ { K L } )$ . 

The decoder receives ?? and outputs probability distributions for each variable. For example, for the variable “tenure,” which is encoded as 

Tenure Vehicle available Gender the first segment of $\hat { X } _ { i } = \widehat { ( 0 . 2 , 0 . 8 , 0 . 1 , 0 . 1 , 0 . 6 } , \widehat { 0 . 9 , 0 . 1 } , \ldots ) .$ , the decoder 

produces probabilities [0.2, 0.8], corresponding to the likelihood of “Owned” and “Rented.” 

Traditionally, during the inference stage, generate tasks (Fig. 4(b)) often involve sampling ?? directly from a normal distribution and inputting it into a well-trained decoder $D _ { \theta }$ to generate a realistic synthetic output embedding vector $\hat { X }$ . However, traditional VAE can only generate data that follows the same distribution as the input data, which differs from our objective. 

In the proposed fine-tuning step (Fig. 4(c)), the latent space is set as a trainable matrix $\mathbf { Z } _ { \theta }$ . Since our objective is to produce a number of households for a specific census tract, which often differs from the size of the input microdata (??), $\mathbf { Z } _ { \theta }$ is sized $\mathbb { R } ^ { N _ { t } \times D }$ . Here, $N _ { t }$ represents the desired number of households to generate for a target census tract, and $D$ is the number of population attributes after encoding (see Section 4.1). The decoder $D _ { \theta }$ still processes each row of $\mathbf { Z } _ { \theta }$ independently, either individ­ ually or in a batch. After obtaining the synthetic embedding vector $\hat { X _ { i } }$ , we then organize the vectors into a matrix $\hat { X } \in \mathbb { R } ^ { N _ { t } \times D }$ . The fine-tuning objective is: 

$$
\min _ {\mathbf {Z} _ {\theta}} \mathcal {L} _ {\mathrm {m a r g}} (D _ {\theta} (\mathbf {Z} _ {\theta})) + \lambda \mathcal {L} _ {\mathrm {D - B C E}} (D _ {\theta} (\mathbf {Z} _ {\theta}), X _ {\mathrm {P U M S m i c r o d a t a}})
$$

where ${ \mathcal { L } } _ { \mathrm { m a r g } }$ is an RMSE loss that steers the aggregate marginals of the generated population $D _ { \theta } ( \mathbf { Z } _ { \theta } )$ with census tract targets, and $\mathcal { L } _ { \mathrm { D - B C E } }$ is our proposed decoupled binary cross-entropy loss that preserves the dependence structure learned during pre-training. This loss is backpropagated through the frozen decoder $D _ { \theta }$ to the trainable latent matrix $\mathbf { Z } _ { \theta }$ , which is updated iteratively. The update continues until the out­ put $\hat { X } \in \mathbb { R } ^ { N _ { t } \times D }$ closely matches the target marginal distribution while maintaining realistic record-level dependencies. Because only the latent inputs $\mathbf { Z } _ { \theta }$ are updated and the model parameters remain fixed, this fine-tuning procedure is highly parameter-efficient. 

The proposed transfer learning procedure can be applied to various generative models, including Variational Autoencoder (VAE), Generative Adversarial Networks (GAN), and Diffusion Models. To demonstrate the effectiveness of the transfer learning approach, we use the VAE model in this study (Fig. 5). An autoencoder (AE) comprises two components: an encoder and a decoder. The encoder compresses data from a higher-dimensional space into a lower-dimensional space, known as the latent space, while the decoder reconstructs the latent space back into the higher-dimensional space. Both components are trained together using a loss function that aims to reconstruct the input accurately at the output. We harness the capabilities of an autoencoder to learn continuous representations of the microdata’s heterogeneous features within the latent space [73]. In contrast, a Variational Autoencoder (VAE) introduces a constraint on the latent distribution, forcing it to follow a normal distribution. This ensures that the latent variable is smooth 

and continuous, thereby enabling the latent space to have generative capabilities. 

The encoder includes six feedforward neural networks. In the last layer, separate fully connected layers and Batch Normalization (BN) are used to output $\mu$ and $\sigma$ . The decoder also includes six feedforward neural networks, with the last layer’s output dimension set to $D$ (i.e., the number of population attributes after encoding). After applying one-hot encoding to a set of vectors for the same variable, softmax is used to generate the probability of each variable. Each feedforward neural network in both the encoder and decoder consists of a fully connected layer, followed by a BN layer and a ReLU (Rectified Linear Unit) activation layer. 

VAE structure and activations. The inputs are high-dimensional, sparse one-hot vectors representing household attributes and a fixed window of person attributes. To process these, we employ a deep multilayer perceptron (MLP) VAE with six layers in both the encoder and decoder. Each hidden block consists of a fully connected layer, followed by Batch Normalization (BN) and a ReLU activation, which stabilizes optimization on imbalanced one-hot signals and improves gradient flow and convergence. 

In the encoder, the final layer uses separate fully connected layers with BN to produce the mean $\mu$ and standard deviation $\sigma$ of the latent space. The decoder also contains six feedforward layers, with the final layer output dimension set to $D$ , the number of population attributes after encoding. One-hot encoded vectors for each variable are converted into per-variable categorical probabilities using group-wise Softmax, while binary outputs are handled with Sigmoid. This design aligns with the tabular, non-sequential feature layout; preliminary tests showed that attention-based transformer variants offered no accuracy gains under comparable compute budgets. 

The latent dimension is set to 512 and the hidden width to 1024, providing sufficient capacity to capture cross-household and individual dependencies while keeping decoder-only fine-tuning efficient. Smaller configurations degraded marginal alignment and joint-structure fidelity, whereas larger ones yielded diminishing returns at higher computational cost. The restructured input dimension is $D =$ lenVarHouse+windowSize× lenVarPerson (e.g., Delaware: 462; North Carolina: 757), which motivates the chosen capacity. 

A standard VAE is trained by maximizing the evidence lower bound (ELBO) over the training distribution $p _ { \mathrm { d a t a } }$ : 

$$
\mathcal {L} _ {\mathrm {E L B O}} (\theta , \phi) = \mathbb {E} _ {p _ {\mathrm {d a t a}} (\mathbf {x})} \left[ \mathbb {E} _ {q _ {\phi} (\mathbf {z} | \mathbf {x})} [ \log p _ {\theta} (\mathbf {x} \mid \mathbf {z}) ] - D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z} \mid \mathbf {x}) \| p (\mathbf {z})\right) \right]. \tag {4}
$$

Optimizing this objective encourages the learned distribution $p _ { \theta } ( \mathbf { x } )$ to reproduce the marginals and dependencies of the training data (PUMAlevel microdata). As a result, a generic VAE will inherently replicate the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/4bfbb3bf6eccfb680c387001e63979919a4daa799a228bf2d6e18ac56f133a4e.jpg)



Fig. 5. VAE structure in the proposed deep generative framework.



Binary Cross Entropy (BCE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/1bfa4411360f074b9779c95473c991d5cde6270cc0c063b80896f5203ba34c06.jpg)


$$
\begin{array}{l} \begin{array}{c} \text {O n e t o O n e} \\ \hline \end{array} \\ = \left\{ \begin{array}{l} \underline {{1}} \underline {{0}}, 0, 0 \\ 0, \underline {{1}}, 0 \\ 0, 0, \underline {{1}} \end{array} \right. \cdot \left\{ \begin{array}{l} \overline {{l (x _ {1} , \widehat {x} _ {1})}}, l (x _ {1}, \widehat {x} _ {2}), l (x _ {1}, \widehat {x} _ {3}) \\ \overline {{l (x _ {2} , \widehat {x} _ {1})}}, \overline {{l (x _ {2} , \widehat {x} _ {2})}}, l (x _ {2}, \widehat {x} _ {3}) \\ l (x _ {3}, \widehat {x} _ {1}), \overline {{l (x _ {3} , \widehat {x} _ {2})}}, l (x _ {3}, \widehat {x} _ {3}) \end{array} \right\} \\ \end{array}
$$


Decoupled Binary Cross Entropy (D-BCE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/a00e0642ac4f1045312cfdcae0592a65e306a5aeeb35cb4396cb9535b5670b7c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/a46360ee721c6f622fee7dd35d40eca7278d684a9923e7b0b52f05ff43c08ef1.jpg)



Fig. 6. Illustrative comparison between BCE and D-BCE.


training marginals at inference and cannot match different tract-level marginals without an adaptation mechanism, such as our latent-only fine-tuning (Section 3.2). 

# 3.3 . Decoupled binary cross-entropy (D-BCE)

During the training of the generative model to emulate the microdata, the Binary Cross-Entropy (BCE) loss function is commonly utilized. We implemented the BCE loss function following its definition in PyTorch [74]. Its mathematical expression is as follows: 

$$
\text {B C E} l \left(x _ {i, j}, \hat {x} _ {i, j}\right) = - \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {D} \left[ \hat {x} _ {i, j} \log x _ {i, j} + \left(1 - \hat {x} _ {i, j}\right) \log \left(1 - x _ {i, j}\right) \right] \tag {5}
$$

where $N$ represents the number of households in the microdata, $D$ denotes the number of population variables after one-hot encoding, $x _ { i , j }$ represents the actual value of the $j$ -th variable in the ??-th record, and $\hat { x } _ { i , j }$ is the generated value of the $j$ -th variable in the ??-th record. When $\log x$ is 0, PyTorch clamps $\log x$ to a value no smaller than $^ { - 1 0 0 }$ to ensure the stability of the model. The BCE loss function necessitates a one-to-one match between the generated and microdata households at the record level, as illustrated in Fig. 6. However, this also results in the generated household’s marginal distribution matching that of the microdata, contradicting our objective of producing authentic households that adhere to the target marginal distribution at each census tract level. 

We propose that each generated household does not need to precisely match its corresponding input data record. Instead, as long as it closely resembles any record in the microdata, we consider it an authentic generated instance. Guided by this principle, we have formulated the Decoupled Binary Cross-Entropy (D-BCE) loss function. The detailed procedure is illustrated in Algorithm 1. 

Variable dependence preservation via a differentiable nearest-sample distance. Our D-BCE loss preserves the dependency structure learned from the PUMS microdata by encouraging each synthetic record to remain close to real records. This builds on the classical nearest-sample distance (NSD), a standard tool for assessing realism and privacy in synthetic data [75,76]. 

Let ?? = ?? ???? ??=1 $X = { x _ { j } } _ { j = 1 } ^ { N }$ be the one-hot, restructured PUMS microdata and $\hat { X } =$ $\hat { x } i i = 1 ^ { N _ { t } }$ the synthetic outputs from the frozen decoder. With $d ( \hat { x } , x ) =$ $\mathrm { B C E } ( \hat { x } , x )$ , the classical NSD for $\hat { x }$ is $\begin{array} { r } { \mathrm { N S D } ( \hat { x } ; X ) = \operatorname* { m i n } _ { j } d ( \hat { x } , x _ { j } ) } \end{array}$ . However, the ’min’ operation in NSD is non-differentiable, making it unsuitable for gradient-based optimization. To address this, we replace it with a soft relaxation: 

$$
\mathcal {L} _ {\text {r e c o r d}} \left(\hat {x} _ {i}, X\right) = \sum_ {j = 1} ^ {N} w _ {j} \left(\hat {x} _ {i}\right) d \left(\hat {x} _ {i}, x _ {j}\right), \quad \text {w h e r e} w _ {j} \left(\hat {x} _ {i}\right) = \frac {e ^ {- d \left(\hat {x} _ {i} , x _ {j}\right)}}{\sum_ {k = 1} ^ {N} e ^ {- d \left(\hat {x} _ {i} , x _ {k}\right)}} \tag {6}
$$

# Algorithm 1 Decoupled Binary Cross-Entropy (D-BCE).

Require: Microdata table $X \in \mathbb { R } ^ { N \times D }$ , Generated data table $\hat { X } \in \mathbb { R } ^ { N _ { t } \times D }$ 

Ensure: Decoupled Binary Cross-Entropy Loss, Decoupled Binary Cross-Entropy Norm KL 

1: Initialize vector ?????????????????? of length $N$ to zeros 

2: for each row ?? in $\hat { X } ~ ( \hat { x } _ { i } )$ do 

3: Initialize vector $\mathsf { b c e } _ { i }$ of length $N$ 

4: for each row $j$ in $X ( x _ { j } )$ do 

5: $\mathsf { b c e } _ { i } [ j ] \gets \mathsf { B C E } ( \hat { x } _ { i } , x _ { j } )$ ⊳ Compute Binary Cross-Entropy Loss 

6: end for 

7: ?????????????????? $_ i \gets$ ??????????????(?????? ) ⊳ Compute the soft minimum index of  ?? ???????? 

8: ?????????????? ← ⟨?????????????????? , ?????? ⟩ ⊳ Compute the soft minimum loss 

9: ?????????????????? ?????????????????? $^ +$ ???????????????????? ⊳ Accumulate softIndex 

10: end for 

11: Decoupled BCE Loss $\begin{array} { r } { \gets \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } } \end{array}$ ?????? ???????? ?? ⊳ Average the soft ?? ?? minimum losses 

12: Decoupled BCE Norm $\mathrm { K L }  \mathrm { K L } ( \mathrm { U n i f o r m } ( N ) .$ , ??????????????????) ⊳ KL divergence 

13: return Decoupled BCE Loss, Decoupled BCE Norm KL 

The D-BCE loss averages record-level values, $\begin{array} { r l } { \mathcal { L } \mathrm { D - B C E } ( \hat { X } , X ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { N _ { t } } \sum _ { i }  } \end{array}$ $\begin{array} { r } { \frac { 1 } { N _ { t } } \sum _ { i } \mathcal { L } \mathrm { r e c o r d } ( \hat { x } _ { i } , X ) . } \end{array}$ , and is optimized jointly with marginal alignment: 

$$
\min  _ {\mathbf {Z} _ {\theta}} \mathcal {L} _ {\operatorname {m a r g}} (\hat {X}) + \lambda \mathcal {L} _ {\mathrm {D} - \mathrm {B C E}} (\hat {X}, X), \quad \text {w h e r e} \hat {X} = D _ {\theta} (\mathbf {Z} _ {\theta}) \tag {7}
$$

with decoder $D _ { \theta }$ frozen. Algorithm 1 implements this efficiently using softmin weighting. 

Proposition 1 (smooth relaxation). The D-BCE objective, implemented via a log-sum-exp (soft-min) formulation, is smooth with respect to $\hat { x }$ and provides a differentiable approximation to the non-differentiable average nearest-sample distance, $\begin{array} { r } { \frac { 1 } { N _ { t } } \sum _ { i } \mathrm { N S D } ( \hat { x } _ { i } ; X ) } \end{array}$ . The properties of log-sum-exp as a smooth approximation of the minimum are well established [77]. 

Proposition 2 (dependence preservation under frozen decoder). Let $\mathcal { M } = \{ D _ { \theta } ( z ) : z \in \mathcal { Z } \}$ be the decoder manifold learned during pre-training and kept fixed in fine-tuning. For any $\lambda > 0 ,$ any stationary point $( \mathbf { Z } _ { \theta } ^ { \ast } , \hat { X } ^ { \ast } )$ of the fine-tuning objective has a bounded D-BCE loss, which decreases as ?? increases. Thus, each synthetic record in ${ \hat { X } } ^ { * }$ is encouraged to stay close to the manifold of the microdata $X$ and, by extension, to the pre-trained decoder manifold M. This ensures that multivariate dependencies captured during pre-training are preserved while $\mathcal { L } _ { m a r g }$ aligns the marginals, producing tract-aligned synthetic populations under Assumption A1. 

The result follows from the soft-min properties of log-sum-exp and the Lipschitz continuity of $D _ { \theta }$ , which together bound deviations from the pre-trained manifold. This approach is closely related to soft 

nearest-neighbor losses for preserving local structure in learned representations [78] and to nearest-neighbor or graph-based two-sample tests for comparing multivariate dependence, including Friedman & Rafsky’s MST approach and Schilling’s nearest-neighbor tests [75,76]. 

Practical considerations. The soft relaxation can be interpreted as replacing hard labels with soft labels, yielding smooth gradients as in knowledge distillation [79]. To prevent synthetic records from clustering too closely to a few microdata samples, we further propose the D-BCE Norm KL, which measures diversity by computing the KL divergence between the aggregated softmin weights and a uniform distribution (Algorithm 1, Line 12). 

D-BCE offers two advantages over standard BCE in population synthesis. First, it permits generated households to resemble real households at the joint-distribution level while still allowing marginals to shift during alignment. Second, it naturally accommodates $N _ { t } \neq N$ , enabling tract-level synthesis from PUMS samples of different sizes. As with BCE, higher D-BCE values indicate poor alignment with microdata, while excessively low values suggest overfitting. In practice, effective finetuning maintains D-BCE at the same order of magnitude as pre-training, ensuring both fidelity and diversity in the generated households. 

# 4 . Experiment

# 4.1 . Data

This study utilized ACS microdata [16] and ACS Census Data Tables (census tables) [17] for both training and testing purposes, focusing on the data from the year 2021. We opted for ACS data over AHS due to its broader coverage and granularity. AHS is limited to data from approximately 100,000 housing units across only 35 metro areas and selected states, whereas ACS encompasses around 3.5 million addresses annually and offers information at national, state, and county levels, down to the tract and block group levels [80]. Consequently, developing a synthetic population based on ACS data enhances the generalizability of our findings to wider geographic regions. 

As shown in Table 1, for households, we included the following variables: TEN (Tenure), HINCP (Household Income), R18 (Presence of Persons Under 18 Years in the Household), R65 (Presence of Persons 65 Years and Over in the Household), HHL (Household Language), and VEH (Vehicles Available). For individual persons, the variables considered were AGEP (Age), SEX (Sex), and SCHL (Educational Attainment). This selection of variables aims to showcase the performance of the proposed model by encompassing various types of data. Specifically, we intentionally selected household attributes such as R18 and R65, as well as the individual’s age, to assess the model’s performance, as detailed in Sections 5.3 and 6. Depending on the intended application of this synthetic inventory, the list can be expanded accordingly. 

In this paper, continuous variables such as income are discretized into categorical bins for both data-driven and methodological reasons. The primary motivation is to ensure consistency with the ACS Census Data Tables, which report tract-level targets (e.g., income, age) in categorical form. To align synthetic populations with these targets during fine-tuning, generated variables must match this granularity, making discretization a necessary pre-processing step. Methodologically, prior work shows that discretization combined with embedding improves deep learning performance on tabular data by enabling flexible, high-dimensional representations [21,81]. While this increases dimensionality and omits explicit ordinal encoding, it allows the model to capture complex, non-linear interactions unconstrained by fixed orderings. Our experiments confirmed this advantage: rank-aware loss functions (e.g., CRPS) did not outperform the discretization-based approach. Following discretization, categorical variables are transformed via one-hot encoding into binary features for model compatibility [82] (Fig. 7). 

We evaluate the approach using Delaware as the study site $\textbf { \em N } =$ 18, 641 households) and North Carolina as the transfer test site $\textbf {  { N } } =$ 


Table 1 Household and individual attributes.


<table><tr><td colspan="2">Household</td><td colspan="2">Individual</td></tr><tr><td>Variable</td><td>Description</td><td>Variable</td><td>Description</td></tr><tr><td>TEN</td><td>Tenure (Owned, Rented)</td><td>SEX</td><td>Sex (Male, Female)</td></tr><tr><td>VEH</td><td>Vehicles Available (No vehicle available, 1 vehicle, 2 vehicles, 3 vehicles, 4 or more vehicles)</td><td rowspan="3">AGEP</td><td rowspan="3">Age (Under 5, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80-84, 85 and over)</td></tr><tr><td>R18</td><td>Presence of Persons Under 18 Years in the Household (Yes, No)</td></tr><tr><td>R65</td><td>Presence of Persons 65 Years and Over in the Household (Yes, No)</td></tr><tr><td>HHL</td><td>Household Language (English only, Spanish, Other Indo-European languages)</td><td rowspan="2">SCHL</td><td rowspan="2">Educational Attainment (NA, Less than high school graduate, High school graduate (or equivalency), Some college or associate&#x27;s degree, Bachelor&#x27;s degree, Graduate or professional degree)</td></tr><tr><td>HINCP</td><td>Household Income (Less than $5,000, $5,000 to $9,999, $10,000 to $14,999, $15,000 to $19,999, $25,000 to $34,999, $35,000 to $49,999, $50,000 to $74,999, $75,000 to $99,999, $100,000 to $149,999, $150,000 or more)</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/37aa7858c6469b99f7cac10d9dfb5b551a2adbbdb7557ad87d494e69934c67cb.jpg)



Fig. 7. An illustrative example of one-hot encoding.


198, 037 households). Delaware, a small coastal state, presents moderate distribution shifts with rarer categories (e.g., minority language, high income, no-vehicle households), making tract-level alignment under class imbalance a relevant test for equity-focused policy contexts. North Carolina, by contrast, exhibits stronger PUMA-to-tract shifts, urban–rural heterogeneity, and longer-tailed distributions (e.g., language, vehicle availability, age–income cross-categories), providing a more challenging setting. Together, these sites bracket moderate and severe distribution-shift regimes, enabling evaluation of our fine-tuning strategy for both interpretability (Delaware) and robustness/transferability (North Carolina). 

# 4.2 . Model setup

One-hot encoding ensures the input data is in a consistent format for pre-training, but it also results in sparse feature representations, with a higher number of 0s than 1s, as illustrated in Fig. 7. This imbalance poses challenges to traditional BCE, particularly in reconstructing the minority class during pre-training. Our preliminary experiments also confirmed this, showing that traditional BCE made the model hard to converge. This challenge can be addressed by adopting the focal loss (FL) technique [83], as described in Eq. (8). FL enhances traditional BCE by incorporating a modulation factor that reduces the loss assigned to well-classified examples, enabling the model to focus more on difficult-to-classify samples. 

$$
\begin{array}{l} \operatorname {F L} \left(x _ {i, j}, \hat {x} _ {i, j}\right) = - \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {D} \left[ \alpha \hat {x} _ {i, j} \left(1 - x _ {i, j}\right) ^ {\gamma} \log x _ {i, j} \right. \\ \left. + (1 - \alpha) \left(1 - \hat {x} _ {i, j}\right) x _ {i, j} ^ {\gamma} \log \left(1 - x _ {i, j}\right) \right] \tag {8} \\ \end{array}
$$

Similar to BCE (Binary Cross-Entropy), when $\log x$ reaches $\log 0 .$ PyTorch clips this value to be no less than −100 to ensure numerical 

stability. $\alpha$ is a weighting parameter ranging from 0 to 1, used to balance positive and negative samples. Its value is determined by the ratio of 0s to 1s in the dataset. ?? controls the influence of the modulation factor, with a larger ?? making the model focus more on difficult-to-classify samples, and vice versa. When $\gamma = 0$ , the focal loss is equivalent to traditional BCE. The other parameters remain the same as in Eq. (5). Focal loss has proven effective in facilitating model pre-training during our experiments. Therefore, we use the FL for pre-training and D-BCE for fine-tuning in this study. 

We train with the Lion optimizer [84], using an initial learning rate of 0.001, full-batch updates, and 4000 epochs. From epoch 1000, the learning rate decays exponentially to a floor of 0.0001. To mitigate overfitting, we monitor SRMSE on the training set (avoiding held-out data to prevent leakage) along with nearest-sample distance (NSD), $\alpha$ -precision, and $\beta$ -recall. The checkpoint minimizing SRMSE on PUMS-derived distributions is retained. Experiments were conducted on a workstation with an NVIDIA GeForce RTX 4060 Ti (8GB) GPU and a 12th Gen Intel i7-12700 F CPU. 

We use Optuna for black-box hyperparameter tuning, combining principled defaults with its HEBO sampler. During pre-training, we tune the learning rate on a log scale [1e-5, 1e-2] with the TPE (Treestructured Parzen Estimator) sampler, while focal parameters are set by data statistics $\mathbf { \Delta } _ { \alpha }$ from positive rates, $\gamma ~ = ~ 6 )$ . During fine-tuning, we optimize the learning rate, ??, and the three margin weights using HEBO with successive halving and early stopping based on tract-level Scaled Root Mean Square Error against official aggregates. The resulting settings generalize across tracts and both states in our experiments. 

# 4.3 . Synthetic data evaluation metrics

We evaluate the framework along three complementary dimensions: (1) alignment of columnwise marginals to the ACS target distribution, (2) preservation of pairwise (2D) joint structure, and (3) sample-level fidelity, diversity, and generalizability assessed through manifold analysis. 

# 4.3.1 . Statistical distribution metrics

To assess column-level alignment between synthetic and target distributions, we report marginal metrics. The primary measure is the Total Variation Complement (TVComplement), derived from the Total Variation Distance (TVD). TVD between real $( R )$ and synthetic $( S )$ distributions is 

$$
\delta (R, S) = \frac {1}{2} \sum_ {\omega \in \Omega} | R _ {\omega} - S _ {\omega} |. \tag {9}
$$

where $\omega$ represents a category within the set of all possible categories $\Omega ,$ and $R _ { \omega }$ and $S _ { \omega }$ are the respective frequencies in the real and synthetic data. The TVComplement is defined as $1 - \delta ( R , S )$ , where a score of 1 indicates perfect alignment and 0 indicates maximal divergence. 

Root Mean Squared Error (RMSE) measures the differences between predicted and actual marginal distribution values. Lower RMSE values indicate better alignment between the synthetic and target marginal distributions. 

$$
\mathrm {R M S E} = \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} \left(x _ {i} - \hat {x} _ {i}\right) ^ {2}} \tag {10}
$$

where $n$ is the number of categories in the marginal distribution, $x _ { i }$ is the actual proportion in category ??, and $\hat { x } _ { i }$ is the predicted proportion. 

Kullback-Leibler (KL) divergence quantifies how one marginal probability distribution diverges from another one. The model with the lower KL divergence is considered to be a better approximation of the true distribution. A KL divergence score of zero indicates that the two distributions are identical. 

$$
D _ {K L} (\hat {\mathbf {X}} | | \mathbf {X}) = \sum_ {i} \left(\hat {x} _ {i} + \epsilon\right) \log \left(\frac {\hat {x} _ {i} + \epsilon}{x _ {i} + \epsilon}\right) \tag {11}
$$

where $\hat { \mathbf { X } }$ is the distribution of the synthetic population, and ?? is the ground truth marginal distribution. $x _ { i }$ and $\hat { X _ { i } }$ follows the same definition in RMSE. $\epsilon$ is a small positive value, often representing an error or tolerance. 

# 4.3.2 . Statistical pairwise distribution metrics

A contingency table (cross-tabulation) summarizes the joint frequency distribution of two categorical variables (e.g., income and age). We compute normalized contingency tables for real and synthetic data and compare the percentage of observations in each cell. Contingency Similarity quantifies how well the synthetic data preserves pairwise joint distributions: 

$$
\text {C o n t i n g e n c y S i m i l a r i t y} = 1 - \frac {1}{2} \sum_ {\alpha \in A} \sum_ {\beta \in B} \left| S _ {\alpha , \beta} - R _ {\alpha , \beta} \right|, \tag {12}
$$

where $\alpha$ and $\beta$ denote categories of $A$ and $B$ , and $R _ { \alpha , \beta }$ , $S _ { \alpha , \beta }$ are the corresponding frequencies. A score of 1 indicates perfect preservation. 

# 4.3.3 . Probability density distribution metrics

While distributional alignment is essential, it does not guarantee that individual synthetic samples are realistic. A model may perfectly match marginals yet generate implausible records (e.g., a 10-year-old with a Ph.D.), indicating a failure to capture the dependence structure of the data. Since the goal is to generate novel but realistic households and individuals, we require sample-level metrics beyond aggregate statistics. 

We adopt three metrics from deep generative modeling that evaluate fidelity, diversity, and generalizability by analyzing the probability density distribution of the data [85]. These provide a rigorous, record-centric assessment of whether the model has learned the true data-generating process. 

Generalizability measures the extent to which the model generates novel samples rather than memorizing the training set. It quantifies the proportion of synthetic records that are distinct from, yet statistically consistent with, the real data. High scores indicate that the model captures underlying patterns and produces authentic new samples, which is critical both for privacy and for ensuring that the synthetic population is not a replica of the source. 

$$
\text {G e n e r a l i z a b i l i t y} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {1} \left\{d \left(\mathbf {x} _ {i}, D _ {\text {r e a l}}\right) > d _ {i ^ {*}} \right\} \tag {13}
$$

where $N$ is the number of synthetic samples, $d ( \mathbf { x } _ { i } , D _ { r e a l } )$ denotes the distance between a synthetic sample $\mathbf { x } _ { i }$ and its nearest neighbor in the training dataset $\mathcal { D } _ { r e a l } , d _ { i ^ { * } }$ $d _ { i ^ { * } }$ is the distance between that nearest training sample and its own nearest neighbor in the remaining training data. The indicator function $\mathbf { 1 } \{ \cdot \}$ determines whether a synthetic sample is authentic. A synthetic sample is considered authentic if its distance from the nearest training sample exceeds the distance of that training sample to its own nearest neighbor. This metric is especially crucial for modeling sensitive data with privacy constraints, as it helps identify whether the model truly generates new samples rather than memorizing training data, avoiding scenarios where high fidelity and diversity scores mask the lack of genuine sample synthesis. 

$\alpha$ -Precision evaluates the fidelity of synthetic samples by determining the likelihood that a synthetic sample falls within the $\alpha$ -support of the real distribution. The $\alpha$ -support refers to the smallest volume subset of the real data distribution that contains the most representative $\alpha$ fraction of real samples. 

$$
\alpha \text { - P r e c i s i o n } = \frac { | \{ \mathbf { x } \in G : \operatorname* { m i n } _ { \mathbf { y } \in R } d ( \mathbf { x } , \mathbf { y } ) <   \alpha \} | } { | G | } \tag {14}
$$

where $G$ denotes the set of generated synthetic samples, ?? represents the set of real samples from the microdata, $d ( \mathbf { x } , \mathbf { y } )$ is the distance metric used to compare samples, and $\alpha$ is a threshold parameter defining what qualifies as a realistic sample. This metric quantifies the proportion of 

synthetic samples that closely align with the microdata. The reported score is an integrated value over all thresholds $\alpha \in [ 0 , 1 ]$ , providing a comprehensive measure of fidelity. 

$\beta$ -Recall evaluates the diversity of a generative model by measuring how well the synthetic distribution represents the variability in the real microdata. It quantifies the proportion of real samples included within the most representative $\beta$ fraction of synthetic samples: 

$$
\beta \text { - R e c a l l} = \frac {\left| \left\{\mathbf {y} \in R : \min  _ {\mathbf {x} \in G} d (\mathbf {x} , \mathbf {y}) <   \beta \right\} \right|}{| R |} \tag {15}
$$

where $\beta$ is a threshold parameter defining the coverage criterion. This metric measures the fraction of real samples effectively represented by the synthetic data, indicating how accurately the synthetic population captures the diversity inherent in the microdata. Similar to precision, the reported recall is an integrated score over all thresholds $\beta \in [ 0 , 1 ]$ . 

A high $\beta$ -Recall suggests that the generative model captures the full range of variability in the real data, while a low score may signal issues like mode collapse, where the model fails to generate certain types of data. By evaluating samples within different density levels ( $\beta$ -supports) of the synthetic distribution, this metric offers more granular diagnostics of the model’s ability to cover the microdata distribution, providing deeper insights than conventional recall measures. 

# 4.3.4 . Structural and sampling zeros

In synthetic data evaluation, particularly for categorical variables, it is useful to distinguish between structural zeros and sampling zeros [86]. Structural zeros correspond to impossible attribute combinations in the real world, whereas sampling zeros are plausible combinations that are absent from the training data due to limited sample size. These concepts relate directly to our fidelity and diversity metrics: 

• Fidelity and Structural Zeros: Generating structural zeros produces unrealistic records. Our $\alpha$ -Precision metric captures this by measuring the proportion of synthetic samples that lie close to the real PUMS microdata manifold. High $\alpha$ -Precision indicates that the model avoids impossible combinations, maintaining high fidelity. 

• Diversity and Sampling Zeros: A generative model should produce plausible but previously unseen samples, effectively “filling in” the sampling zeros. $\beta$ -Recall quantifies the coverage of the real data’s diversity by the synthetic data. High $\beta$ -Recall indicates that the model captures the underlying distribution rather than merely memorizing the training set. 

While structural and sampling zeros are particularly relevant for categorical data, $\alpha$ -Precision and $\beta$ -Recall provide a more general framework applicable to continuous, discrete, and categorical variables alike, making them well-suited for complex, mixed-type data in population synthesis. 

# 5 . Population synthesis performance evaluation

# 5.1 . Experimental design

A rigorous evaluation of the proposed population synthesis framework requires a ground truth dataset to assess its performance. However, since no real population dataset exists for this purpose, precisely the motivation for this research, we create a synthetic ground truth for validation. Using microdata from the ACS, which provides detailed population characteristics, we engineer a ground truth. Specifically, we treat the ACS microdata as the total population of a hypothetical area and randomly sample $5 \%$ of it to serve as the microdata for this area. The goal is to use this $5 ~ \%$ sample as training data to generate a synthetic population of the same size, matching the marginal distribution of the total population. We then compare the synthetic population to the area’s total population, namely the microdata, to evaluate the performance of the population synthesis framework. 

This experimental design enables us to: (1) simulate real-world scenarios where researchers need to generate a complete synthetic population using only the limited $5 ~ \%$ microdata samples. By training on a sampled $5 ~ \%$ microdata, generating a full population, and comparing it against the full microdata, we can evaluate the method’s performance under practical constraints; and (2) assess how well the synthetic data captures the underlying distribution characteristics of the true population, such as density centers and feature space coverage. Although we assume here that microdata can represent a complete population for an area (e.g., population distribution may differ in a real-world area), no other large population dataset is available that provides detailed information on households and individuals. Without this controlled experiment, we would be unable to evaluate critical aspects of the synthetic population’s quality, such as its ability to represent the full range and density of population characteristics in the feature space from a deep learning perspective. To ensure robust validation, all experiments were repeated with 10 different random seeds, and we report both mean values and standard deviations for all metrics. 

Engineered ground truth rationale and limitations. Individual-level ground truth is not publicly available in the real world due to privacy restrictions; only aggregate tract- or block-group statistics are accessible. To assess record-level realism and dependence structure learning, we treat the full PUMS microdata as a pseudo-population and train on a $5 \ \%$ subsample, reserving the remainder for evaluation. This self-similarity/hold-out protocol enables computation of sample-centric metrics—Generalizability, $\alpha$ -Precision, $\beta$ -Recall, and structural/sampling zeros—that go beyond marginal-only measures. We note that PUMS microdata are a probability sample subject to sampling error, disclosure avoidance, and PUMA-level aggregation; thus, this evaluation serves as a controlled comparative testbed rather than a fairness audit. For real-world applications with only aggregate data, evaluation is limited to tract-level alignment and statistical tests. 

Our protocol relies on the self-similarity principle: subsets statistically resemble the whole. This concept underpins fractal analysis [87], long-range dependence in network traffic [88], and methods in non-local means and single-image super-resolution [89,90]. Similarly, using a random microdata subsample for training and the remainder for evaluation provides a scale-consistent test of whether learned multivariate dependencies generalize from subsets to the whole, while tract-level marginals are enforced during fine-tuning. 

# 5.2 . Comparative baselines

# 5.2.1 . Iterative proportional updating (IPU)

We use Iterative Proportional Updating (IPU) as the baseline method to compare against our proposed transfer learning-enabled deep generative framework. IPU is a foundational technique for synthetic population generation and has been widely adopted in fields such as social science, transportation engineering, and urban planning. Its popularity stems from its simplicity, ease of implementation, and ability to handle both household- and individual-level attributes simultaneously [91]. Building on the well-established mathematical principles of iterative proportional fitting (IPF), IPU addresses the limitations of IPF in managing hierarchical data structures [20]. 

The IPU algorithm operates through the following steps [20]: (1) Initialization: All household in the sample population are assigned equal weights to start the iterative process; (2) Weight Adjustment: For each control variable (i.e., variables with known marginal distributions at both household and individual levels), the weights of households are updated by multiplying them by the ratio of the target control total to the current weighted sum. The weights are then normalized to ensure that the total population size remains consistent. The update can be formulated as 

$$
w _ {h} ^ {k + 1} = w _ {h} ^ {k} \cdot \frac {c _ {i}}{\sum_ {h \in H} w _ {h} ^ {k} n _ {h i}} \tag {16}
$$

where $w _ { h } ^ { k + 1 }$ represents the weight of household $h$ at iteration ??, $c _ { i }$ denotes the target control total for attribute ??, and $n _ { h i }$ indicates the number of individuals in household $h$ possessing attribute ??. (3) Iteration: The weight adjustment step is repeated until the error between the target and generated distributions falls below a specified threshold, or the maximum number of iterations is reached. In our experiments, we used the R package ipfr [92] to implement the above IPU algorithm. 

# 5.2.2 . Copula-based transferable models

We also benchmark our method against the copula-based transferable framework proposed by Jutras-Dubé et al. [68], which combines copula theory with generative machine learning models to decouple dependency structure learning from marginal distribution alignment. The copula-based approach proceeds in three main steps: (1) Copula Normalization: Source population data are transformed via empirical cumulative distribution functions (ECDFs) into uniform distributions on $[ 0 , 1 ] ^ { d }$ ; (2) Dependency Learning: A generative model (e.g., Bayesian Network, CTGAN, or TVAE) is trained on the normalized data to learn the copula function representing the dependency structure; (3) Population Generation: Synthetic samples are drawn from the learned copula and transformed back to the original space using the inverse CDFs of the target population’s marginal distributions [68]. 

Formally, by Sklar’s theorem, any joint distribution with continuous margins can be factorized as 

$$
F _ {\mathbf {X}} (\mathbf {x}) = C \left(F _ {1} \left(x _ {1}\right), \dots , F _ {d} \left(x _ {d}\right)\right), \tag {17}
$$

where $C$ is a copula and $F _ { j }$ are marginal CDFs. Enforcing tract-level marginals corresponds to replacing $F _ { j }$ while keeping $C$ fixed. 

Copula limitations and adapted baseline. Our household synthesis relies on fixed-width, person-specific columns (e.g., p1_age, p2_age) to model intra-household dependencies. These columns are artifacts of restructuring and do not have marginals, making standard copula methods structurally incompatible, which require well-defined column-wise distributions. 

To enable a meaningful comparison, we adapted the copula workflow. We first fit copula scalers on the source microdata (householdand individual-level) and developed a fit_from_marginals method to construct target copula scalers directly from aggregate marginal distributions. The copula-transformed tables are reshaped into a wide-format household table and used to train Copula-BN, Copula-CTGAN, and Copula-TVAE with increased capacity and extended training to handle the high-dimensional data. Synthetic samples are generated in wide format, then converted back to long-format households and individuals, removing non-existent members. Finally, marginals are recalibrated to uniform distributions using an empirical CDF with step-aware interpolation for discrete variables, followed by an inverse transform via the target copula scalers to impose the desired tract-level marginals. 

This workflow preserves the core copula methodology while enabling a meaningful comparison using our restructured, multi-entity data. 

Even with adaptations, copula-based methods face structural and practical limitations: (1) discrete/mixed marginals are non-unique [65, 69] and produce piecewise-constant transforms, complicating optimization and inducing sensitivity to label encodings [66,93,94]; (2) high dimensionality degrades dependence estimation and increases modelselection burden [94,95]; (3) geographic stationarity assumptions for variable dependencies may fail when household composition shifts; and (4) household–individual constraints (e.g., age–role consistency) require post-hoc adjustments that may perturb calibration. 

In contrast, TL-VAE addresses these challenges end-to-end. By finetuning with differentiable marginal losses and D-BCE through a frozen decoder, it preserves sample-level realism while aligning aggregates. Parameter-efficient fine-tuning allows smooth, gradient-based adaptation without catastrophic forgetting, directly modeling complex dependencies across household and person-position attributes. Empirically, TL-VAE achieves lower RMSE/KL and higher joint-structure fidelity than the copula baseline. This approach bypasses the identifiability, discretization, and high-dimensional issues inherent to copula methods, providing robust, flexible generation of hierarchical tabular data. 

# 5.3 . Results

Tables 2 and 3 compare TL-VAE with IPU and three copula-based variants (BN, TVAE, CTGAN) [68]. Table 2 reports TVComplement for individual variables, averaged over 10 runs with different random seeds for statistical robustness. 

TL-VAE closely matches IPU in marginal alignment (0.9760 vs. 0.9767) and surpasses all copula variants (BN: 0.6603; TVAE: 0.9572; CTGAN: 0.9714). More importantly, TL-VAE exhibits substantially lower variability (TVComplement: 0.0049, KL: 0.0013, RMSE: 0.0025) than both IPU and copula methods, indicating stable, reliable performance. 

On joint structure, TL-VAE achieves the highest Contingency Similarity (0.9460), slightly exceeding IPU (0.9421) and substantially outperforming the copula baselines (BN: 0.4658; TVAE: 0.8725; CTGAN: 0.9042). This reflects fundamental modeling differences: TL-VAE’s latent space captures complex nonlinear dependencies holistically, while copula methods impose marginals separately from dependence structure, risking information loss [68,69]. IPU leverages empirical distributions but lacks TL-VAE’s flexibility. 

According to Table 3, the five models exhibit distinct trade-offs between fidelity to training distributions $\mathbf { \alpha } _ { . \alpha }$ -precision) and comprehensive population coverage ( $\boldsymbol { \beta }$ -recall). TL-VAE achieves $\beta$ -recall of $0 . 7 3 2 3 \pm$ 0.0284, substantially higher than IPU $( 0 . 4 3 0 7 \pm 0 . 1 2 5 8 )$ and most copula variants (BN: $0 . 0 6 3 4 \pm 0 . 1 1 5 5$ ; TVAE: $0 . 6 3 9 4 \pm 0 . 0 7 3 4 $ ), while maintaining $\alpha$ -precision $( 0 . 9 8 1 1 ~ \pm ~ 0 . 0 0 6 4 )$ ) nearly identical to IPU $( 0 . 9 8 2 7 \pm 0 . 0 0 7 3 )$ ). This balance indicates TL-VAE’s capacity to generate 


Table 2 Performance of TL-VAE, IPU, and Copula-based variants (BN, TVAE, CTGAN) on individual variables using TVComplement. All values are reported as mean $\pm$ one standard deviation over ten runs.


<table><tr><td rowspan="2"></td><td rowspan="2">Variable</td><td colspan="5">TVComplement (+)</td></tr><tr><td>TL-VAE</td><td>IPU</td><td>Copula-BN</td><td>Copula-TVAE</td><td>Copula-CTGAN</td></tr><tr><td rowspan="6">Household</td><td>Tenure (TEN)</td><td>0.9922±0.0055</td><td>0.9910 ± 0.0052</td><td>0.6323 ± 0.4183</td><td>0.5595 ± 0.4183</td><td>0.9917 ± 0.0048</td></tr><tr><td>Vehicles Available (VEH)</td><td>0.9805±0.0085</td><td>0.9624 ± 0.0388</td><td>0.7117 ± 0.1221</td><td>0.7079 ± 0.083</td><td>0.9319 ± 0.0154</td></tr><tr><td>Household Language (HHL)</td><td>0.9838±0.0020</td><td>0.9828 ± 0.0093</td><td>0.5836 ± 0.4122</td><td>0.5133 ± 0.4144</td><td>0.9659 ± 0.0042</td></tr><tr><td>Household Income (HINCP)</td><td>0.9972±0.0021</td><td>0.9523 ± 0.0100</td><td>0.8241 ± 0.0559</td><td>0.8088 ± 0.0549</td><td>0.8229 ± 0.0955</td></tr><tr><td>Persons ≤ 18-yrs (R18)</td><td>0.9849 ± 0.0086</td><td>0.9877±0.0216</td><td>0.5974 ± 0.229</td><td>0.6762 ± 0.2205</td><td>0.9366 ± 0.0352</td></tr><tr><td>Persons ≥ 65-yrs (R65)</td><td>0.9903±0.0067</td><td>0.9876 ± 0.0153</td><td>0.5598 ± 0.2910</td><td>0.5597 ± 0.2908</td><td>0.9738 ± 0.0188</td></tr><tr><td rowspan="3">Individual</td><td>Age (AGEP)</td><td>0.9539 ± 0.0143</td><td>0.9587 ± 0.0528</td><td>0.6767 ± 0.0067</td><td>0.6747 ± 0.0048</td><td>0.9921±0.0175</td></tr><tr><td>Education (SCHL)</td><td>0.9610 ± 0.013</td><td>0.9762 ± 0.0349</td><td>0.6376 ± 0.0013</td><td>0.6369 ± 0.0027</td><td>1±0.0001</td></tr><tr><td>Sex (SEX)</td><td>0.9629 ± 0.0153</td><td>0.9917 ± 0.0167</td><td>0.8060 ± 0.0014</td><td>0.8056 ± 0.0015</td><td>1±0.0000</td></tr></table>


Note: $( + )$ indicates higher is better. The best results are shown in bold. 



Table 3 Overall performance comparison among TL-VAE, IPU, and Copula-based variants (BN, TVAE, CTGAN). All metrics are reported as mean $\pm$ one standard deviation over ten runs.


<table><tr><td>Metrics</td><td>TL-VAE</td><td>IPU</td><td>Copula-BN</td><td>Copula-TVAE</td><td>Copula-CTGAN</td></tr><tr><td>TVComplement (+)</td><td>0.9760 ± 0.0049</td><td>0.9767 ± 0.0257</td><td>0.6603 ± 0.0819</td><td>0.9572 ± 0.0099</td><td>0.9714 ± 0.0072</td></tr><tr><td>KL (-)</td><td>0.0155 ± 0.0013</td><td>0.0318 ± 0.0509</td><td>5.6043 ± 1.9928</td><td>0.1129 ± 0.1605</td><td>0.0221 ± 0.0050</td></tr><tr><td>RMSE (-)</td><td>0.0421 ± 0.0025</td><td>0.0407 ± 0.0063</td><td>0.2221 ± 0.0802</td><td>0.0314 ± 0.0081</td><td>0.0420 ± 0.0054</td></tr><tr><td>Contingency Similarity (+)</td><td>0.9460 ± 0.0083</td><td>0.9421 ± 0.0510</td><td>0.4658 ± 0.1121</td><td>0.8725 ± 0.0216</td><td>0.9042 ± 0.0084</td></tr><tr><td>α-Precision (+)</td><td>0.9811 ± 0.0064</td><td>0.9827 ± 0.0073</td><td>0.4357 ± 0.2918</td><td>0.9560 ± 0.0123</td><td>0.9379 ± 0.0238</td></tr><tr><td>β-Recall (+)</td><td>0.7323 ± 0.0284</td><td>0.4307 ± 0.1258</td><td>0.0634 ± 0.1155</td><td>0.6394 ± 0.0734</td><td>0.8377 ± 0.0160</td></tr><tr><td>Generalizability (+)</td><td>0.6530 ± 0.0244</td><td>0.4030 ± 0.1102</td><td>0.0483 ± 0.0881</td><td>0.5744 ± 0.0560</td><td>0.7528 ± 0.0129</td></tr><tr><td>Structural Zeros Error Rate (-)</td><td>0.4478 ± 0.0602</td><td>0.0000 ± 0.0000</td><td>0.9457 ± 0.0645</td><td>0.7077 ± 0.0545</td><td>0.7603 ± 0.0102</td></tr><tr><td>Sampling Zeros Coverage Rate (+)</td><td>0.1718 ± 0.0304</td><td>0.0000 ± 0.0000</td><td>0.0417 ± 0.0512</td><td>0.2212 ± 0.0244</td><td>0.3649 ± 0.0101</td></tr></table>


Note: (-) indicates lower is better; $( + )$ indicates higher is better. The best results are shown in bold. 


plausible demographic patterns absent from the limited training data without sacrificing individual authenticity. 

Copula-CTGAN achieves the highest $\beta$ -recall $( 0 . 8 3 7 7 \pm 0 . 0 1 6 0 )$ but with much lower $\alpha$ -precision $( 0 . 9 3 7 9 \pm 0 . 0 2 3 8 )$ , reflecting a core limitation of copula-based methods: their modular separation of marginals and dependence structures introduces information loss, yielding broad coverage but unrealistic samples. In contrast, TL-VAE’s end-to-end framework preserves intrinsic dependencies, enabling faithful extrapolation to underrepresented groups, an essential feature for equitable emergency planning [96,97]. 

TL-VAE also achieves good structural zeros performance (0.4478), far surpassing all copula variants, while retaining strong sampling zeros coverage. Copula-based methods expand sampling zeros at the cost of inflated structural zero errors, underscoring their difficulty in simultaneously enforcing demographic constraints and broad coverage. TL-VAE’s ability to balance both demonstrates robust constraint awareness while maintaining diversity, making it particularly suited for applications requiring equitable and representative synthetic populations. 

# 6 . Case study in delaware and North Carolina

Our proposed deep generative TL-VAE method demonstrated superior performance across multiple metrics on the engineered population dataset. Building on this, we applied the population synthesis framework to two study sites: Delaware and North Carolina. A randomly selected real-world census tract in Delaware was used to evaluate the synthetic population’s performance, while North Carolina served as a test site to assess the model’s transferability. The microdata for Delaware includes $N = 1 8 , 6 4 1$ household samples, whereas North Carolina’s dataset comprises 198,037 household samples. Because only aggregate ground truth is available in practice, evaluations focus on tract-level marginal alignment and statistical tests rather than record-level accuracy. 

# 6.1 . Realism of synthetic population using pre-trained model

The pre-trained model aims to accurately capture statistical relationships among households and individuals, as well as interactions between individuals. We utilize the pre-trained VAE model to generate a synthetic population that is the same size as the microdata. The 

realism of the synthetic population is assessed by how closely the distributions derived from the microdata match the distributions of the synthetic data produced by the pre-trained model. This comparison is made for both individual attributes (e.g., household income, household language) and joint variables (e.g., household income-household language). 

We use both household and individual attributes to demonstrate the performance of the pre-trained model, as shown in Fig. 8. The bar plot illustrates a strong resemblance in marginal distributions between the microdata and synthetic data generated by the pre-trained model, indicating that the pre-trained model can effectively generate realistic synthetic household data that aligns well with the microdata. We further conducted a chi-square test to compare the two distributions (null hypothesis). Across all household and individual attributes, we obtained a $p$ -value $( P )$ greater than 0.9, suggesting that the null hypothesis is not rejected and there is no evidence to suggest a statistically significant difference between the marginal distribution from the pre-trained model and that of the microdata. 

We utilize the metrics listed in Section 4.3 to evaluate the performance of the pre-trained model. The results of these metrics are presented in Table 4. The “-” sign suggests that a smaller value (closer to 0) implies better performance. The results show that the pre-trained model achieves a KL divergence score near zero, indicating that the individual attributes of the synthetic population closely approximate those in the microdata. 

Additionally, we analyze the dependence between various attributes within a household to validate that the pre-trained VAE can accurately capture statistical relationships among household attributes. Fig. 9(a) illustrates the relationship between household language and household income in the microdata, while Fig. 9(b) displays the same relationship in the synthetic household data generated by the pre-trained VAE. Each box is colored based on the log-scaled value of the percentage. The log transformation is applied to highlight differences among small values and to moderate extremely large values. Without this transformation, the English-only rows would dominate the coloring scheme, making it difficult to visualize differences in the other categories. The close resemblance between the two colormaps further confirms the pre-trained VAE’s ability to produce realistic synthetic household data. We further 


Table 4 Performance of pre-trained model on individual attributes.


<table><tr><td rowspan="2"></td><td colspan="6">Household</td><td colspan="3">Individual</td><td rowspan="2">Mean</td></tr><tr><td>Tenure (TEN)</td><td>Vehicles Available (VEH)</td><td>Household Language (HHL)</td><td>Household Income (HINCP)</td><td>Persons ≤ 18-yrs (R18)</td><td>Persons ≥ 65-yrs (R65)</td><td>Age (AGEP)</td><td>Education (SCHL)</td><td>Sex (SEX)</td></tr><tr><td>- RMSE</td><td>0.0210</td><td>0.0198</td><td>0.0598</td><td>0.0164</td><td>0.0388</td><td>0.0129</td><td>0.0091</td><td>0.0200</td><td>0.0190</td><td>0.0241</td></tr><tr><td>- KL</td><td>0.0013</td><td>0.0095</td><td>0.0442</td><td>0.0177</td><td>0.0039</td><td>0.0003</td><td>0.0138</td><td>0.0081</td><td>0.0007</td><td>0.0111</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/eaec0da62290bb6fb76b8cbf1b666590304ddf0d78658de4d1a26a781277f679.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/239f0e90333494cabc803661511b874ba3e62ad95b7baf2bcc21c8cbe61de808.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/c9373f18fb206bc92d9105406e0b5188fa0b9dbd1053aa0f68e86c817b1bfe72.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/86117d679328c9c41704bffa0d150bed6d46f83f7eba1818d3c8b36128b9534d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/2eabaf6440f353e479eba5199a7ea00a0b312e85fd923f1795ac1fb64c07e441.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/ac9460c0d3dc62e3773caf7a797efe031daa7ac923728b084bbf74ab95fe0288.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/5f35b4538f4c2ece2d44a3e788202910eb65bc371327645fbd39dc2b460bdbcb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/475459a667ad4972196ffb53656ce7988cb6eed367f221fc07092f18bd6c6b6c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/40cc319f7377e2628d562cbba4721d4cc4281aa4cd22c874b0a047d9d0b26e32.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/a40c5f5e5b2e829b014165d0ae54b85967a25d23c63fd700789480b6bacdb819.jpg)



Fig. 8. Attribute distribution comparison between microdata and pre-trained VAE. (a)–(f) Household attributes, (g)–(i) Individual attributes.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/306389813e80676085242cd5aecadbb49a11adb5eae3def3c3dd9967877e6b11.jpg)



Fig. 9. Joint distribution between different synthetic population attributes. (a) Microdata (b) Pretrain.


conducted a chi-square test to compare the joint distribution of all 36 pairs of variables from the synthetic population with those in the microdata. The results yield a $p$ -value greater than 0.99 for all comparisons, 

suggesting that the null hypothesis is not rejected, and there is no evidence to indicate a statistically significant difference between the two joint distributions. 


Table 5 Performance of pre-trained model on joint attributes. (a) RMSE, (b) KL Divergence.


<table><tr><td></td><td>TEN</td><td>VEH</td><td>HHL</td><td>HINCP</td><td>R18</td><td>R65</td><td>AGEP</td><td>SCHL</td><td>SEX</td></tr><tr><td>TEN</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VEH</td><td>0.0161</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(a)</td></tr><tr><td>HHL</td><td>0.0419</td><td>0.0210</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HINCP</td><td>0.0141</td><td>0.0078</td><td>0.0107</td><td>—</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>R18</td><td>0.0348</td><td>0.0228</td><td>0.0484</td><td>0.0136</td><td>—</td><td></td><td></td><td></td><td></td></tr><tr><td>R65</td><td>0.0180</td><td>0.0166</td><td>0.0303</td><td>0.0102</td><td>0.0250</td><td>—</td><td></td><td></td><td></td></tr><tr><td>AGEP</td><td>0.0067</td><td>0.0041</td><td>0.0049</td><td>0.0021</td><td>0.0076</td><td>0.0088</td><td>—</td><td></td><td></td></tr><tr><td>SCHL</td><td>0.0139</td><td>0.0086</td><td>0.0125</td><td>0.0052</td><td>0.0178</td><td>0.0121</td><td>0.0037</td><td>—</td><td></td></tr><tr><td>SEX</td><td>0.0154</td><td>0.0166</td><td>0.0236</td><td>0.0102</td><td>0.0116</td><td>0.0203</td><td>0.0053</td><td>0.0119</td><td>—</td></tr><tr><td></td><td>TEN</td><td>VEH</td><td>HHL</td><td>HINCP</td><td>R18</td><td>R65</td><td>AGEP</td><td>SCHL</td><td>SEX</td></tr><tr><td>TEN</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VEH</td><td>0.0200</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td><td>(b)</td></tr><tr><td>HHL</td><td>0.0494</td><td>0.0604</td><td>—</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HINCP</td><td>0.0403</td><td>0.0747</td><td>0.0684</td><td>—</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>R18</td><td>0.0071</td><td>0.0247</td><td>0.0595</td><td>0.0337</td><td>—</td><td></td><td></td><td></td><td></td></tr><tr><td>R65</td><td>0.0041</td><td>0.0202</td><td>0.0478</td><td>0.0275</td><td>0.0126</td><td>—</td><td></td><td></td><td></td></tr><tr><td>AGEP</td><td>0.0226</td><td>0.0538</td><td>0.0516</td><td>0.0660</td><td>0.0458</td><td>0.0715</td><td>—</td><td></td><td></td></tr><tr><td>SCHL</td><td>0.0119</td><td>0.0305</td><td>0.0422</td><td>0.0525</td><td>0.0189</td><td>0.0146</td><td>0.1224</td><td>—</td><td></td></tr><tr><td>SEX</td><td>0.0019</td><td>0.0157</td><td>0.0252</td><td>0.0258</td><td>0.0014</td><td>0.0028</td><td>0.0188</td><td>0.0102</td><td>—</td></tr></table>

Similarly, we evaluated the performance of the pre-trained model by calculating the RMSE and KL divergence between the joint distribution of microdata and the synthetic population across all 36 pairs of variables. Table 5 shows consistently low RMSE and KL divergence scores, indicating that the joint distributions of the synthetic population closely approximate those in the microdata. For example, the RMSE for the joint distribution of age and household income between the microdata and the synthetic population is 0.0021. This indicates that the synthetic population’s distribution for this pair of variables deviates from the microdata by an average error of $0 . 2 1 ~ \%$ . While KL divergence has been employed to assess synthetic population performance, these evaluations often focus on specific household sizes and selected variable pairs [56]. Since our joint household-individual population development is highly unique, there are no available benchmarks for comparing KL divergence. Nevertheless, the low KL divergence value in Table 5(b) indicates that the joint variable distributions of the synthetic population closely match those in the microdata. 

According to the performance metrics summarized here, it is evident that the pre-trained model effectively captures relationships between different variables, generating synthetic data with minimal deviations from the microdata. These findings affirm the capability of the proposed VAE model during the pre-training stage to produce realistic synthetic household and individual records, achieving features 1, 2, & 3 of a realistic synthetic population. Therefore, we are assured of using this pre-trained model to produce synthetic household and individual data at the census tract level. 

# 6.2 . Synthetic population of census tracts using fine-tuned model

The objective of the fine-tuning step in the proposed deep generative pipeline is to shift the distribution that the synthetic population adheres to from the microdata distribution to the target marginal distribution at the census tract level. Fig. 10 shows the final results of the fine-tuned 

model. It is evident that attributes in the synthetic household-individual inventory (illustrated by the orange bar) significantly departed from those in the microdata (represented by the blue bar), while closely approximating the target marginal distribution (displayed by the yellow bar) at the census tract level. The chi-square test yielded a $p$ -value of 1 for all attributes, indicating that the generated synthetic householdindividual data from the fine-tuned model aligns accurately with the marginal distribution provided by the census table. This underscores the accuracy of the synthetic population and the effectiveness of the proposed transfer learning approach under the distribution shift. 

We further evaluate the performance of the generated synthetic household-individual inventory using the six statistical metrics listed in Section 4.3. To establish a comparison baseline, we utilize the marginal distribution observed in microdata. Microdata serves as our baseline because existing deep learning methods typically concentrate on generating synthetic individuals that conform to the marginal distribution observed in microdata. Consequently, the baseline performance is assessed by comparing the microdata’s marginal distribution with the target marginal distribution from the census table. As shown in Table 6, we demonstrate the enhancements offered by our proposed method in achieving a more realistic characterization of household-individual characteristics at the census tract level. For example, looking at Fig. 10, existing deep learning-based synthetic populations tend to overestimate the number of households with incomes over $^ { 1 5 0 \mathrm { k } }$ and underestimate those with incomes $7 5 { - } 1 0 0 \mathrm { k }$ because they rely on microdata for generation. In each category, we can see that our synthetic data substantially outperforms the baseline methods by aligning the synthetic data more accurately with the marginal distribution provided in the census table, enabling more precise estimations of these numbers. The close approximation of distributions at the census tract level indicates that our synthetic population is geographically realistic, fulfilling feature 4 of a realistic synthetic population. 


Table 6 Population synthesis performance of fine-tuned model.


<table><tr><td rowspan="2" colspan="2"></td><td colspan="6">Household</td><td colspan="3">Individual</td><td rowspan="2">Mean</td></tr><tr><td>Tenure</td><td>Vehicle Available</td><td>HH. Language</td><td>HH. Income</td><td>Persons ≤ 18-yrs</td><td>Persons ≥ 65-yrs</td><td>Age</td><td>Edu.</td><td>Sex</td></tr><tr><td rowspan="2">- RMSE</td><td>Baseline</td><td>0.0468</td><td>0.1088</td><td>0.0142</td><td>0.0574</td><td>0.0579</td><td>0.0471</td><td>0.0180</td><td>0.0469</td><td>0.0285</td><td>0.0473</td></tr><tr><td>TL-VAE</td><td>0.0021</td><td>0.0018</td><td>0.0011</td><td>0.0034</td><td>0.0024</td><td>0.0021</td><td>0.0068</td><td>0.0169</td><td>0.0057</td><td>0.0047</td></tr><tr><td rowspan="2">- KL</td><td>Baseline</td><td>0.0072</td><td>0.1508</td><td>0.0167</td><td>0.1338</td><td>0.0089</td><td>0.0046</td><td>0.0520</td><td>0.0425</td><td>0.0016</td><td>0.0465</td></tr><tr><td>TL-VAE</td><td>0.0000</td><td>0.0001</td><td>0.0000</td><td>0.0575</td><td>0.0000</td><td>0.0000</td><td>0.0166</td><td>0.0093</td><td>0.0001</td><td>0.0093</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/79dbe522f44cfe7fa93dde34aab679b869227fd06ffa5cb9dc86b564226177fe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/98f89f419dca20cefe9bba37a970effbb4ad8992773cd6ecdce12b7caddbefdd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/b6d7617b1fa69ef17d9ecf8c63beec9a808ed6dcddb561ab6722ed29f2fa15b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/38aed5fe62566a41b1c37c130b9e52598712ceb90d26b0241f25d634ef759ea1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/36f16d06f6ec261b31c54f541451f33515f2bb5da6640bee055c7c0bbfd794fd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/3481d824f3fbab5b9afc6164b96f33321478d51e5e2ba446eceda91a0224f4e1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/fe7f91be79f9975ed5675d7062e8c23a5314a19188533b5e8c39595841631865.jpg)



（g


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/1a6681493ddf2e8fb776cf4752ce9007f713f0b608cc63e90b6e8bfb30484631.jpg)



(h)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/5abd0d09f5850c440ec203389436ee0a3bab70d0895260c3b6b703616a6efbc3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/a3e3fbfd41dd328a82ea5f7aabedf7d9967b4720c8d83bda9804f5b0b0aabaf9.jpg)



Fig. 10. Distributions of generated household and individual attributes using the finetuned model for a randomly selected census tract in Delaware. (a)-(f) household attributes, (g)-(i) individual attributes.


# 6.3 . Transferability of the proposed deep generative population synthesis framework

Ideally, we aim for the proposed deep generative framework for population synthesis to have the flexibility to be applied across various locations. To assess its transferability, we tested the framework in North Carolina. With 198,037 households in the microdata for North Carolina, we utilized it for training purposes and subsequently generated a household-individual population dataset for a census tract in North Carolina. The results depicted in Fig. 11 exhibit a similar pattern to those presented in Fig. 10. Notably, the marginal distribution of the synthetic population (represented by the orange bar) differs significantly from that of the microdata (depicted by the blue bar), while closely matching the target marginal distribution at the census tract level (indicated by the yellow bar). The chi-square test conducted between the target marginal distribution from the census table and the synthetic population yields $p$ -values of 1 for most variables, except for R18 (persons below 18 years of age) $( p$ -value $\ c = 0 . 9$ ) and Sex $\scriptstyle { \dot { p } }$ -value $= ~ 0 . 8 7 $ ). This highlights the robust transferability of the proposed framework to other regions. 

# 6.4 . Population synthesis privacy

A key concern in data synthesis is safeguarding the privacy of the training dataset used to generate the synthetic data, aiming to prevent the disclosure of sensitive information, especially pertaining to human subjects. This becomes even more crucial when researchers intend to create a synthetic population using privately collected data. Fortunately, in this study, the household and individual data within the microdata released by the US Census Bureau have already been anonymized to uphold data privacy. Therefore, data privacy is not the primary concern in this specific case. Nevertheless, we anticipate that our population synthesis pipeline can be widely applicable in various scenarios, particularly in cases when the marginal distribution of the training data differs from the targeted marginal distribution of the synthetic data. In this context, 

we aim to ensure that the transfer learning procedure, particularly the fine-tuning step, does not compromise the privacy-preserving capability of the pre-trained model. 

We assess privacy using Distances to Closest Records (DCR). DCR identifies the record in the microdata that the generated synthetic record most closely resembles (i.e., has the least distance) and calculates the distance between records using simple BCE. Our objective is to ensure there is no statistically significant difference (at alpha $= 0 . 5$ ) between the results of the fine-tuned model and the pre-trained model. As shown in Fig. 12, the fine-tuning step maintains a similar level of privacy as the pre-trained model. Furthermore, we grouped the DCR values into different bins and performed the Kolmogorov-Smirnov (K-S) test to compare the differences between the distributions of households’ and individuals’ DCR. The resulting p-values for the household and individual data were 0.87 and 1.00, respectively, both exceeding the threshold of 0.05. This indicates no statistically significant difference between the DCR distributions of the pre-trained and fine-tuned models. That is to say, transferring learning effectively preserves privacy and does not worsen the privacy performance of the generative model. However, it is important to note that VAE is not the leading method in terms of privacy preservation capabilities compared to other generative methods such as AutoDiff [73]. In scenarios involving the use of private household survey data for synthetic population generation or employing the proposed pipeline in other sensitive data generation tasks, the generative module, VAE in this paper, can be replaced with other methods to meet privacy requirements. 

# 7 . Discussion

The proposed deep generative population synthesis framework enables the generation of records that extend beyond the microdata. While we aim to capture the joint distribution between attributes and closely align the aggregated distribution with the marginal distribution, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/debc18fc823f80253c69e9fa0ae75ac60ee3933d4c15e25390dcd6999593af40.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/85b42224a03f8b254deeece6f85b11ebe131401c26ef627b67fcd0f8be6d2d54.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/3c590c09ff3e64d5f3f93894815a351b9fbff4140d121d0868ead0bad9fcc98c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/4cca3520c6115377b9c55ad6425d5396ce78be511b362a0515af3859c0437eb8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/2ec64bac0e2cb4e80dccd6609263985b63e1e2174d6e759769341f5beb0400f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/eeef911660d5ef3fae0484b119af741d9ecc86c5fb446daeef0b5a6a8ee99ba4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/6ebad4bb3ae7b7b7ed8a7de3091b84bd2a1264a7e5ab3c65cdd194d4f86b32f5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/9b19d20fdf52f5fd87316ca25686c6c016956b920a1aca19e43381c597f007ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/cd52f63fa7f4920bd000dda5f1c4bffa60640b4ce757d9d1d674b629aac0439c.jpg)



Fig. 11. Distributions of generated household and individual attributes using the finetuned model for a randomly selected census tract in North Carolina. (a)–(f) household attributes, (g)–(i) individual attributes.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/ece229c0-3a3d-4c23-9360-939716e65498/570c944e9b81fd9875f6793d71ee3076450339532e1b97286b5e67fdd25c0dda.jpg)



Fig. 12. Population synthesis privacy assessment.


there still exist some discrepancies. Consequently, unrealistic synthetic records that deviate from reality may be generated. However, manually identifying these “unrealistic” records is laborious and challenging. 

Our validation relies on ACS microdata as a pseudo-ground-truth, enabling individual-level evaluation when true records are unavailable. Microdata is a sample subject to disclosure controls and may 

reflect historical or structural biases. Thus, our results should be read as comparative assessments of dependence learning, not fairness audits. Fairness is application-specific and often requires subgroup constraints beyond marginals. While our fine-tuning objective can incorporate such constraints as differentiable penalties, we leave this for future work with fairness-oriented diagnostics. 


Table 7 Unrealistic samples with inconsistent household characteristics and individual attributes.


<table><tr><td>IID</td><td>HHID</td><td>Tenure</td><td>Vehicles Available</td><td>Household Language</td><td>Household Income (thousands of USD)</td><td>Persons ≤ 18years</td><td>Persons ≥ 65years</td><td>Age (in years)</td><td>Educational</td><td>Sex</td></tr><tr><td>29</td><td>10</td><td>Renter</td><td>2</td><td>English only</td><td>$75,000 to $99,999</td><td>no</td><td>no</td><td>25 to 29</td><td>college or associate</td><td>Male</td></tr><tr><td>30</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>30 to 34</td><td>High school</td><td>Female</td></tr><tr><td>31</td><td>11</td><td>Owner</td><td>3</td><td>English only</td><td>$50,000 to $74,999</td><td>no</td><td>yes</td><td>40 to 44</td><td>college or associate</td><td>Female</td></tr><tr><td>32</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>45 to 49</td><td>High school</td><td>Male</td></tr><tr><td>33</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>15 to 19</td><td>High school</td><td>Male</td></tr><tr><td>34</td><td>12</td><td>Owner</td><td>4 or more</td><td>English only</td><td>$100,000 to $149,999</td><td>no</td><td>no</td><td>50 to 54</td><td>Bachelor</td><td>Female</td></tr><tr><td>35</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>55 to 59</td><td>High school</td><td>Male</td></tr></table>

We use discretized versions of variables that are continuous in principle (e.g., income, age). Binning sacrifices within-bin detail but is necessary because ACS tract-level control totals are available only in categorical form, requiring exact categorical support for alignment. Despite this restriction, our results show that the model captures complex dependencies, preserves pairwise structure, and achieves strong sample-level fidelity and diversity in both Delaware and North Carolina. 

To assess the consistencies between each household and the individuals it includes in the generated synthetic population (Feature 2), we specifically kept some variables in both household and individual attributes in our test. For example, we included the presence of individuals under 18 years (R18) or 65 years and above (R65) in the household, as well as the individual’s age. Intuitively, an individual’s age $\ge 6 5$ would correspond to the household attributes R65 being flagged as 1. Otherwise, this record would indicate faulty generated records. Applying this criterion, we conducted a sanity check of the generated synthetic household-individual inventory (Table 7). In the census tract highlighted in Fig. 10, we identified 158 households $( 1 1 ~ \% )$ with contradictory attributes (R65 flagged in the household, but lacking individuals aged $\geq$ 65 years) in synthetic household-individual data, indicating inconsistent records. This suggests that, although we can achieve decent realism at the distribution level by achieving low RMSE and KL divergence of both individual attributes and joint variables (Figs. 8 and 9, Tables 4 and 5), record-level realism requires extra attention. This limitation is inherent in all deep generative methods, yet it is seldom discussed and reported in existing studies. Prior research often prioritizes distribution accuracy over record-level correctness [21,22,52,57]. However, the accuracy of household records is crucial for subsequent tasks such as equity assessment and household decision-making analysis. We intentionally retained these records as they serve as both an indicator of the framework’s advantage (i.e., generating data beyond microdata) and its limitation (i.e., producing faulty records). The record-level examination does not detract from the methodological contributions presented in this paper. Instead, it highlights a direction for future population synthesis research to enhance. To further enhance the proposed framework and minimize detectable unrealistic records, we can leverage human expertise during the generation phase. For instance, we can employ human-in-the-loop learning techniques [98], to identify and label ruleviolating records. Subsequently, the learning algorithm can be informed to avoid generating such records, thus reducing the rate of unrealistic records. 

Furthermore, in this paper, only nine attributes are included to evaluate the proposed framework. However, after recategorizing each variable, applying one-hot embedding, and restructuring the data with fifteen individuals per record, each entry already spans a window size of 462 columns. In practice, we may need to incorporate more household and individual attributes to better characterize household decisionmaking or equity assessment. Increasing the number of attributes will significantly grow the size of the training data, which requires highperformance computing resources and poses a challenge to the generative power of the deep learning model that is beyond what VAE can offer. Therefore, in future research, we plan to explore more efficient data restructuring methods while still capturing household-individual 

and individual-individual relationships. Additionally, we will consider replacing VAE with more powerful generative models such as transformers [99]. A stronger generative backbone method not only enables the handling of large volumes of data and attributes but also improves the realism of synthetic records. 

# 8 . Conclusion

This paper introduces a deep generative framework for synthetic population development. It leverages microdata, which consists of anonymized samples of real households and individuals at the PUMA level, along with marginal distributions (representing the distribution of each attribute at the census tract level) to create a diverse inventory of households with individuals embedded. We aim to ensure this synthetic population is realistic in two main aspects: (1) the marginal distribution of household characteristics (e.g., income, tenure) and individual attributes (e.g., age, education) aligns with the target marginal distribution provided by ACS census data tables at the census tract level; (2) the relationships between household characteristics and individual attributes, as well as dependencies between individuals, accurately reflect those described in the microdata. 

We employ the Variational Autoencoder (VAE) for synthetic population generation (Fig. 5). It allows for the generation of out-of-sample records, enhancing population diversity, as opposed to simply weighing and cloning microdata samples. This deep generative framework presents three methodological contributions aimed at addressing corresponding challenges in existing synthetic populations. Firstly, we introduce a data restructuring scheme (Fig. 3) that captures not only relationships between household and individual attributes but also relationships among individuals within a household. This approach overcomes the limitation of the existing two-step process, where individuals are first generated and then grouped into households, thus failing to capture household-individual relationships. Secondly, we propose a parameter-efficient transfer learning approach with VAE (Fig. 4) consisting of pre-training and fine-tuning. The pre-training step learns the joint distribution of household and individual characteristics in microdata, while the fine-tuning step generates households and individuals that fit a different distribution at the census tract level. This is in contrast to existing population generation that follows the same marginal distribution as the microdata, which is not realistic given the significant differences between the marginal distribution of microdata at the PUMA level and the marginal distribution at the census tract level (as seen in Fig. 1). Thirdly, we introduce a new loss function, Decoupled Binary Cross Entropy (D-BCE) (Fig. 6), which focuses on generating households and individuals similar to any record in the microdata, rather than strictly mirroring a specific record. This decoupling procedure relaxes one-to-one correspondences to one-to-many correspondences, enabling the aforementioned transfer learning process. 

We examined the framework using an engineered population as the ground truth. The results, presented in Tables 2 and 3, demonstrate strong performance in both marginal distribution alignment and synthetic data probability density distribution. These findings validate the effectiveness of our proposed deep generative approach for synthetic 

population generation. Following this, we tested in Delaware using six household attributes and three individual attributes (Table 1). The synthetic inventory yields promising results. The pre-trained model successfully captures relationships between households and individuals, as well as among individuals. Notably, the pre-trained VAE demonstrates strong performance across all employed metrics, ensuring the realism of the generated data in subsequent steps (Figs. 8 and 9, Tables 4 and 5). Moreover, the results from the fine-tuned model indicate our ability to generate a synthetic household-individual inventory that aligns with the marginal distribution of various attributes at the census tract level, as provided by ACS census data tables (Fig. 10). Additionally, we demonstrate that our proposed model outperforms existing deep-generated inventories (Table 6). To ensure the applicability of our framework to other regions, we tested it in North Carolina (Fig. 11), obtaining similarly promising results, thereby confirming the transferability of our methods. Lastly, recognizing the potential adoption of our method by studies dealing with sensitive human-subject information, we examined the privacy implications of our framework using Distance to Closest Record (DCR) (Fig. 12). The Kolmogorov-Smirnov (K-S) test indicates no statistically significant difference, affirming the privacy-preserving capability of our approach. 

Future research will continue enhancing the realism, privacy protection, and generative capabilities of population synthesis. This will involve exploring more robust and privacy-preserving deep generative backbone methods, while also incorporating a wider range of household and individual attributes. 

# CRediT authorship contribution statement

Xiao Qian: Writing – review & editing, Writing – original draft, Visualization, Validation, Supervision, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Utkarsh Gangwal: Writing – review & editing, Writing – original draft, Visualization, Formal analysis, Data curation, Conceptualization. Shangjia Dong: Writing – review & editing, Writing – original draft, Visualization, Validation, Supervision, Software, Resources, Project administration, Methodology, Investigation, Funding acquisition, Formal analysis, Data curation, Conceptualization. Rachel Davidson: Writing – review & editing, Writing – original draft, Visualization, Validation, Project administration, Funding acquisition, Formal analysis, Conceptualization. 

# Declaration of competing interest

The authors declare the following financial interests/personal relationships, which may be considered as potential competing interests: 

Shangjia Dong reports that financial support was provided by the National Science Foundation. Rachel Davidson reports that financial support was provided by the National Science Foundation. If there are other authors, they declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

# Acknowledgments

Shangjia Dong would like to acknowledge funding support from the National Science Foundation #2443784. Rachel Davison would like to acknowledge funding support from the National Science Foundation #2209190. Any opinions, conclusions, and recommendations expressed in this research are those of the authors and do not necessarily reflect the views of the funding agencies. The authors would also like to thank the editor and the anonymous reviewers for their constructive comments and valuable insights to improve the quality of the article. 

# Data availability

The data used are open-source. 

# References



[1] J.A. Maantay, A.R. Maroko, C. Herrmann, Mapping population distribution in the urban environment: the cadastral-based expert dasymetric system (ceds), Cartogr. Geogr. Inf. Sci. 34 (2) (2007) 77–102. 





[2] A. Sodiq, A.A. Baloch, S.A. Khan, N. Sezer, S. Mahmoud, M. Jama, A. Abdelaal, Towards modern sustainable cities: review of sustainability principles and trends, J. Clean. Prod. 227 (2019) 972–1001. 





[3] Y. Zhu, J. Ferreira Jr, Synthetic population generation at disaggregated spatial scales for land use and transportation microsimulation, Transp. Res. Rec. 2429 (1) (2014) 168–177. 





[4] J. Birkmann, B. Wisner, Measuring the Unmeasurable: the Challenge of Vulnerability, UNU-EHS, 2006. 





[5] C. He, Q. Huang, Y. Dou, W. Tu, J. Liu, The population in China-earthquake-prone areas has increased by over 32 million along with rapid urbanization, Environ. Res. Lett. 11 (7) (2016) 074028. 





[6] N. Soleimani, R.A. Davidson, J. Kendra, B. Ewing, L.K. Nozick, Household adaptations to and impacts from electric power and water outages in the Texas 2021 winter storm, Nat. Hazards Rev. 24 (4) (2023) 04023041. 





[7] J. Bouttell, P. Craig, J. Lewsey, M. Robinson, F. Popham, Synthetic control methodology as a tool for evaluating population-level health interventions, J. Epidemiol. Community Health 72 (8) (2018) 673–678. 





[8] U. Gangwal, A.R. Siders, J. Horney, H.A. Michael, S. Dong, Critical facility accessibility and road criticality assessment considering flood-induced partial failure, Sustain. Resil. Infrastruct. 8 (sup1) (2023) 337–355. 





[9] Z. Chen, X. Li, Unobserved heterogeneity in transportation equity analysis: evidence from a bike-sharing system in southern Tampa, J. Transp. Geogr. 91 (2021) 102956. 





[10] Congressional Research Service, Data protection and privacy law: an introduction (2022). https://crsreports.congress.gov/product/pdf/IF/IF11207 (Online; accessed 1 May 2024. 





[11] Global Legal Group, International comparative legal guides (2024). https://iclg. com/practice-areas/data-protection-laws-and-regulations/usa (Online; accessed 1 May 2024. 





[12] J. Verschuur, T. Tiggeloven, H. de Moel, J.C. Aerts, High-resolution synthetic population mapping for hazard and risk assessments, Front. Environ. Sci. 10 (2022) 1033579. 





[13] Z. Chen, Y. Liu, Y. Zhang, Y. Wang, L. Zhang, A. Plaza, Bright: A global, multiresolution, and multi-modal benchmark for building damage assessment in disaster scenarios, arXiv preprint arXiv:2501.06019, 2025. 





[14] N. Rosenheim, R. Guidotti, P. Gardoni, W.G. Peacock, Integration of detailed household and housing unit characteristic data with critical infrastructure for post-hazard resilience modeling, Sustain. Resil. Infrastruct. 6 (6) (2021) 385–401. 





[15] N. Fournier, E. Christofa, A.P. Akkinepally, C.L. Azevedo, Integrated population synthesis and workplace assignment using an efficient optimization-based personhousehold matching method, Transportation 48 (2) (2021) 1061–1087. 





[16] U.S. Census Bureau, American community Survey (ACS) public use microdata sample (pums) (2022). https://www.census.gov/programs-surveys/acs/microdata/ access.html (Online); accessed 1 May 2024. 





[17] U.S. Census Bureau, American Community Survey (ACS) census data tables (2022). https://data.census.gov/table (Online); accessed 1 May 2024. 





[18] R.J. Beckman, K.A. Baggerly, M.D. McKay, Creating synthetic baseline populations, Transp. Res. Part A Policy Pract. 30 (6) (1996) 415–429. 





[19] B. Farooq, M. Bierlaire, R. Hurtubia, G. Flötteröd, Simulation based population synthesis, Transp. Res. Part B Methodol. 58 (2013) 243–263. 





[20] X. Ye, K. Konduri, R.M. Pendyala, B. Sana, P. Waddell, A methodology to match distributions of both household and person attributes in the generation of synthetic populations, in: 88th Annual Meeting of the Transportation Research Board, Washington, DC, 2009. 





[21] S.S. Borysov, J. Rich, F.C. Pereira, How to generate micro-agents? a deep generative modeling approach to population synthesis, Transp. Res. Part C Emerg. Technol. 106 (2019) 73–97. 





[22] Z. Aemmer, D. MacKenzie, Generative population synthesis for joint household and individual characteristics, Comput. Environ. Urban Syst. 96 (2022) 101852. 





[23] Z. Zhao, A. Kunar, R. Birke, L.Y. Chen, Ctab-Gan: effective table data synthesizing, in: Asian Conference on Machine Learning, PMLR, 2021, pp. 97–112. 





[24] Z. Zhao, A. Kunar, R. Birke, L.Y. Chen, Ctab-gan+: Enhancing tabular data synthesis, arXiv preprint arXiv:2204.00401, 2022. 





[25] A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko, Tabddpm: modelling tabular data with diffusion models, in: International Conference on Machine Learning, PMLR, 2023, pp. 17564–17579. 





[26] C. Lee, J. Kim, N. Park, Codi: co-evolving contrastive diffusion models for mixedtype tabular synthesis, in: Proceedings of the 40th International Conference on Machine Learning, ICML’23, JMLR.org, 2023. 





[27] Y.-H. Li, On a built-in conflict between deep learning and systematic generalization (2022). 





[28] X. Zhang, R. Xu, H. Yu, Z. Shi, P. Cui, Deep stable learning for out-of-distribution generalization (2021). 





[29] B. Fabrice Yaméogo, P. Gastineau, P. Hankach, P.-O. Vandanjon, Comparing methods for generating a two-layered synthetic population, Transp. Res. Rec. 2675 (1) (2021) 136–147. 





[30] L. Sun, A. Erath, M. Cai, A hierarchical mixture modeling framework for population synthesis, Transp. Res. Part B Methodol. 114 (2018) 199–212. 





[31] D.R. Pritchard, E.J. Miller, Advances in population synthesis: fitting many attributes per agent and fitting to household and person margins simultaneously, Transportation 39 (3) (2012) 685–704. 





[32] R.J. Little, M.-M. Wu, Models for contingency tables with known margins when target and sampled populations differ, J. Am. Stat. Assoc. 86 (413) (1991) 87–95. 





[33] T. Arentze, H. Timmermans, F. Hofman, Creating synthetic household populations: problems and approach, Transp. Res. Rec. 2014 (1) (2007) 85–91. 





[34] K. Müller, K.W. Axhausen, Hierarchical ipf: generating a synthetic population for Switzerland, Arb.ber.Verk.-und Raumplan. 718 (2011). 





[35] K. Müller, A generalized approach to population synthesis, PhD thesis (2017). 





[36] R. Balakrishnaa, S. Sundaram, J. Lam, An enhanced and efficient population synthesis approach to support advanced travel demand models, population 18 (2019) 19. 





[37] K. Chapuis, P. Taillandier, A brief review of synthetic population generation practices in agent-based social simulation, in: Submitted to SSC2019, Social Simulation Conference, 2019. 





[38] S. Jain, N. Ronald, S. Winter, Creating a synthetic population: a comparison of tools, in: Proceedings of the 3rd Conference Transportation Reserch Group, Kolkata, India, 2015, pp. 17–20. 





[39] J. Barthelemy, P.L. Toint, Synthetic population generation without a sample, Transp. Sci. 47 (2) (2013) 266–279. 





[40] B.M. Paul, J. Doyle, B. Stabler, J. Freedman, A. Bettinardi, Multi-Level Population Synthesis Using Entropy Maximization-Based Simultaneous List Balancing. Technical report, 2018. 





[41] H. Wu, Y. Ning, P. Chakraborty, J. Vreeken, N. Tatti, N. Ramakrishnan, Generating realistic synthetic population datasets, ACM Trans. Knowl. Discov. Data 12 (4) (2018) 1–22. 





[42] D.-H. Lee, Y. Fu, Cross-entropy optimization model for population synthesis in activity-based microsimulation models, Transp. Res. Rec. 2255 (1) (2011) 20–27. 





[43] J.-C. Deville, C.-E. Särndal, O. Sautory, Generalized raking procedures in survey sampling, J. Am. Stat. Assoc. 88 (423) (1993) 1013–1020. 





[44] M. Templ, B. Meindl, A. Kowarik, O. Dupriez, Simulation of synthetic complex data: the r package Simpop, J. Stat. Softw. 79 (10) (2017) 1–38. 





[45] D. Voas, P. Williamson, An evaluation of the combinatorial optimisation approach to the creation of synthetic microdata, Int. J. Popul. Geogr. 6 (5) (2000) 349–366. 





[46] M. Grotschel, L. Lovász, Combinatorial optimization, Handbook of combinatorics 2 (1541–1597) (1995) 4. 





[47] Z. Huang, P. Williamson, A comparison of synthetic reconstruction and combinatorial optimisation approaches to the creation of small-area microdata, Department of Geography, University of Liverpool (2001). 





[48] S. Katoch, S.S. Chauhan, V. Kumar, A review on genetic algorithm: past, present, and future, Multimed. Tools Appl. 80 (2021) 8091–8126. 





[49] Y. Chen, M. Elliot, D. Smith, The application of genetic algorithms to data synthesis: a comparison of three crossover methods, in: Privacy in Statistical Databases: UNESCO Chair in Data Privacy, International Conference, PSD 2018, Valencia, Spain, September 26–28, 2018, Proceedings, Springer, 2018, pp. 160–171. 





[50] P. Williamson, M. Birkin, P.H. Rees, The estimation of population microdata by using data from small area statistics and samples of anonymised records, Environ. Plan. A 30 (5) (1998) 785–816. 





[51] M. Birkin, A. Turner, B. Wu, A synthetic demographic model of the UK population: methods, progress and problems, in: Second International Conference on E-Social Science, Citeseer, 2006, pp. 692–697. 





[52] I. Saadi, A. Mustafa, J. Teller, B. Farooq, M. Cools, Hidden markov model-based population synthesis, Transp. Res. Part B Methodol. 90 (2016) 1–21. 





[53] D. Casati, K. Müller, P.J. Fourie, A. Erath, K.W. Axhausen, Synthetic population generation by combining a hierarchical, simulation-based approach with reweighting by generalized raking, Transp. Res. Rec. 2493 (1) (2015) 107–116. 





[54] M.N. Rahman, M.R. Fatmi, Population synthesis accommodating heterogeneity: a Bayesian network and generalized raking technique, Transp. Res. Rec. 2677 (6) (2023) 41–57. 





[55] J. Young, P. Graham, R. Penny, Using Bayesian networks to create synthetic data, J. Off. Stat. 25 (4) (2009) 549–567. 





[56] D. Zhang, J. Cao, S. Feygin, D. Tang, Z.-J.M. Shen, A. Pozdnoukhov, Connected population synthesis for transportation simulation, Transp. Res. Part C Emerg. Technol. 103 (2019) 1–16. 





[57] L. Sun, A. Erath, A Bayesian network approach for population synthesis, Transp. Res. Part C Emerg. Technol. 61 (2015) 49–62. 





[58] J.K. Vermunt, Multilevel latent class models, Soc. Methodol. 33 (1) (2003) 213–239. 





[59] S. Kotnana, D. Han, T. Anderson, A. Züfle, H. Kavak, Using generative adversarial networks to assist synthetic population creation for simulations, in: 2022 Annual Modeling and Simulation Conference (ANNSIM), IEEE, 2022, pp. 1–12. 





[60] L. Xu, K. Veeramachaneni, Synthesizing tabular data using generative adversarial networks, arXiv preprint arXiv:1811.11264, 2018. 





[61] E. Arkangil, M. Yildirimoglu, J. Kim, C. Prato, A deep learning framework to generate realistic population and mobility data, arXiv preprint arXiv:2211.07369, 2022. 





[62] G. Lederrey, T. Hillel, M. Bierlaire, Datgan: integrating expert knowledge into deeplearning for population synthesis (2021). 





[63] S.S. Borysov, J. Rich, Introducing synthetic pseudo panels: application to transport behaviour dynamics, Transportation 48 (5) (2021) 2493–2520. 





[64] S. Choi, S. Kim, Y. Jeon, G. Lee, P. Bansal, A gentle introduction and tutorial on deep generative models in transportation research, Transp. Res. Part C Emerg. Technol. 176 (2025) 105145. 





[65] R.B. Nelsen, An Introduction to Copulas, 2 ed., Springer, New York, 2006. 





[66] A.K. Nikoloulopoulos, Copula-based models for multivariate discrete response data, In P. Jaworski, F. Durante, W.K. Härdle (Eds.), Copulae in Mathematical and 





Quantitative Finance, Volume 213 of Lecture Notes in Statistics, Springer, Berlin, Heidelberg, 2013, pp. 231–249. 





[67] K. Aas, C. Czado, A. Frigessi, H. Bakken, Pair-copula constructions of multiple dependence, Insur.: Math. Econ. 44 (2) (2009) 182–198. 





[68] P. Jutras-Dubé, M.B. Al-Khasawneh, Z. Yang, J. Bas, F. Bastin, C. Cirillo, Copulabased transferable models for synthetic population generation, Transp. Res. Part C Emerg. Technol. 169 (2024) 104830. 





[69] A. Sklar, Fonctions de répartition à n dimensions ET leurs marges, Publ. Inst. Stat. Univ. Paris 8 (1959) 229–231. 





[70] A.R. Sané, R. Belaroussi, P. Hankach, P.-O. Vandanjon, Population synthesis with deep generative model: a joint household-individual approach, Comput. Urban Sci. 5 (2025) 34. 





[71] H. Sun, T. Zhu, Z. Zhang, D. Jin, P. Xiong, W. Zhou, Adversarial attacks against deep generative models on data: a survey, IEEE Trans. Knowl. Data Eng. 35 (4) (2021) 3367–3388. 





[72] L. Xu, H. Xie, S.-Z.J. Qin, X. Tao, F.L. Wang, Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment, arXiv preprint arXiv:2312.12148, 2023. 





[73] N. Suh, X. Lin, D.-Y. Hsieh, M. Honarkhah, G. Cheng, Autodiff: combining auto-encoder and diffusion model for tabular data synthesizing, arXiv preprint arXiv:2310.15479, 2023. 





[74] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Lau, et al., Pytorch: an imperative style, high-performance deep learning library (2019) https://pytorch.org/docs/stable/generated/torch.nn. BCELoss.html. (Accessed on). 





[75] J.H. Friedman, L.C. Rafsky, Graphics for the multivariate two-sample problem, J. Am. Stat. Assoc. 76 (374) (1981) 277–287. 





[76] M.F. Schilling, Multivariate two-sample tests based on nearest neighbors, J. Am. Stat. Assoc. 81 (395) (1986) 799–806. 





[77] S. Boyd, L. Vandenberghe, Convex Optimization, Cambridge University Press, 2004. 





[78] N. Frosst, N. Papernot, G. Hinton, Analyzing and improving representations with the soft nearest neighbor loss, in: Proceedings of the 36th International Conference on Machine Learning, 2019. 





[79] G. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network, arXiv preprint arXiv:1503.02531, 2015. 





[80] R. Ricciardi, M. Streeter, Comparing the American Community Survey to the American Housing Survey (2023). https://www.census.gov/data/academy/ webinars/2023/comparing-the-acs-to-ahs.html (Online); accessed 1 May 2024. 





[81] C. Guo, F. Berkhahn, Entity embeddings of categorical variables, arXiv preprint arXiv:1604.06737, 2016. 





[82] C. Seger, An investigation of categorical variable encoding techniques in machine learning: binary versus one-hot and feature hashing (2018). 





[83] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2980–2988. 





[84] X. Chen, C. Liang, D. Huang, E. Real, K. Wang, Y. Liu, H. Pham, X. Dong, T. Luong, C.-J. Hsieh, et al., Symbolic discovery of optimization algorithms, arXiv preprint arXiv:2302.06675, 2023. 





[85] A. Alaa, B. Van Breugel, E.S. Saveliev, M. van der Schaar, How faithful is your synthetic data? sample-level metrics for evaluating and auditing generative models, in: International Conference on Machine Learning, PMLR, 2022, pp. 290–306. 





[86] S. Kim, P. Bansal, A deep generative model for feasible and diverse population synthesis, Transp. Res. Part C Emerg. Technol. 148 (2023) 104053. 





[87] B.B. Mandelbrot, The Fractal Geometry of Nature, W. H. Freeman, New York, 1983. 





[88] W.E. Leland, M.S. Taqqu, W. Willinger, D.V. Wilson, On the self-similar nature of Ethernet traffic (extended version), IEEE/ACM Trans. Netw. 2 (1) (1994) 1–15. 





[89] A. Buades, B. Coll, J.-M. Morel, A non-local algorithm for image denoising, in: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 2, IEEE, 2005, pp. 60–65. 





[90] D. Glasner, S. Bagon, M. Irani, Super-resolution from a single image, in: 2009 IEEE 12th International Conference on Computer Vision, IEEE, 2009, pp. 349–356. 





[91] P. Ye, B. Tian, Y. Lv, Q. Li, F.-Y. Wang, On iterative proportional updating: limitations and improvements for general population synthesis, IEEE Trans. Cybern. 52 (3) (2020) 1726–1735. 





[92] K. Ward, G. Macfarlane, Ipfr: list balancing for reweighting and population synthesis (2020). R package version 1.0.2. 





[93] C. Genest, J. Nešlehová, A primer on copulas for count data, ASTIN Bull.: J. IAA 37 (2) (2007) 475–515. 





[94] H. Joe, Dependence Modeling with Copulas, CRC Press, Boca Raton, 2015. 





[95] Y. Okhrin, A. Ristig, Y.-F. Xu, Copulae in high dimensions: an introduction, In P. Jaworski, F. Durante, W.K. Härdle (Eds.), Copulae in Mathematical and Quantitative Finance, 3 ed., Springer, Berlin, Heidelberg, 2017, pp. 247–277. 





[96] B.E. Flanagan, E.W. Gregory, E.J. Hallisey, J.L. Heitgerd, B. Lewis, A social vulnerability index for disaster management, J. Homel. Secur. Emerg. Manag. 8 (1) (2011) 0000102202154773551792. 





[97] S.L. Cutter, B.J. Boruff, W.L. Shirley, Social vulnerability to environmental hazards, in: Hazards Vulnerability and Environmental Justice, Routledge, 2012, pp. 143–160. 





[98] Y. Cao, B. Ivanovic, C. Xiao, M. Pavone, Reinforcement learning with human feedback for realistic traffic simulation, arXiv preprint arXiv:2309.00709, 2023. 





[99] A.V. Solatorio, O. Dupriez, Realtabformer: Generating realistic relational and tabular data using transformers, arXiv preprint arXiv:2302.02041, 2023. 

