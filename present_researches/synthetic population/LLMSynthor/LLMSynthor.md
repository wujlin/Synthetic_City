# LLMSynthor: Macro-Aligned Micro-Records Synthesis with Large Language Models

Yihong Tang1, Menglin Kong1, Junlin He2, Tong Nie2, Lijun Sun1B 

1McGill University, 2The Hong Kong Polytechnic University 

{yihong.tang,menglin.kong}@mail.mcgill.ca, {junlinspeed.he, tong.nie}@connect.polyu.hk, lijun.sun@mcgill.ca 

# ABSTRACT

Macro-aligned micro-records are essential for simulations in social science and urban studies. For instance, epidemic models of urban disease spread are only credible when micro-level records reproduce realistic individual mobility and contact patterns, while macro-level aggregates match real-world statistics such as case counts or travel flows. Still, large-scale collection of such fine-grained data is impractical, leaving researchers with only macro-statistics (e.g., travel surveys or case counts). Large Language Models (LLMs), leveraging rich real-world priors learned from vast corpora, excel at generating realistic micro-records, but standard record-by-record sampling is inefficient and fails to enforce alignment with target macro-statistics. Given this, we propose LlmSynthor, a framework capable of synthesizing realistic micro-records that are statistically aligned with target macro-statistics. LlmSynthor transforms a pre-trained LLM into a macro-aware simulator that incrementally builds a synthetic dataset through an iterative process. At each iteration, a batch of micro-records is generated to reduce the discrepancy between synthetic and target macro-statistics. By treating the LLM as a nonparametric copula for inferring joint dependencies over variable combinations, the iterative process ensures the synthetic data are macro-statistically aligned with the target marginals and joints. To address sampling inefficiency, we introduce LLM Proposal Sampling, where the LLM, guided by discrepancies, generates a plan of proposals, each defining specific values or ranges for all variables and specifying the number of records to generate. This enables the framework to minimize discrepancies efficiently while preserving the realism grounded in the LLM’s priors. Evaluations on synthetic and real-world datasets (mobility, e-commerce, population) encompassing diverse formats and settings show that LlmSynthor achieves high record realism, statistical fidelity, and practical utility, positioning it broadly applicable across economics, social science, urban studies, and beyond. 

# INTRODUCTION

High-stakes decisions in domains like public health and urban planning are increasingly supported by agent-based simulations of complex human behavior 1. At the micro-level, individual records capture behavioral detail such as mobility and contact patterns, which drive realistic dynamics. At the macro-level, aggregated statistics ensure consistency with population-level trends. Only when micro-records are collectively aligned with real-world macro-statistics can such simulations yield valid insights 2, yet these data are unattainable because large-scale collection is infeasible due to both prohibitive costs and stringent privacy constraints 3. Consequently, researchers and policymakers must rely on macro-statistics, such as census reports or case counts, leaving a micromacro gap 4. The core challenge is therefore to synthesize realistic micro-records that are statistically faithful to these known macrostatistics. 

This micro-macro synthesis task, however, is beyond the capabilities of existing generative paradigms. Current methods, from classical statistical models to modern deep generative networks 5–7, are ill-equipped, as they all require access to a large volume of microrecords that are unavailable for model fitting. Furthermore, their reliance on rigid parametric assumptions or implicit model biases often leads to the generation of unrealistic records, such as a six-year-old with a doctorate, necessitating inefficient post-hoc fixes like rejection sampling. Most of these approaches also lack the generality to handle the heterogeneous or unstructured data common in social sciences and urban studies 8, or require extensive manual engineering 9. This highlights the urgent need for a new framework that can synthesize realistic and complex micro-records that are statistically grounded, guided solely by macro-statistics. The advent of Large Language Models (LLMs) offers a compelling yet insufficient solution. Harnessing rich real-world priors, they excel at generating realistic, complex, and even unstructured micro-records without requiring any fine-tuning, making them power-

# Limitations of Existing Methods

# Generative Models (GMs)


Micro-Records


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/be6e936207c24fb2e04a728c12efac39b8254ae98e03fafa0b3f4ba2e2a0fc82.jpg)



GMs


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b625c51967557fb14a3f249d502aa8b2b230c536c7db2fc35f8b63800d22aac5.jpg)



Micro-Records


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5d28b956b1ebf2ac2840c25e1b465c6b6bbb1594fd756ef3115a48062ea06aef.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e94e699bbce8c5ddf1912b47a340881ca511be80b7f9e4a464b9a2a894e152dc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1a2cc74e6cdb126b49acd928374a9c5d20e4272814e4c216a702544dc68c5817.jpg)



Rely on Micro-Records



Unrealistic Records



Data Format-Specific


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/aca7df86ada46209ebfefffbbd353f50d19ba7b4d410a794b0eaa2c9b2bb5991.jpg)


# Large Language Models (LLMs)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/73ac9b7de44abd2aed4d6833af4689dd43f5f542e0e6bb34f0e4028dc4aeac16.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5ba7edbb85c516f7b208c81e2c18f75bb95a17b8b71254c77c61d8ec88cdcc52.jpg)



Inefficient Generation


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5e94c57d9633e529d0f718cccc73981a655c0b3623a82bdcc4860ebcf4c82c3a.jpg)



No Macro-statistics Control


# LLMSynthor

# Macro-Aligned Micro-Records Synthesis


Macro-Statistics


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/88bbb1f0c65ebd6149b2c670001dcdfcc331a61796fe172fbd35e6bd39d28a7c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/2d1b814ec742ae1f0301f675531e0d587d43bc825e19ae49c2837df4716e45a5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b480c70466eb879cb06460519f788be376ab1f92d1db509c2976101be5e738eb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/99353ef674c62019838f7f1408b349fd789a8d5474d945a301d751709382238b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1eb6e2d83aaa9e5c8be2e71d1e8cfb04ae1fae63a11f51e56fb84acb6a8dab93.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/41c26fdbc9f574df44b4506861423983f08ac96b61ec298807a335326ab9dbd6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ea294a2b4fd420501a56f75d4671119a37fa78bf1a1d91fe4991d2f4b024ac15.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0d99e248eb80f8568e4856c28ed12030bb4038db21208a1c250cf5c9f45e42f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/8cec74cf1658175b24d93ebdbe4f71125650967030f76291bffb20417137ab29.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/804c6c5973ccc2aeb0e2344e971e18ae65b0eb007b7435a4a00a28d41ddaa332.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5c0143c6052d2bac34c9f29baca9776a0e5f55005fddd552e61c28d56b68cc49.jpg)



Micro-Records


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c08e96f3b0c88b977ab18951710f7d78283bd2fc6732ad29ccf928828067634a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5a31e8f4efcf318facfe60d3ac3222e43ab912410fb9907152e6245c40a483f7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e51d201902042de295411dcef2e3dfcf2b0ee53fb24a0b1af57967d0617cf7da.jpg)



Efficient



Format-Agnostic


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/dfae787f09daae0dab0db0d5f2df9aa81f973f68b37e9ce205e5415491a6ccbc.jpg)



Macro-Statistics Control


# Supports Various Downstream Applications

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a2fbeae840493d6647593c457f11cd40cba4871c05ad8adeaed15a9b7083807e.jpg)



Macro-Aligned


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/f77da23b044f9e928a4061aff55c7318ecb70890e2cf055d252c912a7d0abdeb.jpg)



Event Simulation


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c38504345fa31dee7a9e42b03a8bd1351c96519178dddd747d1c1fd4067be650.jpg)



Infection Spread


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e7bece35a9ffcc0ef8485a150155051f3abfeb0eff3eaf3874db3fa532076a7a.jpg)



Congestion Forecast under a Concert


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/181304215008d84ebfb936163f3e1b5184b7a3375dd6f607643680efae8d48ce.jpg)



Who gets exposed? Along which paths?



Figure 1: A comparison of existing generative paradigms and LlmSynthor.


ful universal generative priors 10–12. However, existing strategies remain limited. Fine-tuning LLMs requires massive micro-records and significant computational resources, which are often unavailable in practice. In-context or few-shot prompting can condition local generations, but it still operates record by record, making the process both inefficient and incapable of enforcing macrostatistical alignment 13. This inefficiency becomes critical when dealing with large-scale datasets, where generating sufficient high-quality data quickly is essential for population-scale simulations. As a result, synthetic datasets produced by standard LLM generation may be extremely time-consuming and appear realistic at the micro-level but remain statistically unfaithful to the target macro-statistics. The central challenge is therefore to transform the LLM’s generation process into one that is both efficient and macro-statistically controlled. 

To address this challenge, we present LlmSynthor, a framework that turns the LLM into a macro-aware simulator of micro-records. LlmSynthor generates synthetic micro-records and incrementally builds a synthetic dataset through an iterative feedback loop, guided only by the target macro-statistics. In each iteration, LlmSynthor measures the discrepancy between its synthetic and target macro-statistics, then prompts the LLM to generate a corrective batch of micro-records that minimizes this gap. This process is enabled by two core technical innovations. First, we treat the LLM as a powerful nonparametric copula, allowing it to infer the complex, nonlinear joint dependencies between variables without rigid statistical assumptions. This allows LlmSynthor to align the synthetic data with the target by matching all available marginal and joint macro-statistics. Second, to overcome generation efficiency bottlenecks, we introduce LLM Proposal Sampling, where the LLM creates a generation plan of micro-record proposals, each defining a localized joint distribution over all variables with an associated generation count. This approach uniquely combines the LLM’s rich prior knowledge for micro-level realism with rigorous, macro-level statistical control. Figure 1 compares our framework with existing paradigms. In summary, our main contributions are: 

• We introduce LlmSynthor, which transforms a pre-trained LLM into a macro-aware simulator for synthesizing micro-records that preserve realism while aligning with target macro-statistics. 

• We propose (i) a nonparametric copula interpretation of the LLM to capture joint dependencies over variable combinations and (ii) LLM Proposal Sampling to ensure efficient and realistic generation. 

• We design a discrepancy-guided synthesis process that iteratively extends and aligns the synthetic and the target macro-statistics, providing rigorous, dataset-level statistical control. 

• Evaluations on synthetic and real-world datasets (mobility, e-commerce, population) across diverse formats and settings show that LlmSynthor consistently outperforms expert baselines in record realism, statistical fidelity, and practical utility, while maintaining broad versatility. 

# RELATED WORK

Data Synthesis Early data synthesis methods focused on explicit statistical control, using techniques like iterative proportional fitting (IPF) 14,15, Bayesian networks 9,16,17, and copula models 18–20 to match marginals and preserve dependencies. While interpretable, these methods often rely on strong assumptions and face challenges with scalability and heterogeneity. Deep generative models, including VAEs 21,22, GANs 23–26, and recent diffusion- or flow-based models 27–30, have improved realism and high-dimensional modeling. However, they tend to entangle marginals with dependencies and often require expensive retraining for new domains. LLM-based methods, such as GReaT 31 and HARMONIC32, treat structured data as natural language, enabling zero-shot transfer and broad domain coverage through autoregressive decoding. However, these methods lack direct control over marginal and joint distributions, are sample inefficient, and struggle to scale with large or heterogeneous datasets. 

LLMs as Data Generators LLMs have shown exceptional versatility as data generators across various domains. They have been used to augment data 33, create instruction-finetuning datasets 34,35, generate tabular data 31,32, synthesize executable code 36,37, and produce personal mobility data aligned with user preferences 38 and routines 39. LLMs also generate question-answering pairs to enhance model robustness 40 and privacy-preserving text via topic modeling 41. While existing methods excel at ensuring the semantic or functional quality of individual records, they often fail to control global statistical properties of the dataset. Some works introduce limited control, such as ensuring topic completeness 42 or data coverage 43. However, these approaches lack explicit macroscopic statistical control. Most methods generate data record by record or in small, independent batches, without enforcing global statistical properties, which highlights the gap between generating realistic individual records and ensuring statistical fidelity across the entire dataset. This challenge is directly addressed by LlmSynthor. 

# METHODOLOGY

Definition 1 (Micro-record). A micro-record represents a single data record for an individual entity. Each record x is a set of variablevalue pairs: $x \ = \ \{ ( v _ { i } , a _ { i } ) \} _ { i = 1 } ^ { d _ { x } }$ , where each $v _ { i }$ is a variable (discrete or continuous) from a predefined variable set V in the given application context, and $a _ { i }$ is its corresponding value. The number of variables $d _ { x }$ may vary across records to accommodate the flexibility of unstructured data. A dataset is then defined as a set of records $\mathcal { D } = \{ x _ { j } \} _ { j = 1 } ^ { | \mathcal { D } | }$ . 

Definition 2 (Macro-statistics). Macro-statistics are aggregated summaries of micro-records that capture distributional patterns of a dataset. They are often the only information available from external sources, and serve as the target that aggregates of synthetic micro-records should align with. They can be marginal statistics, which describe the distribution of a single variable (e.g., a frequency vector of education levels), or joint statistics, which describe dependencies among combinations of variables (e.g., a contingency table of education by employment status). We define an operator $\Phi$ that maps any dataset D of micro-records into a set of macro-statistics, i.e., $\Phi : { \mathcal { D } } \mapsto { \mathcal { S } } ,$ , where ${ \cal { S } } = \{ \phi _ { i } \} _ { i = 1 } ^ { | S | }$ and each $\phi _ { i }$ is either a marginal or a joint statistic. By definition, Φ can be applied to compute arbitrary marginal or joint macro-statistics from D, but in practice its instantiation is application-specific, determined by the available target macro-statistics in a given context. 

Problem 1 (Macro-aligned Micro-records Synthesis). Given a set of target macro-statistics Starget, the objective is to construct a dataset of n realistic micro-records, $\mathcal D _ { \mathrm { s y n t h } } = \{ \hat { x } _ { j } \} _ { j = 1 } ^ { n }$ , such that the macro-statistics induced from Dsynth closely align with Starget. Formally, we seek minDsynth $Q \big ( \Phi ( \mathcal { D } _ { \mathrm { s y n t h } } )$ , $S _ { \mathrm { t a r g e t } } \rangle$ , where $\Phi$ is the aggregation operator defined in Definition 2, and $Q$ is a suitable discrepancy measure in the macro-statistics space. 

Take a mobility dataset as an example. A micro-record can represent a single trip, such as x = {(origin, ‘Times Square’), (destination, ‘Central Park’), (mode, ‘Bike’), (time, 17)}. Here, the variable set is $\nu = \{ \circ x \mathrm { i g i n } $ , destination, mode, time}, and each micro-record assigns specific values to these variables. Alternatively, an unstructured micro-record could represent an entire daily travel diary, in which the number of trips varies across individuals. 

In practice, only macro-statistics are available from external sources, denoted $S _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } }$ . For example, this set may include marginal statistics, such as the frequency vector of transport modes chosen by travelers, and joint statistics, such as the contingency table of 

# (a) Variable Dependency Inference


Input: Target Macro-Statistics & Variables



Variables: ?= {??????, ???????????, ????, ????, ????????}


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c859ede844be86dea713b535a92243e2f36755a6d8a0a15bb1c376c41a33d587.jpg)



demand by time origin transport mode activity


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/9299dcb5f4a618bbab34ba268b8cd75c1824972de81cf72861ea4c188f64fba3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/9ffd706fe9dfaf09b1984948d2b01a825607d09005b49cedaf176e836517ebba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/023db711e7d3c6c1fd5c0acacd2bbd8b9b74399308b3a4849ef0d34f4af4c790.jpg)


# (b) Discrepancy-Guided Iterative Synthesis (Iteration t)


Updating Synthetic Macro-Statistics


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b132b60181f31b5985964074a58fdc58188dfa4caeee98111a9f7e45c34325d0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e6e7d3d875b1f67bef08872cb22096041610eab495ec487e30e7a63588d0f486.jpg)



Cumulate



Cumulated Micro-Records $\mathcal { D } _ { s v m t h } ^ { ( t ) }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/559aa80376e7eb3dae40414b93bb6c55f77ac92526a3b28138600447e46698c6.jpg)



???????? (??????(?) ) Aggregate


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/2b2b83d4a0dc30665bd80c3a7ba68c070a2bf5aac53128562c2f623403bc086a.jpg)



Discrepancy Attribution


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/17f72ee8d1c8287667bbb99f4daa62c27a9b2538ce35d5b2948189b1ae794567.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1093560efe8f03af6d258f8e49002e61fe9394d570a77c1f371750d5cb781f0e.jpg)



LLM Proposal Sampling



Guide base on Discrepancies



Generate more records:



(i) Marginal: with mode as transit,



(ii) Joint: from Central Park to Time


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/620b496c2e63a6a44135254e0885d39fb73f6fa6458c6e7d93a2b831acca2274.jpg)



Sampling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ac81838644455e650647f8a0c554b2da2b6897398c9d4e9fa42c021bbc24cf94.jpg)



Figure 2: Overview of LlmSynthor. For illustration, the figure uses a mobility dataset as a running example, but the approach is general and applicable to diverse domains and data contexts.


origin and destination zones. The aggregation operator $\Phi _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } }$ mob , defined in Definition 2, is then instantiated for the mobility context so that $\Phi _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } } ( \mathcal { D } _ { \mathrm { s y n t h } } ^ { \mathrm { m o b } } ,$ mobtarget( ) produces those statistics that are directly comparable to $S _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } }$ . The task is therefore to construct a dataset of micro-records $\mathcal { D } _ { \mathrm { s y n t h } } ^ { \mathrm { m o b } }$ such that $\Phi ( { \cal D } _ { \mathrm { s y n t h } } ^ { \mathrm { m o b } } )$ ) closely matches $S _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } }$ . For example, if the target specifies $30 \%$ of trips by bike (marginal) and $1 0 \%$ from Times Square to Central Park (joint), the synthetic dataset must match these proportions when aggregated by $\Phi _ { \mathrm { t a r g e t } } ^ { \mathrm { m o b } }$ . 

# Overview

LlmSynthor consists of two key components: (i) Variable Dependency Inference, where the LLM acts as a copula-like mechanism to capture dependencies among variables and refine the set of target joint macro-statistics, and (ii) Discrepancy-Guided Iterative Synthesis, where at each iteration t, the LLM generates a batch of micro-records to reduce the discrepancy An overview of the frame work is in Figure 2, and a simplified example that illustrates the process is in Figure 19 in Appendix . 

# Variable Dependency Inference

LlmSynthor operates solely on macro-statistics, which provide complementary information, but some joint statistics can be redundant, while others may be missing. However, marginal statistics are essential as they describe the basic distributions of variables. Therefore, dependency inference focuses on identifying variable combinations should be treated as joint dependencies, while retaining all available marginals. This ensures the synthesis is based on the most informative dependencies. 

We draw inspiration from copula theory 44, which represents a joint distribution as marginals plus a dependency structure. We use a pre-trained LLM as a nonparametric copula to infer variable subsets $\mathcal { C } = \{ c _ { k } \} _ { k = 1 } ^ { | \mathcal { C } | }$ , $c _ { k } \subseteq \mathcal { V } _ { \mathrm { ~ } }$ , where each subset $c _ { k }$ is expected to exhibit strong statistical dependence. The LLM infers these dependencies by leveraging (i) the semantic meaning of variable names (e.g., “education” and “income” are related), and (ii) the available macro-statistics $S _ { \mathrm { t a r g e t } }$ : 

$$
\mathcal {C} \sim \operatorname {L L M} \left(\mathrm {p} _ {\text {c o p u l a}} \left(\mathcal {S} _ {\text {t a r g e t}}, \mathcal {V}\right)\right), \tag {1}
$$

where the prompt pcopula asks for informative variable combinations (as provided in Appendix ). For example, for a variable set $\pmb { \nu } = \{ \mathsf { o r i g i n } $ , destination, mode, time, the LLM might infer that the variable subsets C exhibit strong dependencies: $\{ c _ { 1 } = [ \tt o r i g i n . \breve { \tt { i } }$ destination, time], c2 = [time, activity]}. 

Next, we retain all marginals and expand or filter the joint statistics in Starget based on $\boldsymbol { \mathscr { C } }$ , yielding $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ . If an inferred dependency lacks corresponding data, practitioners are encouraged to collect additional statistics, if unavailable, the LLM can approximate the joint distribution using existing macro-statistics. The aggregation operator is then updated as $\Phi _ { \mathrm { t a r g e t } } \mapsto \Phi _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ , ensuring that the induced statistics from any dataset are directly comparable to $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ . 

For example, in Figure 2(a), the LLM identifies the dependency $c _ { 1 }$ between the variables [origin, destination, time], replacing the original joint macro-statistic $\phi _ { \mathrm { O D } }$ with a new one, $\phi _ { \mathrm { T O D } }$ , which captures the time-dependent OD demand. In contrast, the dependency $c _ { 2 }$ between [time, activity] remains unchanged, and the corresponding macro-statistic $\phi _ { \mathrm { T A } }$ is preserved. 

# Discrepancy-Guided Iterative Synthesis

With the updated target macro-statistics $S _ { t a r g e t } ^ { \mathcal { C } }$ and operator $\Phi _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ in place, LlmSynthor then enters an iterative synthesis loop that progressively builds the synthetic dataset $\mathcal { D } _ { \mathrm { s y n t h } }$ . At iteration $t$ , three steps are performed: $( j )$ updates the macro-statistics for the cumulative dataset via screpancy signals, generate a batch of new records . We now elaborate on each step. ${ \mathcal S } _ { \mathrm { s y n t h } } ^ { ( t ) } = \Phi _ { \mathrm { t a r g e t } } ^ { \mathcal C } ( \mathcal D _ { \mathrm { s y n t h } } ^ { ( t ) } )$ Ssynth ; (ii) compute discrepancy signals $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ to reduce $\Delta ^ { ( t ) }$ , and update the cumulative set $\Delta ^ { ( t ) } = Q \big ( S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } } , S _ { \mathrm { s y n t h } } ^ { ( t ) } \big ) ; ( i / i )$ ) under the $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t + 1 ) } =$ $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) } \cup \mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ 

1. Updating Synthetic Macro-statistics. At the beginning of iteration $t$ , the synthetic dataset D(t)synth is aggregated into macro- $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ statistics aligned with the target S(t)synt ${ \cal S } _ { \mathrm { s y n t h } } ^ { ( t ) } = \Phi _ { \mathrm { t a r g e t } } ^ { \cal C } \Big ( { \cal D } _ { \mathrm { s y n t h } } ^ { ( t ) } \Big )$ Dsynth . To ensure comparability across heterogeneous variables, each record x ∈ D(t) $x \in \mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ is mapped into a unified discretized space. Discrete variables are represented by frequency vectors, while continuous variables are discretized into bins (e.g., quantile-based or fixed intervals) that follow the same scheme as available in the target macrostatistics. When multiple variables are involved, the corresponding bins define a contingency table that is directly comparable to the observed target. 

2. Discrepancy Attribution. Given the updated synthetic macro-statistics $S _ { \mathrm { s y n t h } } ^ { ( t ) }$ and the target macro-statistics $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ , the next step is to measure their discrepancy. Formally: 

$$
\Delta^ {(t)} = \left\{\delta_ {1} ^ {(t)}, \delta_ {2} ^ {(t)}, \dots , \delta_ {| \mathcal {S} _ {\text {t a r g e t}} ^ {\mathcal {C}} |} ^ {(t)} \right\} = Q \big (\mathcal {S} _ {\text {t a r g e t}} ^ {\mathcal {C}}, \mathcal {S} _ {\text {s y n t h}} ^ {(t)} \big), \tag {2}
$$

where $Q ( \cdot , \cdot )$ is a discrepancy measure. At each iteration, we generate a batch of micro-records to expand the synthetic dataset. To guide this process, we implement $Q$ as the directed difference $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } } - S _ { \mathrm { s y n t h } } ^ { ( t ) }$ S(t) , which captures the frequency distribution differences of each macro-statistic. The positive and negative values indicate underrepresented and overrepresented portions in the synthetic data compared to the target, respectively. By focusing on large positive discrepancies, we generate additional data to address these gaps, gradually reducing the discrepancy over time. Since $\Phi ( \cdot , \cdot )$ operates on a discretized representation of D synth, the $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ resulting macro-statistics are tied to specific bins or discrete values (e.g., $\mathsf { \Pi } ^ { \ast } \mathrm { t i m e } = 5 . 6 ~ \mathsf { p } . \mathsf { m }$ . and mode $\underline { { \underline { { \mathbf { \delta \pi } } } } }$ bike”), making discrepancies interpretable and attributable to particular subsets of $\nu$ . This enables the formulation of actionable LLM prompts. 

For example, consider the directed frequency difference $( S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } } - S _ { \mathrm { s y n t h } } ^ { ( t ) } )$ for transport modes: $\delta _ { M } ^ { ( t ) } = \{ \mathsf { b i k e } : + 3 0 \% , \mathsf { c a r } : - 2 0 \% \} .$ This difference indicates that the target distribution has a $30 \%$ higher proportion of bike trips compared to the synthetic distribution, so we can prompt the LLM with <generate more bike trips>, gradually reducing the gap and aligning the synthetic distribution closer to the target. An illustrative example of this process is shown in Figure 18. Theorem 1 further proves that this approach converges over time. Additional details on the discrepancy measure can be found in Appendix . 

3. Discrepancy-Guided LLM Proposal Sampling. The discrepancy signals $\Delta ^ { ( t ) }$ are then used to guide the generation of new records. Rather than producing one micro-record per LLM request, which is inefficient and hard to align with the target macrostatistics, the LLM is shifted from a direct generator to a planner. We introduce LLM Proposal Sampling, where the LLM outputs $m$ proposals: 

$$
\left\{\pi_ {1}, \dots , \pi_ {m} \right\} \sim \operatorname {L L M} \left(\mathbf {p} _ {\text {p r o p o s a l}} (\mathcal {V}, \mathcal {C}, \Delta^ {(t)})\right), \tag {3}
$$

given a prompt pproposal that instructs it to design proposals aimed at reducing the discrepancies in $\Delta ^ { ( t ) }$ while respecting the dependency structure C. The implemented prompt is provided in Appendix . Each proposal $\pi _ { i }$ defines specific values or ranges for all variables and specifying the number of records to generate. For discrete attributes, this corresponds to fixing categories (e.g., mode $=$ bike); for continuous ones such as time, it specifies ranges (e.g., time ∈ [17, 20]). In addition, the LLM also plans the number of 

records to be drawn from each proposal, so that generation is not only localized but also quantitatively guided. Thus, every $\pi _ { i }$ is interpretable, directly sampleable, and aligned with the discretization scheme used in $\Phi _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ , while also grounded in the LLM’s priors to ensure that the proposed joint configurations remain realistic. An example is shown in the right panel of Figure 2(b), where the LLM generates multiple proposals for record creation. The first proposal $\pi _ { 1 }$ specifies 50 records with origin “Brooklyn”, destination “Time Square”, mode “transit”, activity “Shop”, and time range [6, 9]. This proposal defines a targeted joint distribution over variables, addressing identified discrepancies and guiding the efficient generation. 

From these proposals, a batch of synthetic records $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ is sampled and merged into the cumulative dataset: $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t + 1 ) } = \mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ = D ( t ) synth ∪ $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ tributional controller: it bridges macro-level discrepancy signals and micro-level record generation, efficiently reducing targeted mismatches while retaining the realism induced by the LLM’s priors. 

Iterative Synthesis. We summarize the iterative synthesis process in Algorithm 1. At each iteration $t$ , three steps are performed: First, the cumulative synthetic dataset $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ D(t)synth is aggregated into macro-statistics S(t)synt $ { S _ { \mathrm { s y n t h } } ^ { ( t ) } }$ via the updated operator $\Phi _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ , and compared with the target $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ using the discrepancy measure $Q$ , yielding discrepancy signals $\Delta ^ { ( t ) }$ . Second, these signals are provided to the LLM, which generates proposal distributions $\{ \pi _ { i } ^ { ( t ) } \}$ through LLM Proposal Sampling, each specifying a localized distribution over joint variable configurations aimed at reducing the dominant mismatches. Third, records sampled from these proposals form a batch $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ , which is merged into the cumulative dataset to obtain $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) }$ This loop progressively reduces the discrepancies, refining $\mathcal { D } _ { \mathrm { s y n t h } }$ until it closely aligns the target macro-statistics. 

# Algorithm 1 Iterative Synthesis

Require: Target $\scriptstyle { S _ { \mathrm { t a r g e t } } }$ , operator $\Phi _ { \mathrm { t a r g e t } }$ , discrepancy Q, iterations T , D(0)synt $ { \mathcal { D } _ { \mathrm { s y n t h } } ^ { ( 0 ) } }  \emptyset$ 

1: ${ \mathcal { C } } \gets \mathtt { L L M } \big ( \mathtt { p } _ { \mathrm { c o p u l a } } ( S _ { \mathrm { t a r g e t } } , \bar { \nu } ) \big )$ 

2: SCtarget ← S C target; ΦCtarget ←C Φtarget 

3: for $t = 1$ to $_ T$ do 

4: h ) 

5: ∆(t) ← Q SCtarget, S(t)syn th  

$\{ \pi _ { i } ^ { ( t ) } \} \gets \mathtt { L L M } \big ( \mathtt { p } _ { \mathrm { p r o p o s a l } } ( \mathcal { V } , \mathcal { C } , \Delta ^ { ( t ) } ) \big )$ 

7: $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }  \bigcup _ { i } \{ \hat { x } _ { i } ^ { ( t ) } \} , \ \hat { x } _ { i } ^ { ( t ) } \sim \pi _ { i } ^ { ( t ) }$ 

8: D(t+1) synth ← D(t) synth ∪ D(t) batch 

10: return D(T ) synt 

# EXPERIMENTS

To comprehensively evaluate LlmSynthor, our experiments are structured to answer three research questions (RQs). RQ1: How effectively can LlmSynthor synthesize realistic and usable micro-records from limited aggregate-level macro-statistics? RQ2: How does LlmSynthor’s statistical fidelity compare to state-of-the-art models trained on full micro-record datasets? RQ3: How realistic can LlmSynthor effectively synthesize unstructured micro-records, without task-specific manual engineering? We address these questions using three practical tasks detailed below. Unless otherwise stated, we perform experiments using the Chat Completion mode of GPT-4.1-nano 45. 

# Mobility Synthesis

Mobility synthesis aims to generate a complete dataset of realistic micro-records, each detailing an individual’s time-stamped origindestination trip, activity, and transport mode. This is an essential capability for urban applications like transport planning and event simulation, as comprehensive, individual-level mobility data for an entire population is impractical to collect. 

Task Setup. To answer RQ1, we design a mobility synthesis task that mirrors a common real-world constraint: fusing aggregate information from multiple complementary data sources. From OpenPFLOW46, we extract trips (origin, destination, timestamp) and assign transport modes. Since OpenPFLOW lacks activity labels, we incorporate time-activity patterns from LLMob 39. This task tests the ability to align spatiotemporal and behavioral data by generating 30,000 trips in a day in Tokyo to match both macro-statistics. As existing methods cannot handle such Mixed-Source synthesis without manual adaptations, we focus on a qualitative assessment of LlmSynthor’s unique capabilities. Further details are provided in Appendix . 

Results. The results provide strong evidence for the first component of RQ1: LlmSynthor’s ability to generate realistic microrecords. Figure 3 compares the synthetic data against the target macro-statistics. The time-activity heatmaps on the left show close alignment, accurately capturing commuting peaks and midday activity rises. The OD flow heatmaps during the morning peak confirm that the synthetic trips reproduce key spatial patterns, matching high-density areas. These findings demonstrate that 

the synthesized population, in aggregate, successfully reproduces the guiding macro-level patterns. Furthermore, as shown in the Appendix (Figures 16 and 17), the generated micro-records also exhibit realistic internal structures, such as realistic correlations between travel mode and distance, reflecting their micro-level realism. 

Controllable Mobility Synthesis for Events Simulation To address the second component of RQ1 concerning the utility of the generated data, we demonstrate a key advantage of our framework: the ability to effortlessly incorporate arbitrary context into the synthesis process. We test LlmSynthor’s controllability in a “what-if” scenario. We simulate a concert at Tokyo Dome (20- 24h) by simply adding the prompt < There will be a concert from 20-24 at Tokyo Dome > during proposal generation. As shown in Figure 3, this simple intervention causes LlmSynthor to generate a surge of trips to the event location while preserving realistic background flows. This demonstrates LlmSynthor ’s potential as a powerful tool for scenario planning, allowing policymakers to simulate the effects of large events using detailed synthetic micro-records. 


Time-Activity Distribution


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ed3d3e83ddf99ebd3ad1291f6e5d398e3bdb116cd4facbf221f443ebc02c8b1b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/6dccf39f0f8ce08adcfeb3a11d1bd4ae89b497f364d9a7495c74e30425042a5f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e40071cffafc78fddeb11bb10a3db68a4b580b4d026c0a9e76fd16fd29e34d04.jpg)



Morning peak (6-9 a.m.) demand flow intensity heatmap


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b77be37567db7026b96363ae9d7de46ce8d5b5ec9c507a135b9a2fde6f73a226.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e937a63125d636459b6e88049a4e84f6131f20c2221440f1bc8146081a0bf80b.jpg)



Simulated OD flow during a concert at Tokyo Dome (20-24)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0c51069d070333902c1939dac82a2d2cad5e63717e78e5b0d0ac81d36331ff34.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/7866249a2563f8407c4864b1e66f24840e711a215ba0a14909eb37924f51828e.jpg)



Figure 3: Real vs. synthetic mobility patterns.


# E-Commerce Transaction Synthesis

Having demonstrated LlmSynthor’s unique capabilities, we now rigorously assess its statistical fidelity against strong, established baselines to answer RQ2. We use a controlled e-commerce synthesis task, providing a comprehensive comparison with state-of-theart Tabular synthesis models. This experiment critically tests whether LlmSynthor, guided only by macro-statistics, can match or exceed the performance of models trained on the full set of micro-records. 

<table><tr><td rowspan="2">Methods</td><td colspan="2">\( v_A \)</td><td colspan="2">\( v_G \)</td><td colspan="2">\( v_L \)</td><td colspan="2">\( v_C \)</td><td colspan="2">\( v_X \)</td><td colspan="2">\( v_M \)</td></tr><tr><td>W</td><td>Gap</td><td>Tvd</td><td>Gap</td><td>Tvd</td><td>Gap</td><td>Tvd</td><td>Gap</td><td>W</td><td>Gap</td><td>Tvd</td><td>Gap</td></tr><tr><td>TVAE</td><td>2.06</td><td>0.032</td><td>0.008</td><td>0.01</td><td>0.056</td><td>0.043</td><td>0.054</td><td>0.02</td><td>113.194</td><td>0.085</td><td>0.017</td><td>0.013</td></tr><tr><td>CTGAN</td><td>4.429</td><td>0.057</td><td>0.117</td><td>0.065</td><td>0.162</td><td>0.076</td><td>0.080</td><td>0.028</td><td>138.998</td><td>0.059</td><td>0.088</td><td>0.022</td></tr><tr><td>CopulaGAN</td><td>4.82</td><td>0.027</td><td>0.052</td><td>0.016</td><td>0.045</td><td>0.031</td><td>0.057</td><td>0.024</td><td>151.239</td><td>0.047</td><td>0.045</td><td>0.014</td></tr><tr><td>GReaT</td><td>2.862</td><td>0.052</td><td>0.016</td><td>0.009</td><td>0.039</td><td>0.020</td><td>0.045</td><td>0.027</td><td>169.866</td><td>0.104</td><td>0.009</td><td>0.012</td></tr><tr><td>TabSyn</td><td>1.196</td><td>0.012</td><td>0.012</td><td>0.022</td><td>0.007</td><td>0.022</td><td>0.045</td><td>0.01</td><td>114.12</td><td>0.067</td><td>0.028</td><td>0.005</td></tr><tr><td>LLMSYNTHOR</td><td>1.13</td><td>0.023</td><td>0.002</td><td>0.008</td><td>0.002</td><td>0.012</td><td>0.010</td><td>0.022</td><td>12.762</td><td>0.011</td><td>0.003</td><td>0.004</td></tr></table>


Table 1: Marginal evaluation, lower values indicate better distribution alignment.


Task Setup. We simulate a controlled environment where each transaction is sampled from a closed-form Bayesian network over six variables: $\left\{ v _ { A } , v _ { G } , v _ { L } , v _ { C } , v _ { X } , v _ { M } \right\}$ , representing user_age, gender, location_tier, product_category, price, and payment_method. The generative process follows a structured probabilistic graphical model with a known joint distribution: $p ( v _ { A } , v _ { G } , v _ { L } , v _ { C } , v _ { X } , v _ { M } ) =$ $p ( v _ { A } ) p ( v _ { G } ) p ( v _ { L } ) p ( v _ { C } | v _ { A } , v _ { G } ) p ( v _ { X } | v _ { C } ) p ( v _ { M } | v _ { L } ) .$ . This controlled setting enables a precise and rigorous evaluation. We generate a 2,000-record reference dataset to serve as the ground truth. To ensure a fair and direct comparison against baselines, we compute a set of target macro-statistics directly from this reference dataset, following the detailed procedure described in Appendix . These macro-statistics serve as the sole input for LlmSynthor. In contrast, all baseline models are trained on the full 2,000 individual records. This setup allows us to compare the statistical fidelity of different data synthesis methods and reflects how well the joint dependencies inferred by LlmSynthor’s internal logic match the real underlying structure. 

Baselines. Given the structured tabular nature of this task, we compare LlmSynthor to representative baselines across major generative paradigms that are trained on the full record-level data: (1) TVAE and CTGAN (VAE- and GAN-based); (2) CopulaGAN (GAN with copula modeling); (3) GReaT (autoregressive transformer); and (4) TabSyn (diffusion-based). For a fair comparison of output quality, we apply rejection sampling to ensure record realism. Detailed baseline information is provided in Appendix . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5b70c85db8b2a41d2bd205565a80e70f86f688d45b671ee4acb4c98470d00230.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/331365cd0543a5c2d04174de926d43ac660dc31e049a5d88f9b1c15c1c10b2e7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/9bbc3df5b713981d26711e54b86dc276be1a5ff74ed107ddd4a5bc3607a5fbf9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/79a769f9276ae0c2133749219180c855ac3ded3df843b1f93bd4278906790f7d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/2423a69f22ee0237aa8d37f5ef9a1fba2c909b51af7fe3410ca7a37a90436fc0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e8446b63a0a084c5b462a6f2929c95f3c5d84547d86e82d5d8aa098775808c5c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/d72d8755917b92a21de1b2913a07341130789b10304155b15758d06c4a8d1e1f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/459dda8119c8c836a1202bcd4afb65db44bc996ee946293ce7380f2ce1f7c0d4.jpg)


<table><tr><td rowspan="2">Methods</td><td colspan="2">[vA, vG, vC]</td><td colspan="2">[vC, vX]</td><td colspan="2">[vL, vM]</td></tr><tr><td>Jsd</td><td>Gap</td><td>Jsd</td><td>Gap</td><td>Jsd</td><td>Gap</td></tr><tr><td>TVAE</td><td>0.23</td><td>0.074</td><td>0.245</td><td>0.106</td><td>0.185</td><td>0.051</td></tr><tr><td>CTGAN</td><td>0.133</td><td>0.055</td><td>0.298</td><td>0.098</td><td>0.145</td><td>0.076</td></tr><tr><td>CopulaGAN</td><td>0.133</td><td>0.057</td><td>0.280</td><td>0.102</td><td>0.069</td><td>0.018</td></tr><tr><td>GReaT</td><td>0.087</td><td>0.058</td><td>0.382</td><td>0.177</td><td>0.038</td><td>0.020</td></tr><tr><td>TabSyn</td><td>0.083</td><td>0.022</td><td>0.237</td><td>0.082</td><td>0.027</td><td>0.015</td></tr><tr><td>Ours</td><td>0.071</td><td>0.022</td><td>0.134</td><td>0.020</td><td>0.007</td><td>0.007</td></tr></table>


Table 2: Joint evaluations.



Figure 4: Qualitative Distributions and Comparisons.


Results. The results clearly answer RQ2 in the affirmative. Despite being restricted to macro-statistics, LlmSynthor consistently outperforms baselines that were trained on the full micro-record dataset across measures of statistical fidelity and downstream utility. We evaluate the synthetic data from two perspectives. First, to assess statistical fidelity, Tables 1 and 2 report key metrics for both marginal and joint distributions. Across all measures, including Wasserstein distance (W), Total Variation Distance (TVD), and the Classifier Two-Sample Test (C2ST) Gap (|acc − 0.5|), LlmSynthor achieves the lowest divergence scores. The visualizations in Figure 4 further confirm that LlmSynthor’s generated distributions most closely match the ground truth, demonstrating its superior ability to preserve the specified dependency structures Second, we evaluate the practical downstream utility of the synthetic data. We introduce two derived variables grounded in economic theory: discount_propensity and lifetime_value_band (see Appendix for definitions). We then train standard classifiers (logistic regression, decision trees, and random forests) on the data from each synthesizer. As shown in Figure 4, models trained on LlmSynthor’s data generalize best to real data, proving that its high statistical fidelity translates directly to superior performance on practical tasks. Additional analyses, including tracking the convergence of distributional divergence over iterations, ablation studies confirming the effectiveness of Variable Dependency Inference, and robustness checks demonstrating stable performance when the backbone LLM is replaced with a smaller, open-source model Qwen2.5-7B, are provided in Appendix . 

# Population Synthesis

After establishing LlmSynthor’s quantitative superiority on structured data, we address RQ3 by evaluating its ability to generalize to complex, unstructured data formats. We tackle a real-world population synthesis task, a domain where data is inherently Unstructured due to varying household size. This experiment tests whether LlmSynthor’s general-purpose framework can outperform specialized models in a setting that typically requires significant, task-specific manual engineering. 

Task Setup. We use population microdata from the American Community Survey (ACS) for households in South Carolina. This dataset is a prime example of the challenge in RQ3, as it includes both household- and person-level attributes, resulting in unstructured records due to varying household sizes. After preprocessing, we obtain a dataset of about 15,000 households. The task is to generate a synthetic population that preserves the complex joint distributions across all demographic and household features. To assess real-world utility, we define 16 policy-relevant queries (e.g., proportion of multigenerational households) across six categories, which serve as a practical proxy for statistical fidelity. Full data and query details are provided in Appendix . 

Baselines. We compare LlmSynthor against a range of strong baselines specifically designed for population synthesis. These include (1) CP (a tensor factorization method), (2) HMM (a hierarchical probabilistic model), and (3) NVI (a deep variational framework). These baselines represent diverse and powerful specialized approaches, providing a rigorous benchmark for evaluating the performance of our general-purpose framework on macro-aware synthesis. 

Results. The results provide a resounding affirmative answer to RQ3. As shown in Table 3, LlmSynthor achieves the lowest mean relative error (MRE) on policyrelevant queries in every single category, often by a large margin. For instance, on equity-related queries, the error is reduced 

<table><tr><td>Methods</td><td>Rej. Rate%</td><td>Demog.</td><td>Employment</td><td>Equity</td><td>Household</td><td>Mobility</td><td>Vuln.</td></tr><tr><td>CP</td><td>73.9</td><td>0.54</td><td>1.02</td><td>5.79</td><td>2.34</td><td>1.47</td><td>0.86</td></tr><tr><td>HMM</td><td>57.8</td><td>0.56</td><td>0.32</td><td>4.23</td><td>2.01</td><td>0.48</td><td>0.91</td></tr><tr><td>NVI</td><td>96.8</td><td>0.53</td><td>0.27</td><td>5.49</td><td>2.06</td><td>0.24</td><td>1.06</td></tr><tr><td>Ours</td><td>13.3</td><td>0.21</td><td>0.2</td><td>0.25</td><td>0.13</td><td>0.35</td><td>0.37</td></tr></table>


Table 3: Rejection rate and category-wise MRE across queries.


from 4.23 (HMM) to just 0.25. Similar significant gains are observed for demographics, employment, mobility, and vulnerability metrics. Although LlmSynthor does not win on every individual query, its consistent superiority at the category level confirms that it more accurately captures the complex joint dependencies inherent in the population data, leading to synthetic populations with far greater practical utility. In addition, the rejection rate results further demonstrate the realism of our generated micro-records, which is especially valuable since real-world population data are inherently complex and no rejection sampling scheme can perfectly capture all realistic constraints. Further per-query results and a breakdown of rejection reasons and ratios are provided in Appendix . 

# Discussion

The experimental results answer the research questions positively: LlmSynthor can synthesize useful micro-records from multisource macro-statistics (RQ1), outperform baselines in quantitative fidelity with full micro-record access (RQ2), and generate realistic micro-records for complex, unstructured data without task-specific engineering (RQ3). 

Despite these strong results, the framework has some limitations. First, LLMs carry inherent priors that can occasionally introduce biases, misaligning with the target macro-statistics. This can be mitigated by using more constrained prompts or removing semantic cues during generation. Second, the framework’s scalability is limited by the context window and reasoning capabilities of the backbone LLM, especially for high-dimensional datasets with hundreds of variables. However, this limitation is expected to improve with future LLM advancements. Third, while LlmSynthor performs well with mixed-type i.i.d. data, it is not designed for perceptual or tightly sequential data such as images or raw time series. Nevertheless, it could be used as a high-level controller to guide domainspecific generators for these modalities. Finally, although LlmSynthor does not provide rigorous privacy guarantees, its synthesis process minimizes the risk of direct data re-identification by focusing on aligning with aggregate statistics rather than memorizing individual records. 

# CONCLUSION

In this work, we introduced LlmSynthor, a novel framework designed to bridge the critical micro-macro gap in data synthesis. We address the challenge of generating realistic, individual-level micro-records when only aggregate macro-statistics are available. Our central contribution is a paradigm shift: we repurpose a pre-trained LLM from an uncontrolled, record-by-record generator into a macro-aware simulator. This simulator operates within an iterative feedback loop, where it is continuously guided by discrepancies to ensure dataset-level statistical alignment. Our experiments demonstrated that LlmSynthor successfully synthesizes realistic micro-records with utility from multi-source statistics (RQ1), quantitatively outperforms state-of-the-art models with full data access (RQ2), and generalizes to realistic, unstructured data where other methods require manual engineering (RQ3). By providing a robust and general-purpose solution for generating statistically grounded micro-records, LlmSynthor opens new possibilities for data-driven research, agent-based simulation, and evidence-based policymaking. As language models continue to advance, the principles of statistically-guided synthesis presented here offer a scalable path toward creating reliable, high-fidelity synthetic worlds for a broad range of scientific and societal applications. 

# REFERENCES



[1] Emma Von Hoene, Amira Roess, Hamdi Kavak, and Taylor Anderson. Synthetic population generation with public health characteristics for spatial agent-based models. PLOS Computational Biology, 21(3):e1012439, 2025. 





[2] Na Jiang, Fuzhen Yin, Boyu Wang, and Andrew T Crooks. A large-scale geographically explicit synthetic population with social networks for the united states. Scientific Data, 11(1):1204, 2024. 





[3] Steven M Bellovin, Preetam K Dutta, and Nathan Reitinger. Privacy and synthetic datasets. Stan. Tech. L. Rev., 22:1, 2019. 





[4] Meng Zhou, Jason Li, Rounaq Basu, and Joseph Ferreira. Creating spatially-detailed heterogeneous synthetic populations for agent-based microsimulation. Computers, Environment and Urban Systems, 91:101717, 2022. 





[5] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 





[6] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 





[7] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





[8] Chao Ma, Sebastian Tschiatschek, Richard Turner, José Miguel Hernández-Lobato, and Cheng Zhang. Vaem: a deep generative model for heterogeneous mixed type data. Advances in Neural Information Processing Systems, 33:11237–11247, 2020. 





[9] Lijun Sun, Alexander Erath, and Ming Cai. A hierarchical mixture modeling framework for population synthesis. Transportation Research Part B: Methodological, 114:199– 212, 2018. 





[10] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





[11] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022. 





[12] Teyun Kwon, Norman Di Palo, and Edward Johns. Language models as zero-shot trajectory generators. IEEE Robotics and Automation Letters, 2024. 





[13] Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language models. arXiv preprint arXiv:2402.02244, 2024. 





[14] Richard J Beckman, Keith A Baggerly, and Michael D McKay. Creating synthetic baseline populations. Transportation Research Part A: Policy and Practice, 30(6):415–429, 1996. 





[15] Kirill Mueller and Kay W Axhausen. Hierarchical ipf: Generating a synthetic population for switzerland. Arbeitsberichte Verkehrs-und Raumplanung, 718, 2011. 





[17] Danqing Zhang, Junyu Cao, Sid Feygin, Dounan Tang, Zuo-Jun Max Shen, and Alexei Pozdnoukhov. Connected population synthesis for transportation simulation. Transportation research part C: emerging technologies, 103:1–16, 2019. 





[18] Roger B Nelsen. An introduction to copulas. Springer, 2006. 





[19] Athanassios N Avramidis, Nabil Channouf, and Pierre L’Ecuyer. Efficient correlation matching for fitting discrete multivariate distributions with arbitrary marginals and normal-copula dependence. INFORMS Journal on Computing, 21(1):88–106, 2009. 





[20] Ostap Okhrin, Alexander Ristig, and Ya-Fei Xu. Copulae in high dimensions: an introduction. Applied quantitative finance, pages 247–277, 2017. 





[21] Patricia A Apellániz, Juan Parras, and Santiago Zazo. An improved tabular data generator with vae-gmm integration. In 2024 32nd European Signal Processing Conference (EUSIPCO), pages 1886–1890. IEEE, 2024. 





[22] Syed Mahir Tazwar, Max Knobbout, Enrique Hortal Quesada, and Mirela Popa. Tab-vae: A novel vae for generating synthetic tabular data. In ICPRAM, pages 17–26, 2024. 





[23] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Modeling tabular data using conditional gan. Advances in neural information processing systems, 32, 2019. 





[24] Mrinal Kanti Baowaly, Chia-Ching Lin, Chao-Lin Liu, and Kuan-Ta Chen. Synthesizing electronic health records using improved generative adversarial networks. Journal of the American Medical Informatics Association, 26(3):228–241, 2019. 





[25] Cristóbal Esteban, Stephanie L Hyland, and Gunnar Rätsch. Real-valued (medical) time series generation with recurrent conditional gans. arXiv preprint arXiv:1706.02633, 2017. 





[26] Frederico Lopes, Carlos Soares, and Paulo Cortez. Privatectgan: Adapting gan for privacy-aware tabular data sharing. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 169–180. Springer, 2023. 





[27] Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. Tabddpm: Modelling tabular data with diffusion models. In International Conference on Machine Learning, pages 17564–17579. PMLR, 2023. 





[28] Sanket Kamthe, Samuel Assefa, and Marc Deisenroth. Copula flows for synthetic data generation. arXiv preprint arXiv:2101.00598, 2021. 





[29] Jayoung Kim, Chaejeong Lee, and Noseong Park. Stasy: Score-based tabular data synthesis. arXiv preprint arXiv:2210.04018, 2022. 





[30] Hengrui Zhang, Jiani Zhang, Balasubramaniam Srinivasan, Zhengyuan Shen, Xiao Qin, Christos Faloutsos, Huzefa Rangwala, and George Karypis. Mixed-type tabular data synthesis with score-based diffusion in latent space. arXiv preprint arXiv:2310.09656, 2023. 





[31] Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, and Gjergji Kasneci. Language models are realistic tabular data generators. arXiv preprint arXiv:2210.06280, 2022. 





[32] Yuxin Wang, Duanyu Feng, Yongfu Dai, Zhengyu Chen, Jimin Huang, Sophia Ananiadou, Qianqian Xie, and Hao Wang. Harmonic: Harnessing llms for tabular data synthesis and privacy protection. arXiv preprint arXiv:2408.02927, 2024. 





[33] Bosheng Ding, Chengwei Qin, Ruochen Zhao, Tianze Luo, Xinze Li, Guizhen Chen, Wenhan Xia, Junjie Hu, Anh Tuan Luu, and Shafiq Joty. Data augmentation using large language models: Data perspectives, learning paradigms and challenges. arXiv preprint arXiv:2403.02990, 2024. 





[34] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with selfgenerated instructions. arXiv preprint arXiv:2212.10560, 2022. 





[35] Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Jiuxiang Gu, and Tianyi Zhou. Selective reflection-tuning: Student-selected data recycling for llm instruction-tuning. In Findings of the Association for Computational Linguistics ACL 2024, pages 16189–16211, 2024. 





[36] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022. 





[37] Daniel J Mankowitz, Andrea Michi, Anton Zhernov, Marco Gelmi, Marco Selvi, Cosmin Paduraru, Edouard Leurent, Shariq Iqbal, Jean-Baptiste Lespiau, Alex Ahern, et al. Faster sorting algorithms discovered using deep reinforcement learning. Nature, 618(7964):257–263, 2023. 





[38] Yihong Tang, Zhaokai Wang, Ao Qu, Yihao Yan, Zhaofeng Wu, Dingyi Zhuang, Jushi Kai, Kebing Hou, Xiaotong Guo, Jinhua Zhao, et al. Itinera: Integrating spatial optimization with large language models for open-domain urban itinerary planning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1413–1432, 2024. 





[39] Wang Jiawei, Renhe Jiang, Chuang Yang, Zengqing Wu, Ryosuke Shibasaki, Noboru Koshizuka, Chuan Xiao, et al. Large language models as urban residents: An llm agent framework for personal mobility generation. Advances in Neural Information Processing Systems, 37:124547–124574, 2024. 





[40] Arijit Ghosh Chowdhury and Aman Chadha. Generative data augmentation using llms improves distributional robustness in question answering. arXiv preprint arXiv:2309.06358, 2023. 





[41] Bowen Tan, Zheng Xu, Eric Xing, Zhiting Hu, and Shanshan Wu. Synthesizing privacy-preserving text data via finetuning without finetuning billion-scale llms. arXiv preprint arXiv:2503.12347, 2025. 





[42] Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, and Jiawei Han. Effective seed-guided topic discovery by integrating multiple types of contexts. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, pages 429–437, 2023. 





[43] Meng Chen, Philip Arthur, Qianyu Feng, Cong Duy Vu Hoang, Yu-Heng Hong, Mahdi Kazemi Moghaddam, Omid Nezami, Duc Thien Nguyen, Gioacchino Tangari, Duy Vu, Thanh Vu, Mark Johnson, Krishnaram Kenthapadi, Don Dharmasiri, Long Duong, and Yuan-Fang Li. Mastering the craft of data synthesis for CodeLLMs. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 12484–12500, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. 





[44] M Sklar. Fonctions de répartition à n dimensions et leurs marges. In Annales de l’ISUP, volume 8, pages 229–231, 1959. 





[45] Openai. Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/, 2025. 





[46] Takehiro Kashiyama, Yanbo Pang, and Yoshihide Sekimoto. Open pflow: Creation and evaluation of an open dataset for typical people mass movement in urban areas. Transportation research part C: emerging technologies, 85:249–267, 2017. 





[47] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 





[48] Brian d’Alessandro, Cathy O’Neil, and Tom LaGatta. Conscientious classification: A data scientist’s guide to discrimination-aware classification. Big data, 5(2):120–134, 2017. 





[49] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 



# APPENDIX

The appendix is organized into several sections, each focusing on different aspects of the work. Section details the datasets used, Section describes the experimental setups and baselines, and Section presents supplementary results. Section provides the theoretical analysis, while Section outlines the prompts used in our framework. Finally, Section offers running examples to illustrate the methodology. 

# Table of Contents

A Data 12 

A.1 E-Commerce Transactions 12 

A.2 Population 14 

A.3 Mobility 17 

B Experiment Setups 18 

B.1 Implementation Details 18 

B.2 Baselines 18 

B.2.1 E-Commerce Transaction Synthesis 20 

B.2.2 Population Synthesis 20 

C Supplementary Results 22 

C.1 E-Commerce Transaction Synthesis 22 

C.2 Population Synthesis 24 

C.3 Mobility Synthesis . 27 

D Theoratical Analysis . 28 

D.1 Convergence Analysis 28 

E Prompts 29 

E.1 LLM Proposal Sampling 29 

E.2 LLM as a Copula . 30 

F Running Examples .31 

F.1 Discrepancy-Guided Generation . 31 

F.2 Example of a Single Iteration 32 

# E-Commerce Transactions

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/456b68943db1b8f176c1c9f6397f92833fd114bfecd3538fc3c313baadbf79f3.jpg)



Figure 5: Bayesian network representing the generative process of e-commerce transactions.


Data Generating Process. Let $v _ { A }$ , vG , vL , vC , $v _ { X }$ , and $v _ { M }$ denote the random variables for user age, gender, location tier, product category, price, and payment method, respectively. We assume the following generative mechanisms: 

$1 . \quad v _ { A } \sim \Big ( \sum _ { i = 1 } ^ { 3 } \pi _ { i } \mathcal { N } ( \mu _ { i } , \phi _ { i } ^ { 2 } ) \Big ) \ : \mathbf { 1 } _ { [ 1 8 , 9 0 ] } ( v _ { A } ) ,$ 

$\mathsf { w h e r e } \sum _ { i = 1 } ^ { 3 } \pi _ { i } = 1 , \pi _ { i } > 0 ,$ 

2. $v _ { G } \sim \mathrm { d i s c r e t e } \big ( p _ { G } ^ { ( \mathsf { m a l e } ) } , p _ { G } ^ { ( \mathsf { f e m a l e } ) } \big ) ,$ p(female), 

$$
p _ {G} ^ {\mathrm {(m a l e)}} + p _ {G} ^ {\mathrm {(f e m a l e)}} = 1,
$$

$v _ { L } \sim \mathrm { d i s c r e t e } \bigl ( p _ { L } ^ { ( 1 ) } , p _ { L } ^ { ( 2 ) } , p _ { L } ^ { ( 3 ) } \bigr ) ,$ 

$$
\sum_ {k = 1} ^ {3} p _ {L} ^ {(k)} = 1,
$$

4. $v _ { C } \mid ( v _ { A } , v _ { G } ) \sim \mathrm { d i s c r e t e } \big ( p _ { C } ^ { ( g ( v _ { A } ) , v _ { G } ) } \big ) ,$ 

$$
g (v _ {A}) = \left\{ \begin{array}{l l} \text {y o u n g}, & v _ {A} <   3 5, \\ \text {m i d d l e}, & 3 5 \leq v _ {A} <   5 5, \\ \text {o l d}, & v _ {A} \geq 5 5, \end{array} \right.
$$

5. vX | vC = c ∼ N (µc, ϕ2c ) 1[ℓ,u](vX ), 

[ℓ, u] denotes the valid price range, 

6. vM | vL = k ∼ discrete p(k)M , 

$$
\sum_ {j} p _ {M} ^ {(k)} (j) = 1 \quad \forall k.
$$

Under this model, the joint distribution factorizes as 

$$
p \left(v _ {A}, v _ {G}, v _ {L}, v _ {C}, v _ {X}, v _ {M}\right) = p \left(v _ {A}\right) p \left(v _ {G}\right) p \left(v _ {L}\right) p \left(v _ {C} \mid v _ {A}, v _ {G}\right) p \left(v _ {X} \mid v _ {C}\right) p \left(v _ {M} \mid v _ {L}\right),
$$

enforcing the exact conditional independencies of the Bayesian network in Figure 5. 

Implementation Settings. In our experiments, we instantiate the model with the following parameter values: 

• Age $( v _ { A } )$ : drawn from a mixture of three truncated Gaussian distributions, 

$$
v _ {A} \sim \sum_ {i = 1} ^ {3} \pi_ {i} \mathcal {N} \left(\mu_ {i}, \phi_ {i} ^ {2}\right) \big | _ {[ 1 8, 9 0 ]},
$$

with mixture weights π = (0.5, 0.45, 0.15), means $\mu =$ (24, 38, 55), and standard deviations $\phi = ( 6 , 1 5 , 1 0 )$ 

• Gender $( v _ { G } )$ : categorical variable over {Male, Female} with probabilities 

$$
p (v _ {G}) = (0. 4 5, 0. 5 5).
$$

• Location Tier $( v _ { L } )$ : categorical variable over {Developed, Developing} with probabilities 

$$
p (v _ {L}) = (0. 4, 0. 6).
$$

• Product Category $( v _ { C } )$ : categorical variable over {Electronics,Apparel,Food & Beverages,Furniture & Appliances}. Conditional probabilities $p ( v _ { C } \mid g , v _ { G } )$ are defined for age group $g \in$ {young, middle, old} and gender vG: 


Table 4: Conditional distributions $p ( v _ { C } \mid g , v _ { G } )$


<table><tr><td>Age Group</td><td>Gender</td><td>Electronics</td><td>Apparel</td><td>Food &amp; Beverages</td><td>Furniture &amp; Appliances</td></tr><tr><td>Young</td><td>Male</td><td>0.50</td><td>0.25</td><td>0.05</td><td>0.20</td></tr><tr><td>Young</td><td>Female</td><td>0.20</td><td>0.50</td><td>0.05</td><td>0.25</td></tr><tr><td>Middle</td><td>Male</td><td>0.20</td><td>0.10</td><td>0.50</td><td>0.20</td></tr><tr><td>Middle</td><td>Female</td><td>0.10</td><td>0.20</td><td>0.55</td><td>0.15</td></tr><tr><td>Old</td><td>Male</td><td>0.10</td><td>0.10</td><td>0.65</td><td>0.15</td></tr><tr><td>Old</td><td>Female</td><td>0.10</td><td>0.10</td><td>0.60</td><td>0.20</td></tr></table>

• Price $( v _ { X } )$ : conditional on category c, drawn from a truncated Gaussian 

$$
v _ {X} \sim \mathcal {N} (\mu_ {c}, \phi_ {c} ^ {2}) \big | _ {[ 0, 2 0 0 0 ]},
$$

with parameters 

$$
\left(\mu_ {c}, \phi_ {c}\right) \in \left\{\left(8 0 0, 1 0 0 0\right), \left(1 0 0, 5 0 0\right), \left(1 0 0, 4 0 0\right), \left(2 0 0, 5 0 0\right) \right\},
$$

corresponding respectively to {Electronics, Apparel, Food & Beverages, Furniture & Appliances}. 

• Payment Method $\left( v _ { M } \right)$ : categorical variable over {Online Payment, Cash on Delivery} with probabilities 

$$
p (v _ {M} \mid v _ {L} = \text {D e v e l o p e d}) = (0. 7 0, 0. 3 0), \quad p (v _ {M} \mid v _ {L} = \text {D e v e l o p i n g}) = (0. 4 0, 0. 6 0).
$$

Derived Economic Variables. To evaluate the practical utility of synthetic data, we introduce two composite variables that simulate realistic segmentation and value estimation tasks. These targets are not included in the data generation process and are computed entirely post hoc. From a machine learning perspective, they serve to test whether models trained on synthetic data can support meaningful discriminative tasks involving high-order, nonlinear feature interactions that resemble real-world decision logic. The variables are constructed to reflect common use cases in business and marketing analytics: discount targeting and customer lifetime value estimation. Both combine domain knowledge and structural dependencies across price, age, payment behavior, and product category into proxy labels that require the generative model to faithfully reconstruct multiple conditional pathways. 

Discount Propensity (d). This variable approximates price sensitivity, inspired by the concept of price elasticity of demand. Instead of modeling actual demand shifts, we simulate a "discount responsiveness score" based on how far a transaction price deviates from the expected norm and is modified by age and behavioral context. We first compute a z-score for the transaction price within its category: 

$$
z = \frac {v _ {X} - \mu_ {v _ {C}}}{\phi_ {v _ {C}}},
$$

where $\mu { } _ { v _ { C } }$ and $\phi _ { v _ { C } }$ are the category-wise mean and standard deviation. The composite score is then 

$$
S = - \tanh (z) + 0. 0 1 \cdot \frac {(v _ {A} - 3 5) ^ {2}}{1 0 0} + 0. 5 \cdot {\bf 1} \{v _ {M} = \mathrm {C O D} \} + 0. 3 \cdot {\bf 1} \{v _ {L} = \mathrm {D e v e l o p i n g} \}.
$$

Each term encodes an interpretable behavioral signal: tanh(z) models diminishing sensitivity to extreme price deviations, the quadratic age term increases discount likelihood for younger and older users, peaking at age 35, COD users $( + 0 . 5 )$ and those in developing regions $( + 0 . 3 )$ are assumed to be more price-conscious based on consumer behavior research. The 0.01 scale factor 

balances the influence of the age component, keeping it comparable to binary modifiers. The final categorization is: 

$$
d = \left\{ \begin{array}{l l} \text {H i g h}, & S > 1, \\ \text {L o w}, & S <   - 1, \\ \text {M i d}, & \text {o t h e r w i s e}. \end{array} \right.
$$

Lifetime Value Band (ℓ). This variable acts as a simplified proxy for customer lifetime value (CLV), an important metric for revenue forecasting and segmentation. CLV is typically defined as the discounted sum of future profits; we approximate this by incorporating transaction amount, demographic potential, and purchase channel/product effects. We compute a proxy score $\ell _ { 0 }$ that reflects expected customer value as a function of transaction size, demographic potential, and behavioral modifiers: 

$$
\ell_ {0} = \frac {\sqrt {v _ {X}} \cdot w _ {m}}{\log (1 + | v _ {A} - 3 5 |) + 1} \cdot w _ {c},
$$

where $w _ { m }$ is a payment-method multiplier (1.2 for online, 0.85 for COD), encoding expected retention and conversion ease, $w _ { c }$ is a product-category multiplier: 

$$
w _ {c} = \left\{ \begin{array}{l l} 1. 3, & v _ {C} = \text {E l e c t r o n i c s}, \\ 1. 1, & v _ {C} = \text {A p p a r e l}, \\ 0. 9, & v _ {C} = \text {F o o d \& B e v e r a g e s}, \\ 1. 4, & v _ {C} = \text {F u r n i t u r e \& A p p l i a n c e s}. \end{array} \right.
$$

The age-adjusted denominator reduces the projected value for extreme age groups, approximating long-term activity potential. We then discretize the proxy score $\ell _ { 0 }$ into ordinal bands to reflect interpretable customer value segments: 

$$
\ell = \left\{ \begin{array}{l l} \text {H i g h}, & \ell_ {0} > 2 0, \\ \text {L o w}, & \ell_ {0} <   1 0, \\ \text {M i d}, & \text {o t h e r w i s e}. \end{array} \right.
$$

The thresholds and coefficients used are fixed, interpretable, and manually defined. They are not optimized on the data and are instead chosen to encode realistic, domain-informed assumptions. All values are fixed across all experiments and applied identically to both real and synthetic datasets to ensure a fair and reproducible evaluation of downstream discriminative utility. 

# Population

Data We use the 2023 American Community Survey (ACS) 1-Year Public Use Microdata Sample (PUMS) , a large-scale, nationally representative dataset released by the U.S. Census Bureau. The ACS is an ongoing survey program that provides detailed annual data on demographic, economic, housing, and social attributes of the U.S. population. Its microdata records offer rich insights that inform resource allocation, urban planning, public policy, and equity analysis at multiple geographic levels. 

The 2023 ACS 1-Year PUMS includes anonymized household- and person-level records, making it well-suited for population synthesis. For this study, we extract data from households in South Carolina. We retain both household-level variables (e.g., number of persons, income, tenure, household type, vehicle ownership) and person-level attributes (e.g., age, race, education, employment, income). Since households vary in size and structure, the resulting data are inherently unstructured. 

To ensure consistency and quality, we apply the following preprocessing steps: 

• Household filtering: We include only family households with 2 to 5 members, drop entries with missing key fields, and normalize housing tenure and household type into interpretable categories. 

• Person filtering: We retain individuals belonging to valid households and ensure non-missing values for core attributes like age and race. Records with anomalous or censored race categories or negative income are excluded. 

• Variable recoding: Education levels are grouped into broader bands (e.g., high school, bachelor, master), employment status is collapsed, and race is mapped into simplified categories. 

• Household-person consistency: Only households whose member count matches the reported size (NP) are retained, ensuring alignment between household and person tables. 

• Final dataset: We obtain a cleaned population microdata sample consisting of 15,380 households, each associated with individuallevel demographic and socioeconomic details. 

The resulting dataset contains both household- and person-level records. The variables are: 

# Household-level variables:

• num_persons (2∼5): Number of people in the household. 

• housing_tenure (Owned, Rented): Indicates whether the household owns or rents their home. 

• householder_type (Couple, Single Male, Single Female): Household composition based on adult structure. 

• num_vehicles (0∼6): Number of vehicles available to the household. 

# Person-level variables:

• age (0∼99): Age of the individual. 

• race (White, Black, Indian, Asian, Combined): Simplified racial categories derived from ACS coding. 

• education (Preschool, Elementary, High School, Bachelor, Master, Doctorate): Highest educational attainment. 

• employment (Under 16, Employed, Unemployed, Military): Employment status grouped into meaningful categories. 

• income (0∼4,209,995): Individual annual income in U.S. dollars. 

Due to household size variation, the data is inherently unstructured, with each household containing a variable number of individual records. This structure poses a realistic and valuable challenge for generative models aimed at cross-level joint distribution synthesis. More detailed variable descriptions can be found at this link. 

Query Definitions Accurate synthetic population data must enable meaningful analysis that informs real-world policy and planning. This includes identifying economic disparities, assessing social vulnerability, and supporting decisions on housing, mobility, and public services. To evaluate the practical value of such data, we define 16 policy-relevant queries grouped into six thematic categories. Each query includes a description and an explanation of its social significance. 

# Equity

(1) median_income_black_households 

Description: Median total household income for households with at least one Black member. 

Social significance: This indicator measures racial income inequality and highlights the economic disparities that affect Black house holds. 

(2) prop_multigenerational_racial 

Description: Proportion of multigenerational households, defined as those with both children under 18 and elderly members aged 65 or older, that include at least one non-White individual. 

Social significance: This indicator sheds light on racial patterns in multigenerational living and provides insight into caregiving structures and household composition across different racial groups. 

# Vulnerability

(1) prop_elderly_poverty 

Description: Proportion of households that include at least one member aged 65 or older and have a per-person income below 12,500. 

Social significance: This indicator reflects the economic vulnerability of older adults and helps identify households at risk of poverty in later life. 

(2) prop_female_headed_poverty 

Description: Proportion of single female-headed households with a per-person income below 15,000. 

Social significance: This indicator highlights the financial challenges faced by single women who lead households, pointing to gendered dimensions of poverty. 

(3) prop_high_dependency_ratio 

Description: Proportion of households in which the combined number of children and elderly members exceeds the number of working-age members. 

Social significance: This indicator identifies households that carry a high care burden and may be more susceptible to financial and caregiving stress. 

# Employment

(1) prop_high_edu_unemployed 

Description: Proportion of households where the highest level of education attained is a Master’s degree or Doctorate, and no household members are employed. 

Social significance: This indicator highlights potential underemployment or barriers to workforce participation among highly educated individuals. 

(2) prop_dual_earner_couples 

Description: Proportion of couple households where both adults are employed. 

Social significance: This indicator reflects the prevalence of dual-income households and offers insight into labor force participation within traditional family structures. 

(3) avg_income_per_person_dual_earner 

Description: Average per-person income in couple households where both adults are employed. 

Social significance: This indicator captures the financial outcomes associated with dual-earning and helps assess economic inequality among working families. 

# Household

(1) age_gap_owners_vs_renters 

Description: Difference in the average age of household members between owner-occupied and renter-occupied households. 

Social significance: This indicator reflects how age distribution relates to housing tenure and supports analysis of life stage patterns in housing access. 

(2) prop_multigenerational 

Description: Proportion of households that include both children under the age of 18 and elderly members aged 65 or older. 

Social significance: This indicator highlights the prevalence of multigenerational living arrangements and the caregiving responsibilities they may involve. 

# Demographics

(1) median_avg_age 

Description: Median value of the average age of household members across all households. 

Social significance: This indicator provides a summary measure of age distribution at the household level and supports analysis related to population aging and demographic representation. 

(2) prop_families_with_children 

Description: Proportion of households that include at least one child under the age of 18. 

Social significance: This indicator reflects the presence of families with children and helps identify demand for youth-oriented services and policies. 

# Mobility

(1) prop_no_vehicle_low_income 

Description: Proportion of households with no vehicles and a per-person income below 15,000. 

Social significance: This indicator measures transportation disadvantage among low-income households and highlights potential barriers to employment, healthcare, and other essential services. 

(2) prop_child_no_vehicle 

Description: Proportion of households with at least one child under the age of 18 and no vehicles. 

Social significance: This indicator identifies families that may face challenges accessing child care, schools, and other child-related services due to limited transportation. 

(3) prop_elderly_no_vehicle 

Description: Proportion of households with at least one member aged 65 or older and no vehicles. 

Social significance: This indicator highlights potential mobility limitations and the risk of social isolation among older adults without access to private transportation. 

(4) prop_high_vehicle_high_income 

Description: Proportion of households that own three or more vehicles and have a per-person income greater than 50,000. 

Social significance: This indicator reflects concentrations of material wealth and can provide insight into patterns of resource consumption and their potential environmental impact. 

# Mobility

Data Sources. To construct our synthetic mobility task, we integrate two complementary datasets capturing different aspects of urban movement. The first is OpenPFLOW46, a high-resolution GPS trajectory dataset collected in the Tokyo metropolitan area. It provides detailed logs of individual movement traces, including agent ID, timestamp, and location (latitude and longitude). However, OpenPFLOW lacks semantic labels indicating trip purpose or activity. To address this, we incorporate external behavioral data from LLMob 39, which uses Foursquare check-ins to infer time-activity distributions across categories (e.g., “Shop & Service,” “Food,” “Travel & Transport”) ag gregated by time-of-day. Combining these two sources allows us to model not only where and when people move, but also why, a key requirement for realistic and policy-relevant mobility synthesis. 

Geographic Scope and Temporal Binning. We focus on a central urban region in Tokyo, bounded by longitude [139.6726, 139.8896] and latitude [35.6004, 35.7788]. This area encompasses both dense commercial hubs 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/d8bf7fe4b0af07846be7d1ee08b0b9970e0b473c05fb2c2a172249e779575a16.jpg)



Figure 6: Study area within the Tokyo metropolitan region.


and residential zones, offering diverse mobility patterns, as shown in Figure 6. Time is discretized into seven semantically meaningful intervals based on established guidelines in activity-based travel modeling: 0∼6, $6 \sim 9$ , ${ 9 \sim } 1 2$ , $1 2 { \sim } 1 4$ , $1 4 { \sim } 1 7$ , $1 7 { \sim } 2 0$ , $2 0 { \sim } 2 4$ . These bins reflect common human routines (e.g., commuting peaks, lunch breaks, nighttime activity) and support alignment with LLMob activity distributions. 

Grid-Based Spatial Representation. To convert raw GPS points into a spatially structured format, we partition the study area into a uniform grid of square regions (2km resolution). Each trip’s origin and destination are mapped to regions via geometric projection. This discretization supports several goals: (1) privacy, by abstracting fine-grained trajectories; (2) compatibility with OD (origin-destination) matrix modeling; and (3) alignment with spatial units used in real-world planning. 

Trip Segmentation and Transport Mode Assignment. To construct trip-level records from raw trajectory data, we first segment each agent’s movement timeline into individual trips by identifying transition points where stationary periods are followed by motion. This segmentation is based on temporal gaps and changes in movement signals. From each trip, we extract its departure and arrival coordinates, time of occurrence, and associated transport mode. However, OpenPFLOW’s original mode labels are often noisy, incomplete, or unreliable, which limits their use for structural modeling and evaluation. 

To address this, we reassign transport modes using a distance-sensitive probabilistic model. Specifically, we calculate the great-circle distance between the origin and destination regions of each trip and assign a transport mode based on a calibrated distribution that reflects common behavioral patterns. Shorter trips are more likely to be labeled as walking, mid-range trips as biking, and longer trips as driving. This approach improves behavioral realism and produces a mode variable that is both interpretable and structurally informative. Placeholder trips and intra-region trips are removed to ensure all retained trips exhibit spatial displacement and meaningful mode behavior. The resulting dataset serves as a consistent and tractable basis for synthesis and evaluation. 

Activity. Since OpenPFLOW lacks activity labels, we do not annotate real trips with activity types. Instead, we leverage LLMob’s time-conditioned activity distribution as external structural guidance. This distribution is used as a target macro-statistic during proposal generation, encouraging synthetic trips to match realistic time-activity patterns without directly assigning activity labels to observed data. 

# Outputs. The final dataset includes:

• A cleaned trip table with fields: time_str, activity, od, and transport. 

• Grid region definitions and metadata (region size, bounds, projection). 

• OD matrices per time bin, capturing trip flows between region pairs. 

• A region-region distance matrix used for spatial modeling and transport inference. 

• A per-region top-k activity summary describing local land use patterns. 

Design Rationale. This mixed-source design reflects a practical reality: in many urban contexts, movement traces (from GPS or transit data) and semantic behaviors (from points of interest or social platforms) are available from separate sources. Synthesizing mobility data that integrates both dimensions requires models to learn from partially aligned, heterogeneous inputs. Our preprocessing aims to bridge this gap, transforming fragmented empirical records into a coherent, interpretable format that supports structure-aware synthesis. By modeling OD patterns, temporal rhythms, transport behavior, and activity semantics jointly, this setup enables robust evaluation of synthetic data quality from both spatial and behavioral perspectives. 

# EXPERIMENT SETUPS

# Implementation Details

Macro-statistics For continuous variables, we adopt a hierarchical, quantile-based binning approach to construct robust and interpretable macro-statistics. In general practice, macro-statistics are defined directly according to the available target specification 

(e.g., survey tables or census summaries). In our experiments, however, since we require real micro-record data to compare against baselines, we compute the corresponding macro-statistics from the available ground-truth datasets. More broadly, when microrecords are available in real-world contexts, this same procedure can be applied to derive finer-grained macro-statistics, enabling more sensitive discrepancy evaluation and correction. 

Specifically, for each continuous variable, the observed value range is partitioned into a fixed number of main bins (by default, 6) using quantiles of the real data distribution, such that each bin contains approximately the same number of records. This avoids bin sparsity and ensures fair representation of all regions of the distribution, including heavy tails and outliers. 

To further capture local distributional features, we perform a secondary refinement step. The main bin with the largest positive discrepancy between real and synthetic data (i.e., where the real proportion exceeds the synthetic proportion by the largest margin) is selected for subdivision. Within this bin, we create a fixed number of sub-bins using uniform spacing over its range. Frequencies of these sub-bins are then normalized so that their sum equals the original frequency of the parent bin, ensuring consistency. 

This two-stage binning scheme highlights both global and local mismatches between real and synthetic data, enabling more sensitive detection and correction of model errors. The quantile-based main bins ensure stability across diverse datasets, while sub-bin refinement targets the most relevant region for detailed adjustment. 

All discrete variables (and discretized bins of continuous variables) are then summarized as frequency tables. For joint macrostatistics, we compute empirical contingency tables over selected groups of variables, applying the same binning strategy to any continuous components. This unified, type-agnostic representation supports scalable synthesis and provides interpretable signals for LLM-guided correction. 

Discrepancy The discrepancy measure $Q ( \cdot , \cdot )$ between target and synthetic data is computed as the difference between their categorical frequency tables, at both marginal and joint levels. In practice, since the synthesis process can only add records rather than remove them, we focus on positive discrepancies, that is, bins where the target proportion exceeds the synthetic proportion. Formally, for a target statistic $\phi$ with synthetic estimate $\hat { \phi } _ { i }$ , the discrepancy is defined as 

$$
Q (\hat {\phi}, \phi) = \phi - \hat {\phi},
$$

measured entry-wise across all bins or contingency cells. This formulation ensures that discrepancy signals always correspond to “underrepresented” regions in the synthetic dataset, which are precisely the areas actionable by further generation. 

Crucially, because our macro-statistics are constructed from interpretable bins, every discrepancy can be attributed to a specific value range (for continuous variables), category (for discrete variables), or a combination thereof (for joints). This attributability property is essential for the iterative synthesis loop: it pinpoints exactly where the synthetic data underestimates the target distribution and guides the LLM to generate corrective records in those regions, rather than making undirected global adjustments. 

Hyperparameters We generate 5 proposals per iteration for a total of 100 iterations. In each iteration, 3 joint variable combinations are inferred for grounding. LLM inference is performed using GPT-4.1-nano with a temperature of 0.8, and Qwen2.5-7B-Instruct via the VLLM47 framework. Generation hyperparameters are as follows: max_new_tokens $=$ 2048, temperature = 0.7, top_k $=$ 20, and top $\mathtt { \mathtt { p } } = 0 . 9 8$ . 

For macro-statistics and contingency calculations, if only partial target statistics (e.g., available macro-statistics) are accessible, we base the calculations on these statistics. However, if the full data is available, we apply a two-level quantile-based binning scheme for continuous variables, with 6 main bins and 8 sub-bins per main bin, capturing both global and local structure. Discrete variables and discretized bins are handled uniformly when computing marginal and joint frequency tables. 

All experiments are repeated at least three times, with the mean result reported. The uncertainty in our results arises from the stochasticity of LLM generation. 

Hardware All experiments are conducted on a server equipped with an Intel Xeon E5-2698 v4 CPU (40 threads), 252 GB of RAM, and four NVIDIA Tesla V100 GPUs with 32 GB of memory each. 

# Baselines

# E-Commerce Transaction Synthesis

We briefly summarize the baseline models used in our experiments: 

TVAE23 Tabular Variational Autoencoder (TVAE) models tabular data by encoding mixed-type variables into a continuous latent space via a VAE framework. It supports conditional sampling and employs Gumbel-Softmax reparameterization for discrete features. TVAE captures global structure but tends to oversmooth discrete modes, especially under imbalanced categories. 

CTGAN23 Conditional Tabular GAN (CTGAN) improves over TVAE by introducing a conditional GAN architecture. It uses modespecific normalization and a balanced training sampler to enhance rare-category representation. While improving fidelity and diversity, CTGAN suffers from the inherent instability of GAN training and mode collapse. 

CopulaGAN48 CopulaGAN integrates copula-based modeling into the GAN framework to separate marginal estimation from dependency modeling. By embedding copula regularization into the loss function, it improves the alignment between synthetic and real data in both marginal and joint distributions. This structure-aware regularization enhances statistical fidelity across variable combinations. 

GReaT 31 GReaT reformulates tabular synthesis as a sequence modeling task, using an autoregressive Transformer trained on linearized table rows. It supports pretraining and zero-shot generation, demonstrating strong generalization. However, it lacks explicit distributional control, and statistical alignment with the real data is not guaranteed. 

TabSyn 30 TabSyn combines a VAE encoder with a score-based diffusion model in the latent space. Mixed-type tabular data are first embedded into a structured continuous space, where a denoising diffusion model performs generative sampling. This hybrid design ensures high fidelity, fast sampling, and better category preservation. TabSyn achieves strong performance on mixed-type datasets with improved statistical accuracy and diversity. 

# Population Synthesis

CP (Candecomp/Parafac) Tensor Factorization The CP baseline discretizes the joint household-person population into a highdimensional contingency tensor $\mathcal { X } \in \mathbb { R } ^ { d _ { 1 } \times \cdots \times d _ { K } }$ , where each axis corresponds to a household or person attribute. A non-negative CP decomposition is applied: 

$$
\hat {\mathcal {X}} _ {i _ {1}, \dots , i _ {K}} = \sum_ {r = 1} ^ {R} \lambda_ {r} \prod_ {k = 1} ^ {K} A _ {i _ {k}, r} ^ {(k)},
$$

where $\lambda _ { r }$ denotes the weight of component $r$ , and $A ^ { ( k ) }$ are factor matrices. After normalization, the model yields a mixture-ofproduct-of-categoricals (MPC) distribution: 

$$
Q _ {\mathrm {C P}} (i) = \sum_ {r = 1} ^ {R} w _ {r} \prod_ {k = 1} ^ {K} a _ {i _ {k}, r} ^ {(k)}.
$$

Households are assigned to their most likely latent group via $\begin{array} { r } { g = \arg \operatorname* { m a x } _ { r } w _ { r } \prod _ { k } a _ { i _ { k } , r } ^ { ( k ) } } \end{array}$ , and within each group, a person-level nonnegative CP factorization model $M _ { g }$ is fit for joint household-member sampling. 

HMM and NVI We presents two hierarchical generative baselines used for structured population synthesis: HMM (Product Multinomial Hierarchical Mixture Model) and NVI (Neural Variational Inference). Both approaches are based on a shared latent graphical model but differ in the inference procedure and parameterization. 

The generative process assumes a two-layer structure: 

$$
\begin{array}{l} z _ {i} \sim \operatorname {C a t} (\lambda), \\ x _ {i} ^ {(k)} \mid z _ {i} = g \sim \operatorname {C a t} \left(\Phi_ {g} ^ {(k)}\right), \\ z _ {i j} \sim \operatorname {C a t} (\mu_ {g}), \\ x _ {i j} ^ {(\ell)} \mid z _ {i j} = m \sim \operatorname {C a t} \left(\theta_ {g m} ^ {(\ell)}\right), \\ \end{array}
$$

where $z _ { i }$ is the latent household class and $z _ { i j }$ is the latent member class of person $j$ in household i. All conditional distributions are categorical, regularized by symmetric Dirichlet priors. 

HMM: Product Multinomial Hierarchical Mixture Model The HMM baseline employs the Expectation-Maximization (EM) algorithm for parameter estimation, alternating between posterior updates of latent variables and maximization steps. 

E-step (posterior responsibilities): 

$$
\begin{array}{l} \gamma_ {i} ^ {g} = \operatorname * {P r} (z _ {i} = g \mid \mathrm {d a t a}), \\ \rho_ {i j} ^ {m} = \Pr (z _ {i j} = m \mid z _ {i} = g, \text {d a t a}). \\ \end{array}
$$

M-step (parameter updates): 

$$
\begin{array}{l} \lambda_ {g} \gets \frac {1}{N} \sum_ {i} \gamma_ {i} ^ {g}, \\ \mu_ {g m} \gets \frac {\sum_ {i j} \gamma_ {i} ^ {g} \rho_ {i j} ^ {m}}{\sum_ {i j} \gamma_ {i} ^ {g}}, \\ \Phi_ {g} ^ {(k)} (c) \leftarrow \frac {\sum_ {i} \gamma_ {i} ^ {g} \mathbb {I} (x _ {i} ^ {(k)} = c)}{\sum_ {i} \gamma_ {i} ^ {g}}, \\ \theta_ {g m} ^ {(\ell)} (c) \leftarrow \frac {\sum_ {i j} \gamma_ {i} ^ {g} \rho_ {i j} ^ {m} \mathbb {I} (x _ {i j} ^ {(\ell)} = c)}{\sum_ {i j} \gamma_ {i} ^ {g} \rho_ {i j} ^ {m}}. \\ \end{array}
$$

This formulation is particularly suited for moderate-scale datasets with clear latent groupings and yields interpretable household and person class structures. 

NVI: Neural Variational Inference The NVI baseline implements amortized mean-field variational inference using neural networks to parameterize posterior distributions. The variational posterior is factorized as: 

$$
q (z, z _ {1: M}) = \prod_ {i} q _ {\Phi} \left(z _ {i} \mid x _ {i}\right) \prod_ {j} q _ {\psi} \left(z _ {i j} \mid z _ {i}, x _ {i j}\right),
$$

Both $q _ { \Phi }$ and $q _ { \psi }$ are implemented using multilayer perceptrons (MLPs). Discrete latent variables $z _ { i }$ and $z _ { i j }$ are reparameterized via the Gumbel-Softmax trick: 

$$
\tilde {z} _ {i} = \operatorname {S o f t m a x} \left(\frac {\log \gamma_ {i} + g}{\tau}\right), \quad g \sim \operatorname {G u m b e l} (0, 1),
$$

with temperature $\tau$ gradually annealed to 0.1. The model is trained to maximize the evidence lower bound (ELBO): 

$$
\mathcal {L} = \mathbb {E} _ {q (z)} [ \log p (x, z) - \log q (z \mid x) ],
$$

where all categorical likelihoods are modeled using softmax-parameterized logits. Optimization is performed via mini-batch stochas-

tic gradient descent, and missing values are handled via masking or learned probabilities. This variational method provides a scalable and flexible alternative to classical EM, suitable for high-dimensional structured populations. 

# SUPPLEMENTARY RESULTS

# E-Commerce Transaction Synthesis

In this subsection, we present additional visualizations comparing the distributions of real and synthetic data. 

Synthetic Distribution Visualization In Figure 7 and Figure 8, we visualize the full marginal distributions for all individual variables as well as selected joint distributions over correlated variable pairs, comparing synthetic data generated by LlmSynthor and GReaT. These plots complement the quantitative metrics in the main paper by illustrating how well each method captures complex interactions and category-conditioned patterns. LlmSynthor shows strong visual alignment with the real data across both marginal and conditional views, especially in structured relationships such as payment method by location tier and product category by age and gender. GReaT captures some trends but tends to over-smooth (Price by Product Category) or under-represent distributional variation (Price). These comparisons highlight the qualitative fidelity of LlmSynthor in preserving both local and global statistical structure. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/28ff339f3e1bad8b2fec9e7408c4f0f0be13340fc8acf01a8e2cf32bf17cb634.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/6f6f24dd7eac322aad0b4c8431f5db94a90b043858bb61c4e97c3d10fecea023.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/63da5d958eb6849f013ffaf867e50b4e797cd74183ca3d0f6bcaa8c345da01db.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/78c72b495e4438570defc6a596230d2f3d975fcf23d3e47b81ad362179a27545.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/8f27a51f7c32f669b90fde750e18a9afe1f2e5e1ca12a417d202cdba97dcb9aa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/fa63eeb2e59d63fc03fc58b8d21083345b727de4a12237638176c7d1ffbfc3b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e90008689bd2c8c4132cd421baf92454f25a76d793c32f2714790f6e310fa1ae.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1f1a0acb0acc4e59eef1cc84fc994c5cf6f9ec563009fdfd41df7c7aa51aed5b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/307b0e3127b0127237db0b82aa6f6444d4b1faeb34f42c9ab07443068053fb1e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c84628ef2c0b83d8389b02b884514a76d5d2adecb9446b4a97f1325608d57773.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/7a3c96724ead7c1c6797376f501c33b3e69ec7ed8daedc126bc404d5cb911c32.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/4dc1204d4c44e84f26b28053181ad36bfb24b1050c95f2cc662d28e8659ef077.jpg)



Figure 7: Visualization of the synthetic distribution generated by LlmSynthor.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1fa082e8d457925ec66da407ffe28fd3b160f43fb9fe8e6063186529a52100c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/15b37c1a285ec7598cd99858aefa86fa1581a37a851a21b1391e2b15c4952044.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/08ccbaeb2eb4a455c53629f30a6cf43a5704f7d6155bd7e9634a56c9ab2dc453.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/855d8951fff9dc21b5ca94191e0a98e7b6419675072cb85f0db8805d87239e87.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/707fe7c1c039c4011c48efbbc0ac2b3ef44c2956677f2ca3bf15b76beb3016e8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ebfd072283456abdfcad18dd4694d2943c96e98833972612ec9884c255ab1442.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0ea86dce9c3049f5a0e1179630916945868cb85320815772eab579ce0554e71e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e6801ff636fdf79c8544b76fd8ffe594163aa56c0e5ca2943d8e8a7d28531c1d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/de9600a96f72b929c59213159891953c0c955c5fa668fba879920b0f34fa5078.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c9e02c57d2d07bafeb0cb0b1c6cf07f72ef8678bf8b3e2b5888624baea202372.jpg)



Figure 8: Visualization of the synthetic distribution generated by GReaT.


Convergence Analysis. Figure 9 presents empirical convergence curves tracking distributional discrepancy metrics across 100 synthesis iterations. We evaluate both marginal and joint distributions using a suite of distances, including total variation distance 

(TVD), Wasserstein distance, energy distance, and Hellinger distance. All metrics consistently decrease over time, indicating that LlmSynthor iteratively reduces structural divergence between real and synthetic data. Joint patterns such as product price by category and payment method by location tier also converge, demonstrating that high-order dependencies are faithfully preserved. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e0505820fd38d403d8c04df1ab443255ca6969ac61a7ed283d93a2ba4858a93f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/823985ae47d9c132ab4e7e746aac37dbc1807d24d3660f164d3a9e8950a9da3e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/9c01dcea89c24fa5fdbf76ab767eda0e02b74a9b7bdbf163c945eca20ea278d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/135b7527924e65d230421785d38a528524ce63e77790359bcf991ed769c5b270.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/4c08b6e6409add55c1a744352702cf8922e66d1ae159c23e8cdca489f809111e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/4225f69cf8de0b936036579f2484e2e7cf6f6580e19dc0d13717a746fe09cd90.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/68d71e301bd24eff9a327b58b644ee4e424f85fb84a70688bf8604f8386f76f7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c4d1f4488e5f0b39ffb587e2fd68164538cca810ef1499a43f9c54a4e236c519.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/01f34eae5360c464002d5c8cb500180c75bc0ba32a2b38ecd3fce63e61685c54.jpg)



Figure 9: Convergence curve showing distributional discrepancy metrics over iterations.


The Figure 10 depicts the overall mean distributional distance between real and synthetic data as a function of synthesis iterations. To compute this curve, we aggregate several distributional metrics, including total variation distance (TVD), Wasserstein distance, energy distance, maximum mean discrepancy (MMD), Hellinger distance, Jensen-Shannon divergence (JSD), and Kullback-Leibler (KL) divergence, across both marginal and joint variables. For each iteration, the mean of all selected metrics over all variables is reported. The solid line indicates this average, while the shaded regions represent uncertainty intervals: $\pm \ 1 , \pm \ 2$ , and $\pm \nobreakspace 3 \nobreakspace$ standard deviations, along with the min-max range across repeated runs. The sharp initial decrease and rapid plateau show that most statistical alignment is achieved within the first 50 iterations, with diminishing returns beyond 200. This highlights the efficiency and stability of the iterative synthesis process and provides practical guidance for setting the number of iterations. 

Efficiency Analysis. As shown in Figure 11, we compare the runtime efficiency of our method (LlmSynthor) against GReaT, a strong LLM-based baseline. Unlike GReaT, which requires training, LlmSynthor performs zero-shot synthesis via LLM prompting. For this evaluation, we generate 2,000 records in both methods. GReaT is run using a batch size of 124, the maximum size recommended in its original paper to ensure performance, and trained over 200 epochs, which we find sufficient for convergence. In contrast, LlmSynthor is evaluated using two backends: GPT-4.1-nano via API and Qwen2.5- 7B-Instruct on a single V100 GPU with 32GB memory. Our method achieves comparable or better runtime without any training overhead. The API version is notably faster due to optimized inference pipelines. In practice, further speedups are possible by increasing the number of records generated per iteration or reducing the number of refinement steps. As LLMs continue to improve in both intelligence and inference speed, LlmSynthor will become even more efficient and widely adaptable for practical data synthesis tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/994fd6842098b076b738aeccf87b93393829b90e06b4333d6e058bea13d9cbe6.jpg)



Figure 10: Mean distributional distance across synthesis iterations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/fa85cee802d659b41e10f19454a2485f76034b5ecc4fcff8c98cbfaf8de4acfb.jpg)



Figure 11: Efficiency comparison between GReaT and LlmSynthor.


Ablation: LLM Backbone Robustness. Figure 12 shows the output of LlmSynthor when using Qwen2.5-7B-Instruct as the LLM backbone instead of GPT-4.1. Despite architectural and training differences, the resulting synthetic distributions remain wellaligned with the real data across both marginal and joint views. This demonstrates the framework’s robustness to LLM choice, as long as the model possesses sufficient language understanding and compositional capacity. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/211db8e93e93df332ad77cfa306688924e6565a6c9ac4ce5c41e2e0ffa91b842.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/70e447d900926c98b53f62270e0821cbb03bd3a0d88190fee58e42cf34e77dcf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/8efcfbaa2fc18ca1a6e3e8d424f642fa524ec39d3d6c62554bbd3e5e208952ac.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/885c2d98baa79acfb3b298a83b040f6240dc57bd92c8737d464d6256b1a17f68.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b56f2b2d241b7e17a7bc4bb1ab9793942c57d0630e1cf0b3bcd556ac6f0d3675.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/7b55697f495b634bb10e0eda353a607e7e296c2f0fa09904db4eade6969d36c4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/2a984a0df0a2867d608e0f15e15616741bc6ca7a0a544c12f9ced3a4a56ae0ed.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/9e7e40b04af242ff86d506bbe1af5ec4f5a1e4a58146d9927d1e89215ca3d205.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c6ddac512cb8f1fd37ade2cfc029d5f6d25ca0577979c9f4b300a3e9d3efc06b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e63a4b973443a5fa4d602efcf489f07f8082243468eac9de76f00c0e710e1aaf.jpg)



Figure 12: Ablation study: performance of LlmSynthor using Qwen2.5-7B-Instruct as the LLM backbone.


Ablation: Effectiveness of Variable Dependency Inference. Figure 13 presents the outcome of disabling joint structure grounding and guiding generation using only marginal statistics. While marginal distributions remain accurate, joint relationships (e.g., age-gender-product interactions) degrade noticeably. This highlights the importance of explicitly modeling structural dependencies for capturing realistic correlations, validating the need for our copula-based joint grounding strategy. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/60e538989af9074b9cd1f175293a57fad3d16da00b1d8e40d8fc42a1709dde08.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ea4ae83066da822450be47dea9cc62b4f2664b6ea19e4680f091c62f54f3822d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a0fa8d0430505a1a21ddea461603e288f05d161116a6dd797ecbb3b5ec9434c0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/f4929297e328071c60b7848f3597041629f535bce960bf29e434f63d8596537b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a1de0e2e3ab24e34d76244be3764a0706232a5ecc70a7cb26bcea8c8bb877ad8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/81c53f4328b0308ea3c739fca5f2a4c6f39f33efcf115b235b376f74ea6b134d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1af911c162402c96713f632143697f408ee813ccfb3c3fb49bf6bbe07a6f3aaa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b5d909f009974b099589142474b85ac2125e44eda486b622af07f48755973186.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/36a8f130f7da2b88845039040e0fb26dbf13ec1b4f517c648b5cf562cec29221.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/044602ac3e0b86e8e8b8437131117e20bb55857c6c3793557164d3afb901a10f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/f673a3299b6c82ff8e2cd425d790898dc60eaceabe80217e79166bde2d0caccf.jpg)



Figure 13: Ablation study: performance of LlmSynthor when guided only by marginal statistics, without joint structure grounding.


# Population Synthesis

# Distributional Comparison in Population Synthesis.

Figure 14 and 15 compare the marginal distributions of real data against synthetic data generated by NVI and LlmSynthor, respectively. Both methods maintain reasonable alignment on household-level and individual-level variables. However, LlmSynthor more closely matches real distributions, particularly on skewed variables such as income and age, as well as on categorical distribu-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/334e44b23072007c3298f6f6324ffe3faa5ac77f10d8cee1e581559e748e5f13.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5e04bf657923adee611ace7ae605222222129af949e48f008b99fa644ca9ca1a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/cc2e47b8814cd76e115e18aa9c2e40aaf349cccfe5279dc840c18b22c30247f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5cf5df223a9438b7c635e64f87c56fec45cd45dfdb55d16a2ce49061afad8731.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/bea668edd210454b4ff4ee4e880a56cc35f517a983c5a512fb9d4b01abad5fd2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b14ce9aac5b51e0aec1c122cbec17e1a5e92f223e8c7f64e75f3c9c723a0a78c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/81cdfabcaebb6001311435ef742e360284fe1f1ec78c6460c19b6ded3bc3fa0b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5593609b15b67845c31c108085c8db03a4419e6a9e9d57ab125af713758733d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/89969528475798afa41f699c8011acfdff01af31786ab90343d0c1dfbff02ffa.jpg)



Figure 14: Visualization of the synthetic distribution generated by NVI.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/68a5987ba0fe5c8f3160c03a19277c9f4dc02828a8d493f14b0cace3e13cc60e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ba90db7f3505b82fbaafaa2956ef1899c4492fe916833ae5fe0bf2ef9f666ba0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/db04bb6dab9b61b1ea545e1cd5672aaa0d4ef71016617f7f59838676c0912f08.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/afd8ecf3575de4ec3e1d29d383c25c9788c48d0c57ec18620015d95ff325e4d1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/dd4e1d085e0e9abe63a4a617dd7b63eafe8e0fb7be769dff4fd28598ede985ce.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c94383a2f794bb07bb0e8e5b9b596aec84cb65c08a01993b36dabb0a1554d21d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/61dc9134d20a81a431afaafe842b54710dfc8a39c879d64f63e589400c105bcb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/c8e81568c7ba57e7bf823c9d6f4dd964348521194e20c462c3e36a944f5a6ddf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/4bb597c901fa9a269b9ce19a8e9227f87acabe1ad33b4ef06b4b392e3c8df6b3.jpg)



Figure 15: Visualization of the synthetic distribution generated by LlmSynthor.


tions like employment and race. This suggests that the iterative structure-guided mechanism in LlmSynthor improves fidelity even in challenging, high-variance settings. 

# Full Utility Results.

Table 5 reports detailed results for all 16 utility queries used in the population synthesis evaluation, including the real values, synthetic estimates, absolute error $( \Delta )$ , and relative error (RE). These queries span equity, vulnerability, employment, demographics, and mobility-related indicators. LlmSynthor consistently achieves the lowest relative error across most metrics, confirming its ability to preserve both marginal statistics and complex structural dependencies. Notably, it significantly outperforms all baselines on sensitive or high-variance indicators such as median_income_black_households, avg_income_per_person_dual_earner, and prop_multigenerational. These results reinforce the model’s utility in supporting policy-relevant analysis with privacy-preserving synthetic data. 

# Manual Rejection Sampling for Record-realism Evaluation.

To quantitatively evaluate the realism of generated micro-records, we construct a rule-based rejection sampling framework grounded in the semantics of population data. Among possible application domains, we choose the household population synthesis setting 

<table><tr><td rowspan="2"></td><td rowspan="2">Real</td><td colspan="3">CP</td><td colspan="3">HMM</td><td colspan="3">NVI</td><td colspan="3">LLMSYNTAXOR</td></tr><tr><td>Synth.</td><td>Δ</td><td>RE</td><td>Synth.</td><td>Δ</td><td>RE</td><td>Synth.</td><td>Δ</td><td>RE</td><td>Synth.</td><td>Δ</td><td>RE</td></tr><tr><td>median_income_whiteHouseholds</td><td>58400</td><td>103128</td><td>44728</td><td>0.77</td><td>89355</td><td>30955</td><td>0.53</td><td>79100</td><td>20700</td><td>0.35</td><td>54334.5</td><td>4065.5</td><td>0.07</td></tr><tr><td>propMULTIGenerational_racial</td><td>0.02</td><td>0.18</td><td>0.17</td><td>10.81</td><td>0.18</td><td>0.16</td><td>10.47</td><td>0.18</td><td>0.16</td><td>10.63</td><td>0.01</td><td>0.01</td><td>0.43</td></tr><tr><td>prop_elderly Poverty</td><td>0.05</td><td>0.12</td><td>0.07</td><td>1.56</td><td>0.15</td><td>0.11</td><td>2.35</td><td>0.12</td><td>0.08</td><td>1.68</td><td>0.06</td><td>0.01</td><td>0.3</td></tr><tr><td>prop_female-headed_poverty</td><td>0.07</td><td>0.06</td><td>0.01</td><td>0.16</td><td>0.04</td><td>0.03</td><td>0.39</td><td>0.04</td><td>0.03</td><td>0.43</td><td>0.1</td><td>0.03</td><td>0.44</td></tr><tr><td>prop_high_edu_unemployed</td><td>0.05</td><td>0.15</td><td>0.09</td><td>1.67</td><td>0.04</td><td>0.01</td><td>0.18</td><td>0.04</td><td>0.02</td><td>0.34</td><td>0.05</td><td>0.01</td><td>0.15</td></tr><tr><td>prop dual_earner_couples</td><td>0.12</td><td>0.01</td><td>0.11</td><td>0.88</td><td>0.09</td><td>0.03</td><td>0.27</td><td>0.16</td><td>0.04</td><td>0.3</td><td>0.17</td><td>0.05</td><td>0.43</td></tr><tr><td>avg_income_per_person_dual_earner</td><td>77984.05</td><td>38113.62</td><td>39870.43</td><td>0.51</td><td>39476.37</td><td>38507.68</td><td>0.49</td><td>64007.69</td><td>13976.36</td><td>0.18</td><td>79218.42</td><td>1234.37</td><td>0.02</td></tr><tr><td>prop_families_with_child</td><td>0.32</td><td>0.4</td><td>0.08</td><td>0.25</td><td>0.41</td><td>0.09</td><td>0.29</td><td>0.42</td><td>0.11</td><td>0.34</td><td>0.27</td><td>0.05</td><td>0.16</td></tr><tr><td>propMULTIGenerational</td><td>0.03</td><td>0.22</td><td>0.19</td><td>6.53</td><td>0.17</td><td>0.14</td><td>4.92</td><td>0.2</td><td>0.17</td><td>5.82</td><td>0.03</td><td>0</td><td>0.02</td></tr><tr><td>prop_high Dependency_ratio</td><td>0.31</td><td>0.23</td><td>0.08</td><td>0.25</td><td>0.24</td><td>0.07</td><td>0.23</td><td>0.3</td><td>0.01</td><td>0.04</td><td>0.38</td><td>0.07</td><td>0.21</td></tr><tr><td>median_avg_age</td><td>48.67</td><td>44</td><td>4.67</td><td>0.1</td><td>44.67</td><td>4</td><td>0.08</td><td>43</td><td>5.67</td><td>0.12</td><td>47</td><td>1.67</td><td>0.03</td></tr><tr><td>age_gap Owners_vsRenters</td><td>14.92</td><td>0.29</td><td>14.64</td><td>0.98</td><td>-0.61</td><td>15.53</td><td>1.04</td><td>-0.11</td><td>15.03</td><td>1.01</td><td>20.65</td><td>5.72</td><td>0.38</td></tr><tr><td>prop_child_novehicle</td><td>0.01</td><td>0.03</td><td>0.02</td><td>2.05</td><td>0.01</td><td>0</td><td>0.22</td><td>0.01</td><td>0</td><td>0.18</td><td>0.01</td><td>0</td><td>0.5</td></tr><tr><td>prop_elderly_noVehicle</td><td>0.01</td><td>0.05</td><td>0.03</td><td>3.17</td><td>0.02</td><td>0.01</td><td>0.46</td><td>0.01</td><td>0</td><td>0.16</td><td>0.01</td><td>0</td><td>0.38</td></tr><tr><td>prop_high_vehicle_high_income</td><td>0.1</td><td>0.08</td><td>0.02</td><td>0.17</td><td>0.08</td><td>0.02</td><td>0.22</td><td>0.09</td><td>0.01</td><td>0.11</td><td>0.08</td><td>0.02</td><td>0.24</td></tr><tr><td>prop_noVehicle_low_income</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.5</td><td>0.01</td><td>0.01</td><td>0.53</td><td>0.01</td><td>0.01</td><td>0.51</td><td>0.02</td><td>0</td><td>0.29</td></tr></table>


Table 5: Full utility results showing synthetic values (Synth.), absolute errors $( \Delta )$ , and relative errors.


for realism evaluation because it contains rich micro-level semantics and well-defined constraints that make it relatively unambiguous to distinguish realistic from unrealistic records. For example, it is unrealistic for a household labeled as a couple to have no adults. This makes household population synthesis a particularly suitable benchmark for assessing fine-grained realism of synthetic records produced by LlmSynthor and baseline generators. 

The rejection framework is manually designed and applied uniformly across all evaluated models. It operates in two stages. Each generated household is first checked by validating every individual it contains. If any person fails person-level checks, the entire household is rejected. If all persons are valid, the household is then validated using household-level rules. Thus, a household is accepted if and only if (i) all persons are individually valid, and (ii) the household metadata and composition satisfy the predefined rules. 

# Person-level rules include:

1. Age range: age must lie within [0, 99]. 

2. Categorical validity: fields such as employment, race, and education must belong to predefined valid categories. 

# 3. Age-education consistency:

(a) individuals under 5 years must have Preschool education; 

(b) individuals under 18 must not exceed High School. 

4. Income bounds: income must be non-negative and less than 4,209,995, an empirically chosen upper limit to exclude unrealistic values. 

# Household-level rules include:

1. Person count range: num_persons must be between 2 and 5. 

2. Vehicle count range: num_vehicles must lie within [0, 6]. 

3. Categorical validity: housing_tenure and householder_type must belong to known enumerated types. 

4. Person count match: the number of listed persons must exactly match num_persons. 

5. Adult consistency for couples: households labeled Couple must contain at least two individuals aged 18 or older. 

For each rule, we record the number of evaluated records, rejection counts, and rejection rates. While the rules are manually specified and do not cover every edge case, they capture a broad range of semantically meaningful inconsistencies. The rejection framework thus provides an interpretable and reproducible basis for benchmarking micro-level realism in synthetic datasets. 

We apply this manual rejection sampling evaluation to synthetic data produced by several baselines (NVI, CP, HMM), our proposed LlmSynthor, and the real dataset. We generate 10,000 records for each model to compute the rejection rates. We report both household- and person-level rejection rates, as well as per-rule rejection rates. Table 6 summarizes the results, omitting checks where all methods had zero rejections (e.g., employment_valid, education_valid, income_range, tenure_valid). These omissions indicate that all models trivially satisfy categorical validity and numerical bounds, while the non-zero rules reveal where models tend to fail. 

<table><tr><td>Check</td><td>NVI</td><td>CP</td><td>HMM</td><td>LLMSYNTHOR</td><td>Real</td></tr><tr><td colspan="6">Overall rejection rates ↓</td></tr><tr><td>Household rejection rate</td><td>96.8 ± 0.3</td><td>73.9 ± 0.6</td><td>57.8 ± 0.5</td><td>13.3 ± 0.4</td><td>0.0 ± 0.0</td></tr><tr><td>Person rejection rate</td><td>45.9 ± 0.4</td><td>34.1 ± 0.7</td><td>27.8 ± 0.6</td><td>6.3 ± 0.2</td><td>0.0 ± 0.0</td></tr><tr><td colspan="6">Household-level checks ↓</td></tr><tr><td>householder_type_adult_consistency</td><td>4.5 ± 0.2</td><td>4.4 ± 0.3</td><td>15.2 ± 0.5</td><td>3.8 ± 0.2</td><td>0.0 ± 0.0</td></tr><tr><td>persons_all_valid</td><td>96.7 ± 0.4</td><td>73.5 ± 0.6</td><td>56.1 ± 0.7</td><td>10.7 ± 0.3</td><td>0.0 ± 0.0</td></tr><tr><td colspan="6">Person-level checks ↓</td></tr><tr><td>age_range</td><td>0.1 ± 0.0</td><td>0.5 ± 0.1</td><td>0.3 ± 0.1</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td></tr><tr><td>race_valid</td><td>0.0 ± 0.0</td><td>0.8 ± 0.1</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td></tr><tr><td>employment_age_consistency</td><td>41.6 ± 0.7</td><td>30.9 ± 0.6</td><td>24.5 ± 0.5</td><td>4.4 ± 0.2</td><td>0.0 ± 0.0</td></tr><tr><td>education_age_consistency</td><td>42.7 ± 0.8</td><td>8.1 ± 0.4</td><td>10.7 ± 0.5</td><td>2.2 ± 0.1</td><td>0.0 ± 0.0</td></tr></table>


Table 6: Rejection rates $( \% )$ by rule across methods. Checks with zero rejections in all methods are omitted for brevity.


Overall, the real dataset exhibits near-zero rejection, validating that the rule set correctly captures meaningful inconsistencies while not over-penalizing valid cases. In contrast, baseline methods produce high rejection rates. For instance, NVI fails on over $40 \%$ of persons due to education-age or employment-age mismatches, and nearly all households are rejected when aggregating these failures. CP and HMM improve over NVI but still suffer from $30 \%$ or more employment-age inconsistencies, and more than half of households are rejected. LlmSynthor markedly outperforms all baselines: household rejection drops to $1 3 . 3 \%$ and person rejection to $6 . 3 \%$ , both much closer to real data. At the person level, LlmSynthor reduces employment-age inconsistency to $4 . 4 \%$ and education-age inconsistency to $2 . 2 \%$ , an order of magnitude lower than baselines. At the household level, only $1 0 . 7 \%$ of households are rejected due to invalid persons, compared to $5 6 \%$ -97% for baselines. The only area with room for further improvement is the household size distribution, where $6 . 5 \%$ of LlmSynthor households fall outside the expected range, but this remains far less consequential than the widespread inconsistencies observed in baselines. 

These results demonstrate that LlmSynthor produces micro-records with substantially higher realism than baseline approaches. It not only avoids obvious contradictions (e.g., children with advanced degrees, unemployed minors with high incomes) but also maintains coherence at the household level, where failures compound across individuals. By narrowing the gap to real data across multiple dimensions, the rejection analysis provides strong evidence that LlmSynthor achieves realistic micro-level semantics while simultaneously enforcing macro-level statistical alignment. 

# Mobility Synthesis

Additional Mobility Visualizations. Figure 16 shows a detailed comparison of spatial-temporal flow intensity between real and synthetic data across seven time intervals throughout the day. Each map captures the aggregate origin and destination activity within the Tokyo metropolitan area during a specific time window. The synthetic data successfully preserves major spatial patterns such as morning and evening commute flows, while also capturing temporal variations in trip density. This highlights the model’s ability to maintain realistic spatiotemporal dynamics. 

Figure 17 presents additional joint distribution visualizations across key mobility attributes. The left plot illustrates the correlation between transport mode and travel distance, showing that synthetic records preserve realistic distance-dependent mode preferences (e.g., longer trips by car). The middle and right plots show marginal distributions for transport modes and time intervals, further 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/aa99863c0a420226cc92797a0ad70d16adc88f1541c20b86fbe5b1ab8f99233e.jpg)



Figure 16: Spatial-temporal flow intensity maps for real and synthetic mobility data across seven time intervals: 0-6, 6-9, 9-12, 12-14, 14-17, 17-20, and 20-24. Each map shows trip density aggregated by region to visualize commuting and activity patterns over the day.


confirming strong alignment between real and synthetic mobility behavior. Together, these results demonstrate that LlmSynthor can faithfully reproduce both spatial structure and behavioral signals critical for urban simulation and mobility planning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0ed36f76d7fc8a4169186186baa81c808363665ee1fd83e37d2bc35b37e79ecc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a2eb939a13a6b40c64a3e6352ce78a074fb4159ac54b0cc968e75ca55c746b9e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/b0e4b26958fe4cb9396da9dac89b57b54b97e94f6d0b5b45e17d7da93d01e533.jpg)



Figure 17: Distribution comparisons for mobility variables.


# THEORETICAL ANALYSIS

# Convergence Guarantee

Let $\tau _ { \nu }$ be the full contingency tensor representing the joint distribution of all variables in $\nu$ . The target macro-statistics $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ and synthetic macro-statistics $ { S _ { \mathrm { s y n t h } } ^ { ( t ) } }$ each describe a corresponding contingency tensor, which can be viewed as a projection of the macro-statistics onto $\tau _ { \nu }$ . We denote these projections as $\tau _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ and $\mathcal { T } _ { \mathrm { s y n t h } } ^ { ( t ) }$ , with $\mathcal { T } _ { \mathrm { s y n t h } } ^ { ( 0 ) } = \mathbf { 0 }$ . 

Theorem 1 (Contraction of the Mean Macro-Discrepancy). Let $\Delta _ { \star } ^ { ( t ) } \equiv Q _ { \star } ( T _ { \mathrm { s y n t h } } ^ { ( t ) } , T _ { \mathrm { t a r g e t } } ^ { \mathcal { C } } )$ with $Q _ { \star } ( x , y ) = \varphi ( x - y )$ for any norm $\varphi$ (positively homogeneous). Define the expected discrepancy: 

$$
\bar {\Delta} ^ {(t)} \equiv Q _ {\star} \big (\mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t)} ], \mathcal {T} _ {\mathrm {t a r g e t}} ^ {\mathcal {C}} \big),
$$

$$
t h e n \quad \bar {\Delta} ^ {(t + 1)} \leq (1 - \eta_ {t}) \bar {\Delta} ^ {(t)}, \quad h e n c e \quad \bar {\Delta} ^ {(t)} \xrightarrow [ t \to \infty ]{} 0 i f \sum_ {t = 0} ^ {\infty} \eta_ {t} = \infty , \quad \eta_ {t} \in (0, 1 ].
$$

This ensures that the synthetic macro-statistics converge to the target macro-statistics as the number of iterations increases. The proof is detailed below. 

Proof of Theorem 1. Write $Q _ { \star } ( x , y ) = \varphi ( x - y )$ with $\varphi$ a norm and let $y = \mathcal { T } _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ . Taking the unconditional expectation in the unbiased update yields: 

$$
\mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t + 1)} ] = \mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t)} ] + \eta_ {t} \cdot \Big (y - \mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t)} ] \Big) = (1 - \eta_ {t}) \mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t)} ] + \eta_ {t} y, \quad \eta_ {t} \in (0, 1 ].
$$

Hence, by the positive homogeneity of $\varphi$ 

$$
\bar {\Delta} ^ {(t + 1)} = \varphi \left(\mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t + 1)} ] - y\right) = \varphi \left((1 - \eta_ {t}) \left[ \mathbb {E} [ \mathcal {T} _ {\mathrm {s y n t h}} ^ {(t)} ] - y \right]\right) = (1 - \eta_ {t}) \bar {\Delta} ^ {(t)}.
$$

Iterating this gives: 

$$
\bar {\Delta} ^ {(t)} \leq \bar {\Delta} ^ {(0)} \prod_ {k <   t} (1 - \eta_ {k}),
$$

which vanishes if $\textstyle \sum _ { k } \eta _ { k } = \infty$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/053e8857f3ac0f240cef16c087db6752fd9763a7d8a38c299f6cc29ef5328cb6.jpg)


Remark (How $Q _ { \star }$ relates to our discrepancy measure $Q ]$ . In our implementation, $Q ( \cdot )$ serves as a practical and actionable interface that directly attributes the directed discrepancy (e.g., “bike: $+ 3 0 \%$ , car: $- 2 0 \% ^ { n }$ ), which can be easily interpreted by the LLM. This differs from the more abstract $Q _ { \star } ( \cdot )$ , which represents the underlying discrepancy (e.g., weighted Total Variation Distance, TVD) used in our theoretical analysis. The function $Q ( \cdot )$ is designed to provide a clear discrepancy signal, guiding the LLM in its generation of data. 

The prompts derived from discrepancy signals effectively ground the LLM to make adjustments in the generated synthetic data in a manner that aligns with the expected unbiased update toward the target macro-statistics $S _ { \mathrm { t a r g e t } } ^ { \mathcal { C } }$ . This process works iteratively, where each new batch of synthetic records is added to the dataset to reduce the discrepancy between the target and synthetic distributions. Specifically, the update rule can be viewed as $\mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t + 1 ) } = \mathcal { D } _ { \mathrm { s y n t h } } ^ { ( t ) } \cup \mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ = D(t) , where $\mathcal { D } _ { \mathrm { b a t c h } } ^ { ( t ) }$ is a corrective batch of records Dsynth synth generated to address the discrepancies in the data distribution. 

The combination of these two factors (i) the discrepancy signal derived from $Q ( \cdot )$ and (ii) the iterative data augmentation process forms a surrogate optimization process that, while not explicitly minimizing $Q _ { \star } ( \cdot )$ , achieves the same end result. Thus, by only adding data (rather than directly modifying the underlying distribution), our framework can converge to the target distribution, as guaranteed by the contraction property in the theoretical analysis. This enables efficient alignment of the synthetic data with the target macro-statistics over successive iterations, as demonstrated by the empirical convergence results in Appendix . 

# PROMPTS

This section provides the exact prompts used in our framework to guide the LLM during proposal generation and variable dependency inference. The first prompt, pproposal, instructs the LLM to generate sampleable generation plans that align with target macrostatistics. The second prompt, pcopula, is used to elicit joint dependency among variable combinations by treating the LLM as a nonparametric copula. Both prompts are designed to be type-agnostic and modular, enabling effective alignment across heterogeneous datasets. 

# LLM Proposal Sampling

# pproposal: LLM Proposal Sampling

## Output Format: 

`json 

{{ 

"n_proposals": n, 

"proposal1": {{ 

"reason": 

"proposal": "..." 

"num": n1 

}} 

"proposal2": {{ 

"reason": 

"proposal": ". 

"num": n2 

}}, 

}} 

## Information: 

**Joint Guidance:** `{joint_guide}` 

**Marginal Guidance:** `{marginal_guide}` 

**Variables:** `{data_desc}` 

# ## Rules:

Create no more than {n_proposals} proposals totaling {n_samples} samples: 

1. Each proposal must follow Joint and Marginal Guidance, do not improvise beyond the provided Guidance. 

2. The reason for each proposal should explain the realistic meaning of this proposal and how it follows the provided Guidance by referencing frequencies one by one. 

3. 'min', 'max', 'category' must use the actual values mentioned in **Variables**: Categorical variables must be a single valid candidate string (case-sensitive), and numerical variables must be a list of two integer or float numbers (e.g., [3.0, 5.1]). 

4. Do not include multiple ranges or categories per variable within a single proposal. 

5. Structure of generated data must match the **Output Format**, and all variables must be included. 

6. If a variable value has a high frequency, that value should be selected in multiple proposals. 

7. Most generated samples should, as much as possible, prioritize satisfying components that are common across the Guidances, and the “num” for each proposal should be determined based on the frequencies specified in the Guidance. 

8. Only list generation proposals; do not generate data or add extra text. 

9. Do not return an empty JSON. Do not use escaped characters such as \\r\\n, \\t, or \". Do not include any comments, markdown formatting, or explanatory text. 

Now return a pure, valid, and non-empty JSON in English that can be directly parsed by json.loads() in Python 

In our implementation, each proposal specifies a well-defined distribution for every variable. For discrete variables, the proposal directly assigns a single valid category value. For continuous variables, the proposal gives an explicit value range (for example, [3.0, 5.1]), from which values can be sampled uniformly or with another simple scheme (e.g., using numpy in Python). The num field in each proposal specifies how many records to generate from that proposal, enabling the LLM to allocate record counts across proposals based on the frequencies and guidance provided. This mechanism lets the LLM actively plan for diversity and statistical alignment among the generated records by balancing the number and distribution of proposals. Importantly, to encourage transparent and high-quality reasoning, we employ chain-of-thought 49 prompting: the LLM is instructed to explicitly explain the rationale behind each proposal, referencing frequency statistics and the provided guidance step by step. This makes the proposal process more interpretable and reliably aligned with the specified constraints. 

Importantly, this proposal format is just one practical instantiation of LLM Proposal Sampling. The framework is highly extensible. A proposal’s “distribution” could be defined not just as value ranges or categories, but as executable code, tool calls, or pointers to external generators (such as diffusion models with ControlNet for images, or other LLM-based agents for specialized content). This flexibility allows LLM Proposal Sampling to serve as a universal, high-level distributional controller, guiding external generators or hybrid pipelines toward statistically faithful and scenario-aligned synthetic data, regardless of data type or target domain. 

# LLM as a Copula for Variable Dependency Inference

pcopula: LLM as a Copula for Variable Dependency Inference 

## Information 

**Variables:** ```{data_desc}``` 

## Output Format: 

{{ 

"1": ["var1", "var2", ...], 

"2": ["var1", "var2", ...], 

}} 

### Example Output JSON: 

{{ 

```txt
"1": ["Temperature", "Humidity"],  
"2": ["Humidity", "WeatherCondition"],  
} 
```

Your task is to extract **at most {n_joints}** correlated variable groups based on the **Variables** summaries and present them in JSON format. 

1. Ensure each group contains two or more variables. 

2. Format the correlated variable groups according to the **Output Format**. 

3. Output must be valid JSON. 

# RUNNING EXAMPLES

# Discrepancy-Guided Generation

This subsection explains how the model identifies the most significant discrepancies in each iteration and samples accordingly, thereby guiding the generation of micro-records to progressively reduce the gap between the synthetic and target distributions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/1f924bf89ca9ac4c4761ca8b3a6e0e2b633d55531ce4374e3cd357e990d534e2.jpg)



Figure 18: Discrepancy-guided Iterative Synthesis. The figure illustrates how discrepancies between the target and synthetic distributions guide the generation of micro-records. In each iteration, the model samples from the top discrepancies (highlighted in red) to determine where more records should be generated. In iteration 1, the synthetic dataset is initialized as $( 0 , 0 , 0 )$ , and the model generates more records with category A to match the target distribution. In subsequent iterations, discrepancies for categories B and C are addressed by generating additional records, improving statistical alignment with the target.


This subsection illustrates how each iteration enhances the alignment between the synthetic and target distributions. 

(a) Variable Dependency Inference 

# ???????

# ## Information

Variables: 

Variables: $\mathcal { V } =$ ????, ????, ???????? 

time: integer, range 0~23 

mode: transport mode, select from [‘transit’, ‘bike’, ‘car’] 

activity: activity type, select from [‘home’, ‘work’, ‘other’] 

Macro-Statistics: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/194dd453acdf3f6bb30f1eaf2311e7cb3c9eb95a9857d536c6a37f87bd3b5d81.jpg)



Marginal


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/8fb23350d3b0c703a5015debbdeacdf4d2dd79f214fbc4b7b93571feffbedd34.jpg)



Joint


Infer the variable combinations that demonstrate strong dependencies. 

# LLM

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/342ccd30d5bac9695148c2389459c2fbdbec32e8f978e6363dc0307b22951588.jpg)


# Inferred Dependencies ?

??: [????, ????] 

??: [????, ????????] 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5a5b4b085afe2987673fe1328882be9011ae27305261e397b0592ea206f879e5.jpg)


# External

Sources 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a69e3b38af71b01ae8437cf6eaaf1b29ccd75801772d8e86faebad50d80fac26.jpg)



Transport Mode


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/29504e445babdb9f0503668468fae825b73c8df4cbe4a9b1c49897b498029e45.jpg)



Time


# Updated Macro-Statistics

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/6bb5400da2d01ce7c9f7273d71da06ba94ef190321e431f9b153e652f4e9b5e6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/5e2ea24feddee2f8c39791707a26cc672eb2cbac857e98910075ae4e11e157d1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/cf04a275d684e2c73af4a89d3bb419c480740e0014dc8844284859b272556e96.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/a49636109ac07c8585cf00948512a346180fc8e32b48a19e0601193609b3ee6c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/000aa10d14585e373e62b7e97041f594fa134d1e09934b4050ac087c6badded4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e3c14912cf2d513dec0fca203c145c84f39c7647ea11f54800de568ff8c4c3a6.jpg)


# Modify

# Joints

# demand by time

# transport mode

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/ecb03b6d6be6b767f2b9a1ea249658a56a341d87f93ccf05b96eac1006fe4449.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/2d2897c78c6d3fe8ff9a3680f8712fe721c18865965e5724cc344d5852b1de18.jpg)


# activity

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/fa37dfe91d0f8f2e7cb7f56a6310293963e39708cd76d69b3ecf539c857266cd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/6555d89aee7f2918e8b1c52a1e868ef1f98f72096efa80eb5fd43846a9dbe015.jpg)



(b) Discrepancy-Guided Iterative Synthesis (Iteration t)



Updating Synthetic Macro-Statistics


D (t) synth 

{9h,bike,work} 

{8h, transit, work} 

{13h, bike,other} 

{17h,transit,home} 

{18h, car, other} 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/492bbd5a920231b6a673641c913b023297843c9fa60e56b7489f007cc407f835.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0012efcf497c405a7e2c6520c1b5812403397770c8ac8ff52a56cb9a4d2ad670.jpg)


Synthetic (t） synth 

Mode: 

bike $1 4 \%$ , transit $20 \%$ ,car $43 \%$ , walk 23% 

Time: 

0-6h 10%, 6-12h 20%, 12-18h 40%, 18-24h 30% 

Time X Mode 

(12-18h, bike): 12%,(18-24,car): 24%,.. 

Time X Activity 

(6-12h, work): 20%, (18-24h, home): $2 8 \% ,$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/0efbd2cd467bbbf9f86ef8055e2f0b6d5a1f77a8093f78c3f038987c916ecd9b.jpg)


Target Slarget 

Mode: 

bike $20 \%$ , transit $40 \%$ ，car $30 \%$ ,walk $10 \%$ 

Time: 

0-6h 10%, 6-12h 40%, 12-18h 20%, 18-24h 30% 

Time X Mode 

(12-18h, bike): 5%,(18-24, car): 29%,.. 

Time X Activity 

(6-12h, work): 26%, (18-24h, home): $1 6 \% ,$ 


Discrepancy Attribution


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/16f370e73131f12630d143d79c53056c9c4bcf4bde2d91d3c552df0539ff37d2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/054d84829a8f93a28bb1c71b017d844885390875bfbe8beb53981edb4c7be52e.jpg)


# Discrepancies △(t)

Mode: 

bike $+ 0 \%$ ,transit $+ 2 0 \%$ car $- 1 3 \% ,$ walk $- 1 3 \%$ 

Time: 

0-6h 0%, 6-12h +20%, 12-18h -20%, 18-24h 0% 

Time X Mode 

(12-18h, bike): -7%, (18-24, car): +5%,... 

Time X Activity 

(6-12h, work): +6%, (18-24h, home): -12%,.. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/49cf3a81aa5f842bedeb49027705487b0b641c8f2b996937f2d3ce8f521d3329.jpg)


# Generate more records that:

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/e7cfa4c00fae8bf52d5580078bb7e1238113eda0e7d93b1e95d38f7e2dbb6fe3.jpg)


(i) Marginal: 

increase trips with mode $=$ transit, time E [6-12] 

(ii) Joint: 

add trips where time E [6-12] and activity $=$ work add trips where time E [18-24] and mode $=$ car. 


Discrepancy-Guided LLM Proposal Sampling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/d4954b81317191f8b46f6f9a712301977807f6e7506a66e99393211a064c7907.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/56b61abc79edcae535042c989714c6ca585934e11eef4c37f44ff9e38bd8a1bd.jpg)


π1 

time: [6, 12] 

mode: transit 

activity: work 

n_sample: 200 

π2 

time: [18, 24] 

mode: car 

activity: home 

n_sample: 100 

π 

time: [6, 12] 

mode: transit 

activity: other 

n_sample: 50 

D (t) batch 

{7h, transit, work} 

{8h, transit, work} 

{10h, transit, work} 

{19h, car, home} 

{21h,car, home} 

{9h, transit,other} 

.. 

extend 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/edc16e0eb4bde4f7338fcdbb55ddee8d37260b5f09887dcf09e79db3c9d6c94d.jpg)


D(t) 

synth 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-23/f08dc382-7eb1-4e1e-b55f-f9ba4500ed92/99109e986c923f8b72eb9c181952f96ac9306f2ee2bb9bfa614bfab6f20a5a4d.jpg)


D(t+1) 

synth 


Figure 19: A simplified running example of discrepancy-guided iterative synthesis.
