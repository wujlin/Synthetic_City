# A stochastic theory of urban metabolism

Martin Hendrick<sup>a</sup>, Andrea Rinaldo<sup>b,c,1</sup>, and Gabriele Manoli<sup>a,1</sup>

Affiliations are included on p. 8.

Contributed by Andrea Rinaldo; received January 22, 2025; accepted July 7, 2025; reviewed by Marta C. González and Diego Rybski

A current tenet in the science of cities is the emergence of power-law relations between population size and a variety of urban indicators, echoing allometric scaling in living organisms akin to Kleiber's law. However fascinating, existing scaling theories suffer from biases related to the ad-hoc definition of city boundaries and to their neglect of intraurban variability of city properties. Here, to deal rigorously with biases, we explore the hypothesis that the empirical statistics of intracity variations in population counts, road networks, and carbon emissions-across various cities and spatial scales-can be interpreted as resulting from the joint fluctuations of spatially dependent random variables. Rather than relating urban characteristics to overall city size, we focus on how intraurban properties and local population patterns vary together across space. We find that the marginal and joint probability distributions are characterized by finite-size scaling functions which, upon suitable rescaling, collapse onto a set of universal curves. These results are analogous to those relating intraspecies variability in living organisms where the scaling of mean body mass with a characteristic metabolic rate clouds the effects of the variance of both traits. Our findings lay the foundations for a generalized theory of urban metabolism, linking city-scale quantities to the covariation of intraurban characteristics. This also opens up opportunities for a full exploitation of available urban data allowing the integration of biologically inspired theories into the modeling and planning of cities.

urban scaling | population density | road networks | carbon emissions | finite-size scaling

Cities are frequently compared to living organisms with complex systems and dynamics that mirror those found in nature (1-3). Analogous to a biological circulatory system, urban transportation networks—encompassing roads, railways, and other infrastructures—serve as vital arteries, facilitating the flow of people, goods, energy, and resources. Similarly, the distribution of urban population and anthropogenic emissions (e.g.,  $\mathrm{CO}_{2}$ , pollutants) can be compared to the masses and metabolic rates of organisms, contributing to what might be considered the metabolism of cities (4-6). This analogy is supported by the emergence of seemingly universal urban scaling laws resembling allometric scaling in biological systems (e.g. refs. 7-10). Specifically, despite the variety of historical, political, and socioeconomic factors shaping the development of urban areas, evidence shows that many urban characteristics follow predictable, allometric relationships with population size  $M$  (used as a proxy for a city's mass), i.e.  $Y = Y_{0}M^{b}$ , where  $Y$  is a given urban feature (such as built area, road network length, or anthropogenic emissions),  $Y_{0}$  is a suitable constant and  $b$  is the scaling exponent (e.g. ref. 8). For example, wealth creation and innovation often exhibit superlinear scaling (i.e.,  $b > 1$ ) associated with social interactions (8), while urban infrastructure exhibits a sublinear scaling (i.e.,  $b < 1$ ).

These scaling relations are consistent with biological systems and have been linked to the distribution of resources through efficient transport networks (e.g., refs. 11 and 12). In such systems, allometric scaling is thought to arise from the general properties of tree-like networks, largely independent of specific dynamical or geometric assumptions. However, this analogy meets its limits when the distinct separability of urban systems is considered. In biological entities, clear physical boundaries defined by skin or cell membranes are crucial for the application of scaling principles, such as Kleiber's law, which relates a species' characteristic mass to its average metabolic rate (13, 14).

Urban systems often lack clearly delineated perimeters, causing their boundaries to blend into surrounding environments and complicating spatial definitions of "urban boundaries" (15, 16). This spatial ambiguity poses a significant challenge for urban scaling theories, which frequently rely on ad-hoc delineations of urban areas (17, 18). Additionally, these theories often overlook intraurban variability and the spatial organization of urban indicators that critically influence aggregate city-level patterns (e.g. ref. 19). The distinction between urban and rural areas is blurred by gradients of

# Significance

Cities can be viewed as living organisms and their metabolism as the set of processes controlling their evolving structure and function. Urban population, transport networks, and all anthropogenic activities have been proposed to mimic body mass, vascular systems, and metabolic rates of living organisms. This analogy is supported by the emergence of seemingly universal scaling laws linking city-scale quantities to population size. However, such scaling relations critically depend on the choices of city boundaries and neglect intraurban variations of urban properties. By capitalizing on today's availability of high-resolution data, findings emerge on the generality of small-scale covariations in city characteristics and their link to city-wide averages, thus opening broad avenues to understand and design future urban environments.

Author contributions: M.H., A.R., and G.M. designed research; M.H. and G.M. performed research; M.H. and G.M. analyzed data; and M.H., A.R., and G.M. wrote the paper.

Reviewers: M.C.G., University of California Berkeley; and D.R., Leibniz Institute of Ecological Urban and Regional Development.

The authors declare no competing interest.

Copyright © 2025 the Author(s). Published by PNAS. This open access article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND).

PNAS policy is to publish maps as provided by the authors.

To whom correspondence may be addressed. Email: andrea.rinaldo@epfl.ch or gabriele.manoli@epfl.ch.

This article contains supporting information online at https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.250122412/-/DCSupplemental.

Published August 11, 2025.

land use functions and characteristics (20) and the spatial distribution of urban properties is significantly influenced by factors such as historical urban planning [exemplified by Haussmann's renovation of Paris (21, 22)] and geographical/topographical features (23).

These factors pose challenges to the application of simple scaling laws to characterize cities, often necessitating the computation of "effective exponents" that depend not only on the system's size but also on external effects (24). This complexity can lead to ambiguous results, as highlighted by several studies (15, 16). Similarly, the use of different population density thresholds for the definition of urban boundaries can lead to vanishing nonlinearity or nonuniversal exponents (17). The challenge of defining city boundaries and spatial units of observation is often labeled as the Modifiable Areal Unit Problem (16), an issue that remains unresolved in the context of urban scaling (3).

To address this knowledge gap and move beyond the analysis of relationships between macroscopic quantities averaged over entire cities, we propose a stochastic framework inspired by recent advances in biological (25-27) and urban systems theories

(28, 29). This approach shifts the focus from mean values to the full probability distributions of masses and metabolic rates, treated as statistically interdependent variables. Specifically, we focus on the marginal and joint probability distributions of intracity variables (i.e., population counts, street network properties, and  $\mathrm{CO}_{2}$  emissions), and demonstrate that these distributions are characterized by universal finite-size scaling functions. By examining intracity variations and applying a scale-invariant perspective to aggregated intraurban properties at different spatial scales, we overcome the challenges posed by diffuse boundaries and describe the full range of urban characteristics observed in cities worldwide. This proves possible because of the wealth of data available for urban systems, in stark contrast with typically scarce joint biological measurements of mass and metabolic rates (25, 26).

While traditional urban scaling laws focus on aggregate city-level indicators, recent studies have emphasized the importance of spatial heterogeneity—revealed in radial profiles of land use and housing prices (30, 31) or the mesoscopic interplay between population and infrastructure (32). These findings highlight

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/6a6846f70ea781478604fa9c3b9f5d41bba699488e65d4af58307d679ca11ca8.jpg)



Fig. 1. Urban metabolism and coarse-graining of urban variables. Panels (A-C) show the street network, population counts, and residential  $\mathrm{CO}_{2}$  emissions (tonnes of carbon per year) for Portland, while panels  $(D - F)$  show the same variables for Los Angeles. Panel (G) indicates the geographical locations of the cities included in this study. Panel (H) illustrates the coarsening process applied to the data. This methodology involves studying the probability density functions (PDFs) and averages of different variables (street network properties, population counts, and  $\mathrm{CO}_{2}$  emissions) at varying spatial scales, from smaller grid sizes (box size  $I$ ) to the macroscopic city scale ( $L_{k}$ ).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/0ac5517ba6d7776e1e23f223badc99c1f0908937db57a8cec296bae2991ff948.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/70df4afeda57ce26849a01af7727a778c843ca4fd42736375136771ad5584f44.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/e4d1706d8565d619af13be5b66a7508699b78bc049f1d0cda1faace5d407a541.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/bb035a440254d651710011d6449871d1b6eb4a309b9445c4cfaacfbe163b8ecf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/a301ba0dd6b4b6a73f6c8bd57d20bfac79ef4a8bf70d4f065399564584f10454.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/09041e7902a579ea293d7beb5641ff8a6253fc0420cdbccdfbecdf50e0553fc2.jpg)



Fig. 2. Universality of urban fluctuations. (A) Probability density function (PDF) for street intersections  $N(\ell = 500\mathrm{m})$  . (B) Collapse of the PDFs of  $N$  following the finite-size scaling ansatz (Eq. 1). (C) PDF for population count  $M(\ell = 500\mathrm{m})$  . (D) Collapse of the PDFs of  $M$  . (E) PDF for residential  $\mathrm{CO}_{2}$  emissions  $(\ell = 1000\mathrm{m})$  - (F) Collapse of the PDFs of  $\mathrm{CO}_{2}$  emissions. In panels  $B,D,$  and  $F_{r}$  red lines show the PDFs of variables aggregated at city scale (e.g., for population,  $M_{k} = \sum_{i}M_{k,i},$  where  $M_{k,i}$  is the population in cell  $i$  of city  $k$  ) collapsed log-normal PDFs are also shown for comparison. Insets in these panels display the residual  $E(\alpha)$  (cumulative area enclosed between all pairs of collapsed curves (36)); red dots indicate the maxima of each curve. PDFs are computed using kernel density estimation (KDE); raw (binned) distributions are provided in the SI Appendix, Fig. S19.


the need to move beyond averages and focus on the joint distributions of urban variables. In response, we propose a unified framework that extends metabolic theories by incorporating the joint statistical behavior of population and metabolic rates, explicitly accounting for environmental stochasticity and spatial fluctuations across scales. Our results generalize existing approaches by introducing a stochastic description of intraurban variations, suggesting that cities may be better understood as complex adaptive systems shaped by distributed interactions rather than as centrally controlled machines (33).

# Results

Universal City Fluctuations. We analyzed a dataset encompassing 137 cities worldwide (Fig.  $1G$ ) to explore intraurban variability in population counts, street network properties, and  $\mathrm{CO}_{2}$  emissions (the latter for US cities only). An illustration of raw data is presented in Fig.  $1A - F$ . Each city is partitioned into square cells of linear size  $l$  (measured in meters), enabling the computation of total population,  $\mathrm{CO}_{2}$  emissions (tons of carbon per year), and number of street intersections within each cell (Fig.  $1H$ ). Utilizing kernel density estimation (34), we derived probability density functions (PDFs) for these variables within each city (see Materials and Methods for details).

Our analysis reveals distinct PDFs for population counts, residential  $\mathrm{CO}_{2}$  emissions, and street network properties (number of street intersections), with variations of several orders of magnitude across cities (Fig. 2 A, C, and E). However, after normalization, these PDFs exhibit a common scaling behavior, suggesting a universal pattern of urban structure (Fig. 2 B, D, and F). Specifically, the PDFs follow a finite-size scaling given by the ansatz (e.g. refs. 26, 29, and 35):

$$
p (V | \langle V (l) \rangle) = V ^ {- \eta} F \left(\frac {V}{\langle V (l) \rangle^ {\alpha}}\right), \tag {1}
$$

where  $F(x)$  is a finite-size cutoff function, i.e.,  $F(x) \to 0$  for  $x \to \infty$  and  $F(x)$  becomes constant for  $x \to 0$ . Here,  $\alpha = 1 / (2 - \eta)$  (due to the normalization requirement  $\int_0^\infty p(x) dx = 1$ ),  $V$  represents  $\mathrm{CO}_2$  emissions, population counts, or street intersections within cells of size  $l$  and  $F$  is a common (universal) function for all cities which depends only on  $V$ .

We also show, by minimizing the residual  $E(\alpha)$  [i.e., the cumulative area enclosed between all pairs of collapsed curves (36)], that  $\alpha \approx \eta \approx 1$  provide the best data collapses (Insets in Fig. 2 B, D, and F and SI Appendix, Figs. S20-S22). Hence, according to the ansatz (Eq. 1) the entire variability of city-specific characteristics is fully described by their average value  $\langle V(l) \rangle$  (see ref. 35 and Materials and Methods for further details).

The identification of the exact form of the PDFs is beyond the scope of this study but one may observe that the collapsed curves are described reasonably well by a log-normal distribution (Fig. 2 B, D, and F). The observed deviations from the log-normal form, especially for small variable values, could be attributed to the coarse-graining procedure and the granularity of the raw data used in this analysis (SI Appendix).

Similar results are obtained for different network properties (total street length and node degrees among others) and coarsening sizes  $l$ , as reported in SI Appendix, Figs. S1-S3.

It is also observed that the PDFs of  $V_{k} = \sum_{i} V_{k,i}$  which represents the total value of the variable  $V_{k,i}$  across all cells  $i$  in a city  $k$ , i.e., the city macroscopic behavior—follow the same finite-size scaling ansatz described in Eq. 1 (Fig. 2 B, D, and F). This result can be explained by considering the Moment Generating Function [MGF; (37)] and its reproductive property (Materials and Methods). SI Appendix, Fig. S5 further illustrates this phenomenon and its connection to the Zipf's law, which describes the rank-size distribution of cities (38).

Typically, Zipf's law suggests that the frequency of an event is inversely proportional to its rank. While the traditional Zipf's law appears to break down at the intricacy level, the PDFs of the variables we studied—regardless of the aggregation level (intraurban, country, or global scale)—collapse into a single curve. We therefore argue that the study of variables fluctuation offers a more robust framework than rank-size rules, or other allometric relations, for understanding the scaling behavior of urban phenomena across different aggregation levels.

The collapsed PDFs are characterized by a distinct bump (located at high variables' values), as shown in Fig. 2 and SI Appendix, Fig. S5. This phenomenon arises due to gaps in the discrete range of values representing city sizes within countries, which is often attributed to the Zipfian behavior of city rank.

From Fluctuations to Macroscopic Scaling Laws. Building on our analysis of intricacy properties, we now shift our focus to the average values of these properties, as they play a central role in the PDF scaling ansatz (Eq. 1). We identified power-law relationships (Fig. 3 A-C) of the form  $\langle V_k(l) \rangle = v_k l^\alpha$ , where  $V_k$  represents population  $(M)$ ,  $\mathrm{CO}_2$  emissions  $(B)$ , or the number of street intersections  $(N)$ , and  $v_k$  denotes the respective variable generalized densities  $(m_k, b_k, \text{and } n_k)$  where "generalized" indicates that densities are defined with respect to spatial support scaling that is not necessarily an integer. For simplicity, we will refer to these as "densities" in the remainder of the text.

The scaling exponents for population,  $\mathrm{CO}_{2}$  emissions, and the number of street intersections ( $\theta$ ,  $\beta$ , and  $\gamma$ , respectively) are generally less than 2, indicating that urban structures are not perfectly space-filling (see SI Appendix, Figs. S14-S18 for individual city exponents). This deviation suggests the presence of gaps within cities, which can be attributed to the heterogeneity of urban functions and land cover types (e.g., green spaces, industrial areas, and brownfield land).

The ratios of these exponents  $(\beta /\theta$  and  $\gamma /\theta)$  are approximately 1 across all cities (Fig. 3  $D$  and  $E$  and SI Appendix, Figs. S16 and S18). This suggests a proportional relationship between the number of street intersections  $/\mathrm{CO}_{2}$  emissions and population. Indeed, as mentioned in ref. 39, "people reside in buildings which are situated along streets." Thus, we can write:  $\langle B_k(l)\rangle = b_k / m_k\langle M_k(l)\rangle$  and  $\langle N_{k}(l)\rangle = n_{k} / m_{k}\langle M_{k}(l)\rangle$  . As cell size  $l$  approaches the city size  $L_{k}$ , these relationships converge to macroscopic scaling laws:  $B_{k} = b_{k} / m_{k}M_{k}$  and  $N_{k} = n_{k} / m_{k}M_{k}$  (indeed  $V_{k} = \lim_{l\to L_{k}}\langle V_{k}(l)\rangle ,V$  being  $M,B,$  or  $N$  -

To better understand the connection between intra- and intercity patterns, we compare the city size limit of the intracity relationships (characterized by the scaling exponents  $\theta$ ,  $\beta$ , and  $\gamma$ )

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/b4329afec8098784abd4e789d2a21f8f9d3134276bd6685c88937cbcd69e9ac4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/c37acb51956b6be3d2eacdff26df932f0929ae814cefbcc79e5482547d6f2078.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/74e8a4d10261fd29fc4ab33aa63f91cde9a9e5e3f76f7e56304d48cca1a68cea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/c75abae02b3a378f5b3a7a004cd2c2da54deaae4c3aea3460d1799a8e6968afd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/826ebbe33f5899b8c57ccb0f2080f93ba0899af2074c9dd6560758e044ab4a84.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/d57b0eac0c8d78747d9726c052b0bca1af7be3b3281db36f2b75877e90de6281.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/9107f2e2e05d0e9ea4f2c19489f344c424694f188f3ec4cf3481faf9f953c905.jpg)



Fig. 3. From intraurban variability to macroscopic scaling. City properties display power-law relationships with coarsening level  $l$  (provided in meter). Panels (A-C) illustrate street intersections (N), population (M), and residential  $\mathrm{CO}_{2}$  emissions (B), respectively. Panels (D and E) present street intersections and residential emissions as functions of population. The macroscopic limit (i.e., at the city level) is depicted in panels (F and G). In (F), points marked as Paris (administrative boundary; red triangle) and Greater Paris (red square; SI Appendix, Fig. S7) demonstrate how boundary selection could influence the effective exponent  $(\mathcal{F})$ . In panels (A-E), the provided exponents are the average exponents across all cities (see SI Appendix, Figs. S14-S18 for exponents for each individual city).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/e2714ca82f61480e19735b9e2b594304b36567d2e78d9b2ee637795d8b21c81b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/c0be2e802cadb5adddd740b05fe19f7261d9d52ccf4b4ab8ad24ebaf0a570d9c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/d40984791a4d483efb2ebeaf283cf27bc3359b94fd227886101a93d348213e1b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/1f63a7be41d0d81bdb1150e9020c2ae6d99d67cf7dbf75504849232930d56136.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/9c4bf155c36b5e95d0a53ec6dc2528e7d80223d5c0b4f06c6d1e7c41649f5b9d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/82509e8c-a920-4a39-9fe6-090681eebce1/91734ca0ec93a76a06e8ed4abe39513ad496635629d583db75b5b5fc2e8be2a9.jpg)



Fig. 4. Covariations in urban metabolism. (A) Joint probability density function (PDF) for population and street intersections for six cities (see S1 Appendix, Figs. S10-S16 for more cities). (B) Data collapse of the joint distribution in A according to Eq. 2. (C) Heat map showing the residual  $E(\kappa, \lambda)$  as a function of the exponents  $\kappa$  and  $\lambda$  for the collapse in B. (D) Joint PDF for population and residential  $\mathrm{CO}_{2}$  emissions for the same six cities. (E) Data collapse of the joint distribution in D. (F) Heat map of the residual  $E(\kappa, \lambda)$  for the collapse in E. The minima of the residuals indicate the best collapses, found around  $\kappa \approx \lambda \approx 1$  for both joint distributions. Residuals are computed following (36) on the full set of 137 cities. Results use a spatial resolution of  $I = 1$  km.


with the standard allometric relations proposed elsewhere for urban macroscopic properties (e.g., 8 and 1). These relations are expressed as  $B_{k} = mM_{k}^{\mathcal{E}}$  and  $N_{k} = m^{\prime}M_{k}^{\mathcal{F}}$ , where  $m$  and  $m^{\prime}$  are city-independent constants, and  $\mathcal{E}$  and  $\mathcal{F}$  are effective exponents (Fig. 3F and G). This intraurban/allometric comparison clearly points out that the effective exponents  $\mathcal{E}$  and  $\mathcal{F}$  encompass city properties (summarized by the densities  $m_{k}, b_{k},$  and  $n_k$ ) that may have no clear physical meaning (such as arbitrary city boundary definition, geographical/political constraints, and finite size effects). A simple illustrative example is the boundary choice for delineating the urban area of Paris (Fig. 3F), which affects the numerical value of  $\mathcal{F}$ .

This confirms that the study of allometric relations can lead to confusing results and obscure the physical processes shaping cities due to finite size effects and city-specific characteristics (15, 16).

Generalized Scaling of Joint Probability Distributions. After demonstrating that population, residential  $\mathrm{CO}_{2}$  emissions, and street network properties exhibit universal scaling behaviors individually, we now investigate whether similar scaling laws apply to their joint probability distributions.

Specifically, we analyzed how the combined characteristics of street intersections/CO $_2$  emissions and population counts conform to a generalized finite-size scaling ansatz of the form (25, 27):

$$
p (M, V | \langle M (l) \rangle , \langle V (l) \rangle) = \frac {1}{M ^ {\alpha}} \frac {1}{V ^ {\alpha^ {\prime}}} G \left(\frac {M}{\langle M (l) \rangle^ {\kappa}}, \frac {V}{\langle V (l) \rangle^ {\lambda}}\right), \tag {2}
$$

where  $V$  is either  $N$  or  $B$ ,  $\alpha = \alpha' = 1$  are normalization exponents (25, 35), and  $G$  is a scaling function that ideally must be independent of the specific city. Results are presented in Fig. 4 A, B, D, and E. As for the marginal distributions discussed above, the best collapses [obtained by minimizing the

residuals (36)], are given by  $\kappa \approx \lambda \approx 1$  (Fig. 4 C and F). Using the result of the previous Section, one can further write  $p(M,V|\langle M(l)\rangle ,\langle V(l)\rangle) = \frac{1}{M}\frac{1}{V} G\left(\frac{M}{\langle M(l)\rangle},\frac{V}{c_k\langle M(l)\rangle}\right)$ , where we used  $\beta /\theta \approx \gamma /\theta \approx 1$  and  $c_{k} = m_{k} / v_{k}$  with  $m_{k}$  being the population density and  $v_{k}$  being the street intersection or  $\mathrm{CO}_{2}$  emission density for city  $k$ .

This result highlights that all cities analyzed here are characterized by the same joint PDF after rescaling by their densities  $\nu_{k}$  and  $m_{k}$ . Although we do not study the underlying dynamics, this universal behavior resembles features of self-organized critical processes (40), in which open, dissipative systems with many degrees of freedom converge to statistically similar states regardless of quenched randomness, substrate properties, or other environmental, socioeconomic, and political factors shaping the context. Our approach allows us to identify such common behaviors among cities even when their macroscopic properties appear as outliers (for example, Paris; see Fig. 3F).

# Discussion

The investigation of macroscopic scaling laws in urban studies draws on our understanding of how biological systems optimize energy expenditure and resource allocation as they grow and develop (11-14, 41) and aims at portraying also cities as self-similar entities (2). However, this approach faces challenges primarily due to the ambiguity surrounding the definition of spatial boundaries (15, 16, 18), and the effects of external constraints (e.g., topography, urban planning).

Following the same rationale, our work is also inspired by biological theories of metabolism (26, 35), but overcomes the aforementioned limitations by analyzing intraurban properties as jointly varying random variables. By analyzing data at high spatial resolution from cities around the world (including North America, Asia, Africa, and Europe; see Fig. 1G), we

uncovered universal patterns in urban structure and fluctuations. Specifically, we found that marginal and joint PDFs of key urban variables, including population counts, street network properties, and  $\mathrm{CO}_{2}$  emissions, exhibit a common finite-size scaling behavior (Fig. 2).

With regard to the transport system, we focused our attention on street intersections but, given the planarity of urban networks (42), many related properties (e.g., total street length) exhibit similar statistical behaviors (SI Appendix, Figs. S1-S3).

The same cannot be said for carbon emissions. In the analysis here we have only considered the residential sector but, for the sake of completeness, one should consider the total  $\mathrm{CO}_{2}$  emissions  $(B_{k}^{\mathrm{total}})$  that encompass a diverse range of sources (including residential, but also commercial, industrial, on-road, nonroad, etc). When  $B_{k}^{\mathrm{total}}$  is considered, we no longer observe a satisfying collapse for its joint PDFs (SI Appendix, Fig. S8), reflecting the fact that total emissions depend on city-specific characteristics beyond population distribution alone. However, despite this added layer of complexity, the effective scaling relation  $B_{k}^{\mathrm{total}} = m''M_{k}^{\mathcal{E}}$  seems to hold reasonably well (SI Appendix, Fig. S9). This underscores the importance of studying intraurban distributions to gain a deeper insight into the actual relations between city size, structure, and emissions.

The generalizability of our results is further confirmed by a follow-up study where we have applied the same approach to investigate intraurban variations of urban climate variables, namely temperature and air pollution (43). The findings support the broader applicability of our framework beyond the three variables studied here but also points to nontrivial relations between urban characteristics, which lead to distinct distributional forms.

In general, our study highlighted the limitations of traditional effective macroscopic scaling laws (as exemplified by the case of Paris in Fig. 3). Effective exponents, which are typically observed or measured over a finite range of system sizes, parameters, or time, may obscure the true asymptotic exponent that describes the behavior as the system size, parameter, or time approaches infinity. By looking at intraurban properties across different spatial scale, we avoid drawing inaccurate conclusions (e.g., on whether larger cities emit more or less carbon, see discussion in refs. 15 and 44) and show that basic metabolic arguments explain the covariation of different urban properties.

To clarify how microscale urban structures aggregate to shape city-wide phenomena, we also studied the behavior of intraurban properties at their macroscopic limits, i.e., at city scale. We demonstrated (Materials and Methods) that the sum of random variables that satisfy Eq. 1 also adheres to the same distributional form (Fig. 2 and SI Appendix, Fig. S5). This finding extends traditional approaches based on Zipf law and allometric scaling (45), offering a powerful tool to describe emergent relationships among city properties. The observation that both intricacy and city level (and beyond) aggregations follow a universal distribution also highlights that, regardless of the scale, urban areas exhibit a consistent pattern in their spatial and statistical properties, reflecting common underlying dynamics in the way humans occupy space, build transport systems, and emit carbon into the atmosphere.

This result resonates with Bettencourt's (46) observation of consistent growth mechanisms across agents, groups, and city levels. In a similar vein, our work shows that PDFs of key urban variables—whether at intraurban, city, country, or global scales—all conform to the same universal form. This underscores a

profound self-similarity across scales in urban systems, suggesting that cities, regardless of spatial level, exhibit analogous statistical distributions that reflect shared underlying dynamics.

It is important to note that the statistics of city properties depend on the level (and the procedure) of spatial aggregation and cutoff choices, as shown in refs. 47 and 48. Coarsening and cutoffs modify the shape of the PDFs or, even more drastically, their nature (e.g., log-normal, normal, Pareto). More precisely, the author of ref. 47 showed that at the country scale, by considering cities as macroscopic entities, population distribution follows a log-normal form (also confirmed recently in ref. 49). The author of ref. 47 then showed that, truncation of the log-normal form leads to a Pareto distribution (and the related Zipf law) whose scaling exponent increases for an increase of the truncation point (increase of the lower cutoff for population distribution). However, our findings are more general because they are valid for a class of PDFs provided by Eq. 1 which encompasses a log-normal distribution under certain constraints (35).

Interestingly, our results and the emergence—at least as a first approximation—of a log-normal distribution, can be understood through multiplicative processes such as Gibrat's law (47, 50, 51). According to this model, population growth for city  $k$  is driven by random multiplicative factors,  $M_{k}(t + 1) = M_{k}(t)(1 + \epsilon_{k}(t))$ . Assuming independent and identically distributed growth rates  $\epsilon_{k}(t)$  and taking the logarithm, we have that  $\ln(M_{k}(t + 1)) = \ln(M_{k}(t)) + \ln(1 + \epsilon_{k}(t))$ . Hence, over many time steps, the central limit theorem implies that  $\sum_{t} \ln(1 + \epsilon_{k}(t))$  will be approximately normally distributed, leading to  $M_{k}$  being log-normally distributed. This phenomenon can be extended beyond population to other urban variables such as street network properties and residential  $\mathrm{CO}_{2}$  emissions as they coevolve with population (e.g. ref. 52).

Beyond Gibrat's law, recent studies have proposed more refined stochastic equations to describe urban population and city's rank dynamics, revealing deviations from Zipf's law (45) and highlighting the importance of migration flows between cities (50, 53, 54). In particular, in ref. 50, the authors showed that migratory shocks drive city growth as a Lévy distribution multiplicative noise. Their approach leads to a nonstationary population distribution whose exact form depends on the country, which is neither power law nor log-normal but satisfies Eq. 1 (in agreement with our results). Note, however, that to derive the city macroscopic PDFs, we have assumed here that cities are independent (Materials and Methods), which is in contradiction with the interactions (i.e., migration flows) considered in ref. 50. To reconcile these intra- and interurban dynamics, future work should focus on building a bottom-up theoretical framework linking urban heterogeneities to emergent macroscopic flows and behaviors. This is in line with the recent speculation that cities, in the future, should be described by their internal dynamics rather than their size (55).

In conclusion, our study demonstrates the power of a stochastic finite-size framework to unify the understanding of urban properties at multiple spatial scales, challenging traditional views that rely heavily on arbitrary boundary definitions and specific distributional assumptions. This paradigm shift opens broad avenues for exploring how cities function, evolve, and ultimately alter the biosphere.

By revealing a consistent distributional form across various city properties, our study provides a lens through which to view—and unify—the complexity of urban systems. It suggests that despite apparent differences in individual city structure, underlying

universal principles govern urban growth and organization. This universality will be instrumental to develop and constrain urban models (from simple growth approaches (e.g., refs. 52 and 56 to detailed land-use transport models, e.g. ref. 57) and include complexity theories in the development and use of urban digital twins (33), thus offering a pathway for urban planners and policymakers to anticipate and manage future scenarios more effectively.

Moreover, the validity of metabolic laws originally developed for biological systems reveals that cities offer a fertile playground for testing—and eventually revising—theories and hypothesis from the biological and ecological domains, where theoretical frameworks are more mature but experimental data are scarce, which hampers the possibility of analyzing joint probability distributions (25, 26). This can spark new ideas for urban sciences but also improve our understanding of other complex systems found in nature, thus stimulating a productive convergence of disciplines.

# Materials and Methods

Data Sources. Our study integrates data from three sources, each providing key information on the different aspects of urban metabolism.

First, we utilize OpenStreetMap (OSM) (58) data, accessed and processed through OSMnx(59). OpenStreetMap is an open-access collaborative project that compiles editable maps of the world, created by volunteers and largely employed in the literature to study global transport networks (e.g. refs. 60 and 61). The spatial resolution of OSM data varies, reflecting the diverse contributions of its numerous volunteers. In urban areas, the accuracy is typically high, providing detailed mappings of streets, buildings, and other infrastructures. This dataset was employed to analyze street network properties, allowing for a comprehensive investigation of urban form, including road density and connectivity, across different levels of coarsening. The use of OSMnx streamlined the entire workflow, from data acquisition to analysis, thereby enhancing the efficiency and depth of our urban infrastructure studies.

Second, we incorporate data from the Vulcan Project (62), which provides high-resolution estimates of  $\mathrm{CO}_{2}$  emissions from fossil fuel combustion in the United States. The Vulcan Project categorizes emissions into ten source sectors: residential, commercial, industrial, electricity production, onroad, nonroad, commercial marine vessel, airport, rail, and cement. This dataset offers  $\mathrm{CO}_{2}$  emissions data at a spatial resolution of  $1\mathrm{km} \times 1\mathrm{km}$  for the entire United States. This high level of detail allowed for an in-depth local and regional emissions analysis, which was critical for examining the carbon footprint associated with urban areas and exploring the relationship between urban form and greenhouse gas emissions.

Last, we use WorldPop data (63), which combines remote sensing, census, and geospatial technologies to create detailed population distribution maps. WorldPop provides population data at a high spatial resolution, down to  $100\mathrm{m} \times 100\mathrm{m}$  in some areas, enabling detailed analysis of population distribution. This dataset enabled us to analyze population counts and distribution patterns within cities across different spatial scales. SI Appendix, Table S1 summarizes the source and resolution of all data employed in this paper.

Note that the scope and selection of datasets may introduce biases, particularly when it includes a limited sample of cities or when the quality and availability of data may vary across regions and/or over time. Here, we minimize such potential biases by considering a wide range of cities worldwide and by using the latest data available. Also, by focusing on emergent statistical features rather than city-specific patterns (at specific times or locations in space), random biases across cities and variables are minimized (e.g. ref. 64).

Data Integration. Integration of the different datasets was achieved through a systematic approach. First, city linear spatial extent,  $L_{k}$ , is determined using administrative boundary. Then, each city is enclosed in a rectangle domain that

does not only include the area enclosed by a city's administrative boundaries but also extends to incorporate adjacent regions (see SI Appendix, Fig. S6 for an illustration).

Subsequently, a regular square grid, enclosed in the city's rectangular domain, was overlaid onto each city domain, creating a tessellation of square cells that partitioned the urban area into spatially uniform units. Each square cell encompassed both the area within the city boundaries and surrounding regions, ensuring comprehensive coverage of the urban environment.

Each dataset was merged and analyzed across a spectrum of coarsening levels by superimposing the regular square grid over the designated city domains. This approach provided a robust framework for dissecting and understanding the relationship between population counts,  $\mathrm{CO}_{2}$  emissions, and street network configurations.

Coarsening Procedure. To analyze urban properties at different spatial resolutions, a coarsening procedure was implemented. The regular square grid was systematically refined and coarsened to generate square cells of varying sizes. Coarsening involved merging adjacent square cells to form larger cells, thereby reducing the spatial resolution of the analysis. This stepwise coarsening process allowed the investigation of intraurban variability across multiple scales, from fine-grained local fluctuations  $(\approx 500\mathrm{m})$  to macroscopic trends (city size).

Statistical Analysis. Kernel density estimation (KDE) (34) was employed to derive PDFs for key urban variables within each square cell. PDFs were computed for population counts,  $\mathrm{CO}_{2}$  emissions, and street network properties at each coarsening level, providing insights into the spatial distribution and variability of these variables within cities. Additionally, the quality of the data collapse as a function of the choice of the scaling exponents was assessed using the residual-based method for both marginal and joint-probability PDFs as described in ref. 36.

From Intracity Variability to City Macroscopic Properties. In ref. 35, the authors extensively studied the properties of the finite-size scaling distribution given by Eq. 1, i.e.,  $p_k(V) = V^{-\eta} F(V / \langle V \rangle_k^\alpha)$  where  $k$  is used here to label cities. The normalization requirement,  $\int dx p_k(x) = 1$ , implies that  $\alpha = \frac{1}{2 - \eta}$ . Our data analysis indicated that  $\alpha = \eta = 1$  provides the best data collapse, thus confirming the theoretical result.

Examining the  $n$ -th moments of  $p_k(V)$ , i.e.  $\langle V^n\rangle_k$ , revealed key information about the properties of  $p_k$  for the random variable  $V$ . Notably, as shown in ref. 35:

$$
\langle V ^ {n} \rangle_ {k} = c _ {n} \langle V \rangle_ {k} ^ {n} \tag {3}
$$

$$
\approx c \langle V \rangle_ {k} ^ {n}, \tag {4}
$$

where  $c_{n}$  is a constant dependent on  $n$  but not on city  $k$  and  $\lim_{n\to \infty}c_n = c$  with  $c$  a constant. Thus,  $p_k(V)$  is fully determined by the average of  $V$ .

This property allowed us to explore  $p_k(V)$  further by studying its moment-generating function (MGF)  $M$  for  $V$  (37):

$$
\begin{array}{l} M _ {V} (t) := \left\langle \exp (t V) \right\rangle_ {k} (5) \\ = \sum_ {h = 0} ^ {\infty} \frac {\left\langle V ^ {h} \right\rangle_ {k} t ^ {h}}{h !} (6) \\ = \sum_ {h = 0} ^ {\infty} \frac {c _ {h} (\langle V \rangle_ {k} t) ^ {h}}{h !} (7) \\ \approx \exp (c \langle V \rangle_ {k} t), (8) \\ \end{array}
$$

where we successively used the linearity of expectation, the proportionality of the  $h$ -th moments to  $\langle V\rangle_{k}^{h}$ , and the approximation  $c_{h} = c$  valid for large  $h$ .

The function  $M_V(t)$  fully characterizes the random variable  $V$  and thus provides an alternative method for studying the properties of  $V$  (37). MGFs are particularly useful for finding moments and for proving the distribution of sums of independent random variables  $X_{k}$  (i.e., the reproductive property):

$$
M _ {X _ {1} + \dots + X _ {K}} (t) = M _ {X _ {1}} (t) \dots M _ {X _ {K}} (t). \tag {9}
$$

Moreover, if  $X, Y$  are two random variables and if  $M_X(t) = M_Y(t)$  then  $X$  and  $Y$  have the same distribution (37).

These results provide us with the tools (especially the reproductive property) to link intricacy properties with macroscopic city behavior. This allows us to move beyond standard allometric laws that focus solely on macroscopic city-average properties by fully accounting for cities' variability. At the city scale, we can consider the sum of variables  $V_{k} = \sum_{i}^{K} V_{k,i}$  for city  $k$  where  $V_{k}$  is either the total population counts, total number of street intersections or total  $\mathrm{CO}_{2}$  emissions in city  $k$ ,  $V_{k,i}$  is the variable value in the cell  $i$  and  $K$  the total number of cells covering the city  $k$ . Thus we can write:

$$
\begin{array}{l} M _ {V _ {k, 1} + \dots + V _ {k, K}} (t) = M _ {V _ {k, 1}} (t) \dots M _ {V _ {k, K}} (t) (10) \\ \approx \exp (c \left\langle V _ {k, 1} \right\rangle t) \dots \exp (c \left\langle V _ {k, K} \right\rangle t) (11) \\ = \exp (c \left\langle V _ {k} \right\rangle t). (12) \\ \end{array}
$$

This demonstrates that the PDFs of cities' variables, when considered at the city scale as entire entities, also satisfy Eq. 1.



1. H. Samaniego, M. E. Moses, Cities as organisms: Allometric scaling of urban road networks. J. Transp. Land Use 1, 21-39 (2008).





2. L. M. Bettencourt, The origins of scaling in cities. Science 340, 1438-1441 (2013).





3. F. L. Ribeiro, D. Rybski, Mathematical models to explain the origin of urban scaling laws. Phys. Rep. 1012, 1-39 (2023).





4. A. Wolman, The metabolism of cities. Sci. Am. 213, 178-193 (1965).





5. C. Kennedy, S. Pincetl, P. Bunje, The study of urban metabolism and its applications to urban planning and design. Environ. Pollut. 159, 1965-1973 (2011).





6. N. B. Grimm et al., Global change and the ecology of cities. Science 319, 756-760 (2008).





7. A. Isalgue, H. Coch, R. Serra, Scaling laws and the modern city. Phys. A Stat. Mech. Appl. 382, 643-649 (2007).





8. L. M. Bettencourt, J. Lobo, D. Helbing, C. Kuhnert, G. B. West, Growth, innovation, scaling, and the pace of life in cities. Proc. Natl. Acad. Sci. U.S.A. 104, 7301-7306 (2007).





9. L. Bettencourt, G. West, A unified theory of urban living. Nature 467, 912-913 (2010).





10. G. West, Scale: The Universal Laws of Life, Growth, and Death in Organisms, Cities, and Companies (Penguin, 2018).





11. G. B. West, The origin of universal scaling laws in biology. Phys. A Stat. Mech. Appl. 263, 104-113 (1999).





12. J. R. Banavar, A. Maritan, A. Rinaldo, Size and form in efficient transportation networks. Nature 399, 130-132 (1999).





13. M. Kleiber et al., Body size and metabolism. *Hilgardia* 6, 315-353 (1932).





14. M. Kleiber, Body size and metabolic rate. Physiol. Rev. 27, 511-541 (1947).





15. R. Louf, M. Barthelemy, Scaling: Lost in the smog. *Environ. Plan. B Plan. Des.* 41, 767-769 (2014).





16. C. Cottineau, E. Hatna, E. Arcaute, M. Batty, Diverse cities or the systematic paradox of urban scaling laws. Comput. Environ. Urban Syst. 63, 80-94 (2017).





17. E. Araute et al., Constructing cities, deconstructing scaling laws. J. R. Soc. Interface 12, 20140745 (2015).





18. L. Dong et al., Defining a city-delineating urban areas using cell-phone data. Nat. Cities 1, 117-125 (2024).





19. M. Arvidsson, N. Lovsjö, M. Keuschnigg, Urban scaling laws arise from within-city inequalities. Nat. Hum. Behav. 7, 365-374 (2023).





20. J. van Vliet et al., Bridging the rural-urban dichotomy in land use science. J. Land Use Sci. 15, 585-591 (2020).





21. B. Chapman, Baron Haussmann and the planning of Paris. *Town Plan. Rev.* 24, 177-192 (1953).





22. M. Barthelemy, P. Bordin, H. Berestycki, M. Gribaudi, Self-organization versus top-down planning in the evolution of a city. Sci. Rep. 3, 2153 (2013).





23. A. Bertaud, S. Malpezzi, "The spatial distribution of population in 35 world cities: The role of markets, planning and topography" in *The Center for Urban Land Economic Research* (University of Wisconsin Press, 1999).





24. V. Privman, Finite Size Scaling and Numerical Simulation of Statistical Systems (World Scientific, 1990).





25. E. Botte et al., Scaling of joint mass and metabolism fluctuations in in silico cell-laden spheroids. Proc. Natl. Acad. Sci. U.S.A. 118, e2025211118 (2021).





26. S. Zaoli, A. Giometto, A. Maritan, A. Rinaldo, Covariations in ecological scaling laws fostered by community dynamics. Proc. Natl. Acad. Sci. U.S.A. 114, 10672-10677 (2017).





27. S. Zaoli et al., Generalized size scaling of metabolic rates based on single-cell measurements with freshwater phytoplankton. Proc. Natl. Acad. Sci. U.S.A. 116, 17323-17329 (2019).





28. C. A. La Porta, S. Zapperi, Urban scaling functions: Emission, pollution and health. J. Urban Heal. 101, 752-763 (2024).





29. E. Strano et al., The scaling structure of the global road network. R. Soc. Open Sci. 4, 170590 (2017).





30. R. Lemoy, G. Caruso, Radial analysis and scaling of urban land use. Sci. Rep. 11, 22044 (2021).





31. R. Lemoy, G. Caruso, Evidence for the homothetic scaling of urban forms. Environ. Plan. B Urban Anal. City Sci. 47, 870-888 (2020).





32. L. Dong, Z. Huang, J. Zhang, Y. Liu, Understanding the mesoscopic scaling patterns within cities. Sci. Rep. 10, 21201 (2020).



This approach links intricacy variability with the macroscopic intercity behavior, highlighting that the sum of urban variables, while seemingly complex due to variations across cities, ultimately adheres to a universal distribution pattern.

Data, Materials, and Software Availability. Previously published data were used for this work (See refs. 58-64).

ACKNOWLEDGMENTS. G.M. acknowledges support from the Swiss National Science Foundation (SNSF) Weave/Lead Agency funding scheme (grant No. 213995). A.R. acknowledges support from SNSF (grant No. CRSII5_186422).

Author affiliations: aLaboratory of Urban and Environmental Systems, School of Architecture, Civil and Environmental Engineering, École Polytechnique Fédérale de Lausanne, Lausanne 1015, Switzerland; bLaboratory of Ecohydrology, School of Architecture, Civil and Environmental Engineering, École Polytechnique Fédérale de Lausanne, Lausanne 1015; and cEuro-Mediterranean Center on Climate Change, Lece 73100, Italy



33. G. Caldarelli et al., The role of complexity for digital twins of cities. Nat. Comput. Sci. 3, 374-381 (2023).





34. B. W. Silverman, Density Estimation for Statistics and Data Analysis (Routledge, 2018).





35. A. Giometto, F. Altermentt, F. Carrara, A. Maritan, A. Rinaldo, Scaling body size fluctuations. Proc. Natl. Acad. Sci. U.S.A. 110, 4646-4650 (2013).





36. S. M. Bhattacharjee, F. Seno, A measure of data collapse for scaling. J. Phys. A Math. Gen. 34, 6375 (2001).





37. G. Casella, R. Berger, Statistical Inference (CRC Press, 2024).





38. S. Arshad, S. Hu, B. N. Ashraf, Zipf's law and city size distribution: A survey of the literature and future research agenda. Phys. A Stat. Mech. Appl. 492, 75-92 (2018).





39. C. Molinero, S. Thurner, How the geometry of cities determines urban scaling laws. J. R. Soc. Interface 18, 20200705 (2021).





40. P. Bak, How Nature Works. The Science of Self-Organized Criticality (Springer-Verlag, Berlin, 1996).





41. J. Brown, J. Gilloly, A. Allen, V. Savage, G. West, Toward a metabolic theory of ecology. Ecology 85, 1171-1789 (2004).





42. M. Barthelemy, A. Flammini, Modeling urban street patterns. Phys. Rev. Lett. 100, 138702 (2008).





43. M. Duran-Sala, M. Hendrick, G. Manoli, Universal scaling of intra-urban climate fluctuations. arXiv [Preprint] (2025). http://arxiv.org/abs/2505.1998. (Accessed 21 September 2025).





44. R. Gudipudi, D. Rybski, M. K. Lüdeke, J. P. Kropp, Urban emission scaling-research insights and a way forward. Environ. Plan. B Urban Anal. City Sci. 46, 1678-1683 (2019).





45. M. Newman, Power laws, Pareto distributions and Zipf's law. Contemp. Phys. 46, 323-351 (2005).





46. L. M. Bettencourt, Urban growth and the emergent statistics of cities. Sci. Adv. 6, eaat8812 (2020).





47. J. Eeckhout, Gibrat's law for (all) cities. Am. Econ. Rev. 94, 1429-1451 (2004).





48. J. Eeckhout, Gibrat's law for (all) cities: Reply. Am. Econ. Rev. 99, 1676-1683 (2009).





49. Y. Wang, B. Sun, The types of city size distributions and their evolution. Cities 150, 105045 (2024).





50. V. Verbavatz, M. Barthelemy, The growth equation of cities. Nature 587, 397-401 (2020).





51. G. Toscani, Kinetic and mean field description of Gibrat's law. Phys. A Stat. Mech. Appl. 461, 802-811 (2016).





52. I. Capel-Timms, D. Levinson, B. Lahoorpoor, S. Bonetti, G. Manoli, The angiogenic growth of cities. J. R. Soc. Interface 21, 20230657 (2024).





53. S. M. Reia, P. S. C. Rao, S. V. Ukkusuri, Modeling the dynamics and spatial heterogeneity of city growth. NPJ Urban Sustain. 2, 31 (2022).





54. S. M. Reia, P. S. C. Rao, M. Barthelemy, S. V. Ukkusuri, Spatial structure of city population growth. Nat. Commun. 13, 5931 (2022).





55. M. Batty, The shape of future cities: Three speculations. Trans. Urban Data Sci. Technol. 1, 7-12 (2022).





56. H. D. Rozenfeld, D. Rybski, X. Gabaix, H. A. Makse, The area and population of cities: New insights from a different perspective on cities. Am. Econ. Rev. 101, 2205-2225 (2011).





57. F. D. Lopane, E. Kalantzi, R. Milton, M. Batty, A land-use transport-interaction framework for large scale strategic Urban modeling. Comput. Environ. Urban Syst. 104, 102007 (2023).





58. OpenStreetMap contributors, Planet dump retrieved from https://planet.asm.org, https://www.openstreetmap.org (2017). Accessed 21 July 2025.





59. G. Boeing, Osmnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Comput. Environ. Urban Syst. 65, 126-139 (2017).





60. C. Bongiorno et al., Vector-based pedestrian navigation in cities. Nat. Comput. Sci. 1, 678-685 (2021).





61. H. Haberl et al., Built structures influence patterns of energy demand and  $\mathrm{CO}_{2}$  emissions across countries. Nat. Commun. 14, 3898 (2023).





62. K. R. Gurney et al., The Vulcan version 3.0 high-resolution fossil fuel  $\mathrm{CO}_{2}$  emissions for the United States. J. Geophys. Res. Atmos. 125, e2020JD032974 (2020).





63. C.T. Lloyd, A. Sorichetta, A.J. Tatem, High resolution global gridded data for use in population studies. Sci. Data 4, 1-17 (2017).





64. G. Manoli et al., Magnitude of urban heat islands largely explained by climate and population. Nature 573, 55-60 (2019).

