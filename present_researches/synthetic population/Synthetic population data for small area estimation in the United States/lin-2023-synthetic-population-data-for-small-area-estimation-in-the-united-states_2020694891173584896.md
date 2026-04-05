# Synthetic population data for small area estimation in the United States

EPB: Urban Analytics and City Science 2024, Vol. 51(2) 553–562 

© The Author(s) 2023 

Article reuse guidelines: 

sagepub.com/journals-permissions 

DOI: 10.1177/23998083231215825 

journals.sagepub.com/home/epb 

S Sage 

# Yue Lin

Center for Spatial Data Science, The University of Chicago, Chicago, IL, USA 

# Abstract

Small area estimation is critical for a wide range of applications, including urban planning, funding distribution, and policy formulation. Individual-level population data, which typically include each individual’s socio-demographic characteristics and small area location, are a rich source of information for small area estimation. However, individual-level population data are often not made public due to confidentiality concerns. This paper describes the development of a public-use synthetic individual-level population dataset in the United States that can be useful for small area estimation. This dataset contains characteristics of housing type, age, sex, race, and Hispanic or Latino origin for all 308,745,538 individuals in the United States at the census block group level, based on publicly available aggregated data from the 2010 Census. Experimental results suggest the validity of the synthetic data by comparing it to different data sources, and we show examples of how this dataset can be used in small area estimation. 

# Keywords

Spatial microdata, small area estimation, open data, population data 

# Introduction

Small area estimates are statistical estimates of subpopulation characteristics in small geographic areas (e.g., counties, census tracts, census block groups) [Rao and Molina, 2015]. Various stakeholders (e.g., policymakers, planners, and analysts) are interested in these estimates as they use this information to understand local communities for purposes of policymaking, regional planning, and business development [Gonzalez and Hoza, 1978; Flowerdew and Goldstein, 1989]. To support small area estimation, statistical agencies regularly collect individual-level demographic, social, and economic data at fine geographic levels, primarily through censuses and administrative records [Brackstone, 1987]. Although these data enable direct estimation for small areas, due to the risk of revealing the identities of data subjects, the public release of such data is often restricted by privacy 

laws and regulations, such as Title 13 of the United States Code (13 U.S.C. $\ S 9$ ) and the Privacy Act of 1974 (5 U.S.C. $\ S 5 5 2 \mathrm { a } )$ . As a result, instead of releasing individual-level data, statistical agencies commonly disseminate preaggregated tables of data for small areas to the public. 

Aggregated data can be useful in many applications, but it has limitations that prevent it from fulfilling the increasing demand for small area estimates driven by many stakeholders. One major limitation is that aggregated data often only contain at most three or four subpopulation characteristics at a time [Williamson et al., 1998]. For example, aggregated data from the United States Census Summary File 1 (SF1) are cross-tabulated for at most three of the four characteristics of age, sex, race, and Hispanic or Latino origin. The utility of publicly available aggregated data is limited for small area estimation involving more characteristics or cross-tabulations of characteristics that are not included in the current aggregated data products (e.g., cross-tabulations for all four characteristics of age, sex, race, and Hispanic or Latino origin). 

To address the limitations imposed by aggregated data, various methods and software systems have been proposed to generate synthetic individual-level population data to support fine-grained decision making and micro-level analysis. One commonly utilized method for creating such synthetic population data is the iterative proportional fitting (IPF) approach. IPF employs an iterative process to adjust the weights assigned to each individual within a non-spatial individual-level survey dataset until the desired fit to the target aggregate constraints is achieved [Choupani and Mamdoohi, 2016; Lovelace et al., 2015; Simpson and Tranmer, 2005]. Several population synthesis systems have been developed based on IPF, including SYNTHESIS (Synthetic Spatial Information System) [Birkin and Clarke, 1988], PopGen [Konduri et al., 2016], and SPENSER (Synthetic Population Estimation and Scenario Projection Model) [Spooner et al., 2021]. The advancement of methods and systems for population synthesis has also led to the availability of open datasets in countries such as the United Kingdom [Lomax and Smith, 2017; Smith and Russell, 2018; Wu et al., 2022], Ireland [Farrell et al., 2012; Morrissey et al., 2015], and Canada [Predhumeau and Manley, 2023 ´ ]. 

However, the existing methods, software systems, and data have limitations. First, the IPF method typically relies on the availability of a non-spatial individual-level survey dataset for the specific study area. While such datasets are available in certain countries through sources such as the IPUMS International (Integrated Public Use Microdata Series, International) [Minnesota Population Center, 2022], they may not be accessible in many other countries, especially those in underdeveloped regions and the Global South. This reliance can limit the general applicability of IPF-based methods and systems. In addition, while open synthetic population data have gained popularity in European countries and Canada, this trend is not mirrored in the United States, which limits the potential applications that can utilize such data. 

The purpose of this paper is to describe a synthetic individual-level population dataset in the United States that is open and realistic and can be used to support small area estimation. This dataset is generated using a new method [Lin and Xiao, 2022; Lin and Xiao, 2023b] that solely relies on public aggregated data, which eliminates the need for individual-level survey data and can enhance replicability and generalizability across space and time. Specifically, we generate the synthetic data based on public census tables from the United States Census SF1. An optimization model is used to construct the synthetic data by minimizing the difference between summarized information of the synthetic population and statistics in publicly available census tables. The validity of the synthetic data is assessed by comparing it with published census tables as well as sampled national individuallevel data. 

# Methods

We aim to generate synthetic population data for all 308,745,538 individual in 220,334 block groups in the United States. Each individual has five socio-demographic characteristics of housing type, 

age, sex, race, and Hispanic or Latino origin (Table A.1). The following describes the process of synthetic data generation. 

# Materials

The United States Census Bureau collects socio-demographic characteristics and small area locations at the individual level for the entire national population. The individual-level data are confidential, but they are used to compile public-use tables of population counts by sociodemographics and small area geography. These public-use tables are included in the SF1 Dataset for the 2010 Census. We retrieve 12 2010 Census SF1 P (population) and H (housing) tables at the block group level from the National Historical Geographic Information System (NHGIS) [Manson et al., 2022], with each table including the population count by housing type, age, sex, race, and Hispanic or Latino origin, or a combination of these characteristics. Table 1 presents details of the selected census tables, and sample rows from one of the selected tables are shown in Table 2. 

# Optimization modeling

An optimization approach [Lin and Xiao, 2022; Lin and Xiao, 2023b; Lin, 2023] is used to construct the synthetic population data. We begin with a matrix representation of the individual-level population data that need to be synthesized. Let n denote the number of block groups covered by the individual-level data $( n = 2 2 0 , 3 3 4 )$ ), and $d$ the number of characteristics for each individual in the data $( d = 5 )$ . A predicate is formed to contain one value from each of the $d$ characteristics. For example, ≤Household, Under 5 years, Male, White alone, Not Hispanic or Latino≥ is a predicate. The number of all possible predicates is denoted as m $( m = 3 \times 2 3 \times 2 \times 7 \times 2 = 1$ , 932). The individual-level data can then be represented using an $m \times n$ matrix ${ \bf X } = \{ x _ { k j } \}$ , where each element $x _ { k j }$ denotes the number of individuals in block group j ( $1 \leq j \leq n )$ who can be characterized by predicate $k \left( 1 \leq k \leq m \right)$ . le t $q _ { p }$ denote the number of be represented by a ns in tabmatrix $p$ $( 1 \leq p \leq 1 2 )$ of the 12 SF1 tablehere each element $p$ $q _ { p } \times n$ $\mathbf { Y } ^ { ( p ) } = \{ y _ { i j } ^ { ( p ) } \}$ $y _ { i j } ^ { ( p ) }$ denotes the cell value of row j $( 1 \leq j \leq n )$ and column $i ( 1 \leq i \leq q _ { p } )$ f. Let $\mathbf { W } ^ { ( p ) } = \{ w _ { i k } ^ { ( p ) } \}$ be a $q _ { p } \times m$ matrix with each element $w _ { i k } ^ { ( p ) }$ ¼ f gequal one if individuals who can be characterized by predicate $k ( 1 \leq k$ 


Table 1. Census tables selected for synthetic data generation.


<table><tr><td>NHGIS code</td><td>Name</td></tr><tr><td>H7V</td><td>Total population</td></tr><tr><td>H7Z</td><td>Hispanic or Latino origin by race</td></tr><tr><td>H8I</td><td>Group quarters population by sex by age by group quarters type</td></tr><tr><td>H9A</td><td>Sex by age (white alone)</td></tr><tr><td>H9B</td><td>Sex by age (black or African American alone)</td></tr><tr><td>H9C</td><td>Sex by age (American Indian and Alaska native alone)</td></tr><tr><td>H9D</td><td>Sex by age (Asian alone)</td></tr><tr><td>H9E</td><td>Sex by age (native Hawaiian and other Pacific Islander alone)</td></tr><tr><td>H9F</td><td>Sex by age (some other race alone)</td></tr><tr><td>H9G</td><td>Sex by age (two or more races)</td></tr><tr><td>H9H</td><td>Sex by age (Hispanic or Latino)</td></tr><tr><td>H9I</td><td>Sex by age (white alone, not Hispanic or Latino)</td></tr></table>

$\leq m )$ ) can be counted in column i $( 1 \leq i \leq q _ { p } )$ of table $p$ . With each element of X being a decision variable, an optimization model can be formulated so that the squared difference between synthetic $( \mathbf { W } ^ { ( p ) } \mathbf { X } )$ and actual $( \mathbf { Y } ^ { ( p ) } )$ census tables is minimized 

$$
\min  \sum_ {p = 1} ^ {1 2} \| \mathbf {W} ^ {(p)} \mathbf {X} - \mathbf {Y} ^ {(p)} \| ^ {2}, \tag {1}
$$

$$
\text {s u b j e c t} x _ {k j} \in \mathbb {Z} ^ {*} \quad \forall k, j, \tag {2}
$$

where constraints two ensure integer decision variables. A mathematical optimization Python library called gurobipy is applied to solve the optimization problem [Gurobi Optimization, LLC, 2021]. The resulting X can be converted equivalently into a list of individuals (examples provided in the following section). 

# Data Records and Usage

The synthetic individual-level data for the United States are publicly and freely available through Figshare (https://doi.org/10.6084/m9.figshare.22056893). The data are presented in commaseparated values (CSV) format. We release one file for the entire country, as well as separate files for each state and territory. We use geographic identifiers that are consistent with those used in the United States Census Bureau data portal [United States Census Bureau, 2021] to code geographic locations at various levels. Geographic identifiers are 2-digit numerical codes at the state level, 5-digit at the county level, 11-digit at the tract level, and 12-digit at the block group level. Reference shapefiles at each geographic level are available from TIGER/Line Shapefiles [United States Census Bureau, 2010]. The values of five socio-demographic characteristics are also numerically coded in the data files to reduce file size. The data codebook is available in the same Figshare repository. Table 3 shows sample rows of the synthetic data, with each row corresponding to an individual. Figure 1 shows examples of using the synthetic data for small area estimation at the block group level. 


Table 2. Sample rows from the H7Z (Hispanic or Latino Origin by Race) table. This is only for illustrative purposes and does not present all columns.


<table><tr><td>Block group</td><td colspan="3">Not Hispanic or Latino</td><td colspan="3">Hispanic or Latino</td></tr><tr><td></td><td>White alone</td><td>Black or African American alone</td><td>...</td><td>White alone</td><td>Black or African American alone</td><td>...</td></tr><tr><td>010010201001</td><td>584</td><td>78</td><td>...</td><td>5</td><td>0</td><td>...</td></tr><tr><td>010010201002</td><td>1017</td><td>139</td><td>...</td><td>16</td><td>0</td><td>...</td></tr><tr><td>010010203001</td><td>1928</td><td>477</td><td>...</td><td>35</td><td>0</td><td>...</td></tr></table>


Table 3. Sample rows of the synthetic data.


<table><tr><td>YEAR</td><td>STATEA</td><td>COUNTYA</td><td>TRACTA</td><td>BLKGRPA</td><td>HTYPE</td><td>AGE</td><td>ETHN</td><td>RACE</td><td>SEX</td></tr><tr><td>2010</td><td>01</td><td>01001</td><td>01001020100</td><td>010010201001</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>2010</td><td>01</td><td>01003</td><td>01003010703</td><td>010030107032</td><td>1</td><td>2</td><td>1</td><td>1</td><td>2</td></tr></table>

# Technical validation

Internal validation is first performed to assess the validity of the synthetic data, in which the synthetic data are compared to the 12 SF1 tables used in data generation. Specifically, we compute the squared difference between each synthetic census table $( \mathbf { W } ^ { ( p ) } \mathbf { X } )$ and its corresponding actual table $( \mathbf { Y } ^ { ( p ) } )$ (Eq. (1)). Our experimental results show a squared difference of zero for all the 12 tables 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-09/73ce7e36-6032-46c8-a693-69861a86168b/5def4f80fc4cfba369282d5b36239badffad204523ddaa6c584e01c2a2a115f8.jpg)



Figure 1. Estimating the percentage of non-Hispanic White females aged 18 to 19 who live in households using synthetic data.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-09/73ce7e36-6032-46c8-a693-69861a86168b/bd925f3d34a790f8a5fa401f8890bdfb069d8fd5b08073277104d1a5a7159add.jpg)



Figure 2. The distribution of cosine similarity for the 2351 PUMAs.


examined. In other words, the synthetic census tables are identical to the actual tables. This indicates that synthetic data preserve statistics from actual publicly available census tables. 

We also conduct external validation to compare the synthetic data with an external data source known as the American Community Survey Public Use Microdata Sample (ACS PUMS), a five percent sample of the national individual-level data. We retrieve the 2010 5-Year ACS PUMS from the Integrated Public Use Microdata Series (IPUMS) USA [Ruggles et al., 2022]. Each individual in the ACS PUMS shares the same five socio-demographic characteristics as the synthetic data. We process the values of these characteristics to match those in the synthetic data shown in Table A.1. The ACS PUMS uses Public Use Microdata Areas (PUMAs) as the smallest geographic unit for each individual, with each PUMA consisting of a group of adjacent block groups. We aggregate the synthetic data to 2351 PUMAs to make comparisons. We represent the synthetic data for each PUMA as an $m$ -length vector $\mathbf { a } = \{ a _ { k } \}$ , with each element $a _ { k }$ denoting the number of individuals who can be characterized by predicate $k ( 1 \leq k \leq m )$ . We represent the ACS PUMS using a similar mlength vector ${ \bf { b } } = \{ b _ { k } \}$ , where each element has the same meaning but for a different data set. Cosine similarity [Dangeti, 2017] is used to compare the distributions of the synthetic data (a) and the ACS PUMS (b) 

$$
S (\mathbf {a}, \mathbf {b}) = \frac {\sum_ {k = 1} ^ {m} a _ {i} b _ {i}}{\sqrt {\sum_ {k = 1} ^ {m} a _ {i} ^ {2}} \sqrt {\sum_ {k = 1} ^ {m} b _ {i} ^ {2}}}. \tag {3}
$$

The cosine similarity ranges between 0 and 1, where a value of 1 indicates the same distribution between two datasets and 0 indicates the opposite. Figure 2 presents the distribution of cosine similarity across all 2351 PUMAs. All of the cosine similarity values are above 0.6, and the majority $( 6 3 \% )$ of them are greater than 0.95. This suggests that the synthetic data can well represent the population in the sample individual-level data. 

# Conclusions

This paper presents a synthetic population dataset that contains artificially generated values for housing type, age, sex, race, and Hispanic or Latino origin for all 308,745,538 individuals in the United States as of the 2010 Census. This dataset includes small area locations at the county, tract, and block group levels, with block groups being the finest geographic level chosen for individual privacy preservation. Compared to public aggregated data, the synthetic data offer fine-grained individual-level information that is highly desirable. 

In recent years, there has been a notable rise in the development of new and advanced methods for generating realistic synthetic population data. These methods offer improved capabilities in capturing the complexity and diversity of real-world populations. For example, Casati et al. (2015) extend the traditional IPF method by integrating advanced techniques such as Gibbs sampling and generalized raking. Farooq et al. (2013) introduce a Markov Chain Monte Carlo (MCMC) simulation method that draws from the original distribution using partial views of joint attribute distributions to synthesize population data. In addition, deep generative models, such as variational autoencoders (VAEs), have gained attention for synthesizing population data by capturing complex relationships and generating realistic populations [Borysov et al., 2019]. However, these methods can be computationally intensive and are typically more suitable for local-scale applications when computational resources are limited. In contrast, the method employed in this paper has a simpler form and can be effectively implemented at the country scale. 

Validating synthetic population data has long been a challenge. In this paper, we conduct both internal and external validation. Internally, we compare the synthetic data with the aggregated data used in its generation. In addition, we perform external validation by comparing the synthetic data 

with available ground-truth data at the individual level. However, as the actual census individuallevel responses are not publicly accessible, the data used for external validation may exhibit inherent spatial and temporal mismatches with the synthetic data, which can impact the robustness of the validation process. Fortunately, there are alternative methods outlined in the literature that have potential to address this limitation. For example, Lovelace et al. (2017) suggest considering new sources of data, such as consumer surveys, commercial data, and even social media data, to provide “sanity checks” on the results when direct external validation is not feasible. This offers potential avenues for further improving the research and enhancing the validation process. 

The openly available synthetic data are accompanied by open source code that serves as a framework for other researchers to update datasets if changes occur in the census tables coding (such as SF1). The feasibility and implications of updating the datasets depend on the nature and extent of the changes made in the census tables. To ensure adaptability, the code is designed in a modular and organized manner, allowing easy identification and modification of components related to census table integration, including clear separation of preprocessing, modeling, and converting steps. Collaboration and future references are facilitated through versioning implemented in the database, readily maintained on Figshare, which enables tracking changes and maintaining different versions. In addition, the data includes a “YEAR” column for time integration. 

The synthetic population data have potential for advancing various research and practical applications. They encompass person-level socio-demographic information collected by the census at a highly detailed geographic level, enabling stakeholders to conduct tailored analyses of sociodemographic characteristics for subpopulations in specific areas. This can facilitate precise estimation and analysis of population patterns, trends, and changes at the local level [Lomax and Smith, 2017; Wu et al., 2022]. Such data can also be used to empower policymakers and planners to simulate and evaluate the impact of various policies, interventions, or scenarios on individual behavior and movement within urban or regional contexts [He et al., 2020; Lin and Xiao, 2023a; Papyshev and Yarime, 2021; Tanton et al., 2009], thus supporting applications in domains such as public health [Grefenstette et al., 2013; Spooner et al., 2021] and transportation planning [Horl and ¨ Balac, 2021, Zhu and Ferreira, 2014]. In addition, there is existing literature on enhancing synthetic population data by incorporating census variables with external data sources such as health and commercial surveys [Spooner et al., 2021; Morrissey et al., 2015]. This integration enables realistic simulation and analysis, suggesting a potential avenue for future research to enrich the usability of our data. Further research will also explore additional applications of the synthetic dataset to broaden its potential impact across various domains. 

# Declaration of conflicting interests

The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article. 

# Funding

The author(s) received no financial support for the research, authorship, and/or publication of this article. 

# Data Availability

The fully reproducible code for data generation, usage examples, and technical validations is publicly available on GitHub at https://github.com/linyuehzzz/synthetic-populations. 

# References



Birkin M and Clarke M (1988) Synthesis—a synthetic spatial information system for urban and regional analysis: methods and examples. Environment and planning A 20(12): 1645–1671. 





Borysov SS, Rich J and Pereira FC (2019) How to generate micro-agents? A deep generative modeling approach to population synthesis. Transportation Research Part C: Emerging Technologies 106: 73–97. 





Brackstone G (1987) Small area data: policy issues and technical challenges. In: Platek R, Rao JN, Sarndal CE, ¨ et al. (eds), Small Area Statistics: An International Symposium. New York: Wiley, pp. 3–20. 





Casati D, Müller K, Fourie PJ, et al. (2015) Synthetic population generation by combining a hierarchical, simulation-based approach with reweighting by generalized raking. Transportation Research Record 2493(1): 107–116. 





Choupani AA and Mamdoohi AR (2016) Population synthesis using iterative proportional fitting (ipf): a review and future research. Transportation Research Procedia 17: 223–233. 





Dangeti P (2017) Statistics for Machine Learning. Packt Publishing Ltd. 





Farooq B, Bierlaire M, Hurtubia R, et al. (2013) Simulation based population synthesis. Transportation Research Part B: Methodological 58: 243–263. 





Farrell N, Morrissey K and O’Donoghue C (2012) Creating a spatial microsimulation model of the Irish local economy. In: Spatial microsimulation: A reference guide for users. Springer, p. 105–125. 





Flowerdew R and Goldstein W (1989) Geodemographics in practice: developments in North America. Environment and Planning A 21(5): 605–616. 





Gonzalez ME and Hoza C (1978) Small-area estimation with application to unemployment and housing estimates. Journal of the American Statistical Association 73(361): 7–15. 





Grefenstette JJ, Brown ST, Rosenfeld R, et al. (2013) Fred (a framework for reconstructing epidemic dynamics): an open-source software system for modeling infectious diseases and control strategies using census-based populations. BMC Public Health 13(1): 1–14. 





Gurobi Optimization LLC (2021). Gurobi Optimizer Reference Manual. 





He BY, Zhou J, Ma Z, et al. (2020) Evaluation of city-scale built environment policies in new york city with an emerging-mobility-accessible synthetic population. Transportation Research Part A: Policy and Practice 141: 444–467. 





Horl S and Balac M (2021) Synthetic population and travel demand for paris and ¨ ˆıle-de-France based on open and publicly available data. Transportation Research Part C: Emerging Technologies 130: 103291. 





Konduri KC, You D, Garikapati VM, et al. (2016) Enhanced synthetic population generator that accommodates control variables at multiple geographic resolutions. Transportation Research Record 2563(1): 40–50. 





Lin Y (2023) Privacy and utility of geographic data: revealing, evaluating, and mitigating the externalities of geographic privacy protection. PhD Thesis. The Ohio State University. 





Lin Y and Xiao N (2022) Developing synthetic individual-level population datasets: the case of contextualizing maps of privacy-preserving census data. AutoCarto 2022, the 24th International Research Symposium on Cartography and GIScience. 





Lin Y and Xiao N (2023a) Assessing the impact of differential privacy on population uniques in geographically aggregated data: the case of the 2020 us census. Population Research and Policy Review 42(5): 81. 





Lin Y and Xiao N (2023b) Generating Small Area Synthetic Microdata from Public Aggregated Data Using an Optimization Method. The Professional Geographer. 





Lomax N and Smith AP (2017) An introduction to microsimulation for demography. Australian Population Studies 1(1): 73–85. 





Lovelace R and Dumont M (2017) Spatial Microsimulation with R. Chapman and Hall/CRC. 





Lovelace R, Birkin M, Ballas D, et al. (2015) Evaluating the performance of iterative proportional fitting for spatial microsimulation: new tests for an established technique. The Journal of Artificial Societies and Social Simulation 18(2). 





Manson S, Schroeder J, Van Riper D, et al. (2022) IPUMS National Historical Geographic Information System: Version 17.0. Dataset. DOI: 10.18128/D050.V17.0. 





Minnesota Population Center (2022). Integrated Public Use Microdata Series, International: Version 7.3. Dataset. DOI: 10.18128/D020.V7.3. 





Morrissey K, Clarke G, Williamson P, et al. (2015) Mental illness in Ireland: simulating its geographical prevalence and the role of access to services. Environment and Planning B: Planning and Design 42(2): 338–353. 





Papyshev G and Yarime M (2021) Exploring city digital twins as policy tools: a task-based approach to generating synthetic data on urban mobility. Data & Policy 3: e16. 





Predhumeau M and Manley E (2023) A synthetic population for agent-based modelling in Canada. ´ Scientific Data 10(1): 148. 





Rao JN and Molina I (2015) Small Area Estimation. John Wiley and Sons. 





Ruggles S, Flood S, Goeken R, et al. (2022) IPUMS USA: Version 12.0. Dataset. DOI: 10.18128/D010.V12.0. 





Simpson L and Tranmer M (2005) Combining sample and census data in small area estimates: iterative proportional fitting with standard software. The Professional Geographer 57(2): 222–234. 





Smith AP and Russell T (2018) ukpopulation: unified national and subnational population estimates and projections, including variants. Journal of Open Source Software 3(28): 803. 





Spooner F, Abrams JF, Morrissey K, et al. (2021) A dynamic microsimulation model for epidemics. Social Science and Medicine 291: 114461. 





Tanton R, Vidyattama Y, McNamara J, et al. (2009) Old, single and poor: using microsimulation and microdata to analyse poverty and the impact of policy change among older australians. Economic Papers: A Journal of Applied Economics and Policy 28(2): 102–120. 





United States Census Bureau (2010). TIGER/Line Shapefiles. Dataset. https://www2.census.gov/geo/tiger/ TIGER2010/ 





United States Census Bureau (2021) Understanding geographic identifiers (GEOIDs). https://www.census. gov/programs-surveys/geography/guidance/geo-identifiers.html 





Williamson P, Birkin M and Rees PH (1998) The estimation of population microdata by using data from small area statistics and samples of anonymised records. Environment and Planning A 30(5): 785–816. 





Wu G, Heppenstall A, Meier P, et al. (2022) A synthetic population dataset for estimating small area health and socio-economic outcomes in great britain. Scientific Data 9(1): 19. 





Zhu Y and Ferreira J (2014) Synthetic population generation at disaggregated spatial scales for land use and transportation microsimulation. Transportation Research Record 2429(1): 168–177. 



Yue Lin is Assistant Instructional Professor in the Center for Spatial Data Science at the University of Chicago, Chicago, IL. Email: liny2@uchicago.edu. Her research interests include spatial data science, geocomputation, and digital privacy. 

# Appendix

# A Appendix


Table A1. Socio-demographic characteristics for each individual in the synthetic data.


<table><tr><td>Characteristic</td><td>Values</td><td>No. of values</td></tr><tr><td rowspan="2">Housing type</td><td>• Household</td><td rowspan="2">3</td></tr><tr><td>• Institutional facilities</td></tr><tr><td rowspan="24">Age</td><td>• Non-institutional facilities</td><td rowspan="24">23</td></tr><tr><td>• Under 5 years</td></tr><tr><td>• 5 to 9 years</td></tr><tr><td>• 10 to 14 years</td></tr><tr><td>• 15 to 17 years</td></tr><tr><td>• 18 to 19 years</td></tr><tr><td>• 20 years</td></tr><tr><td>• 21 years</td></tr><tr><td>• 22 to 24 years</td></tr><tr><td>• 25 to 29 years</td></tr><tr><td>• 30 to 34 years</td></tr><tr><td>• 35 to 39 years</td></tr><tr><td>• 40 to 44 years</td></tr><tr><td>• 45 to 49 years</td></tr><tr><td>• 50 to 54 years</td></tr><tr><td>• 55 to 59 years</td></tr><tr><td>• 60 to 61 years</td></tr><tr><td>• 62 to 64 years</td></tr><tr><td>• 65 to 66 years</td></tr><tr><td>• 67 to 69 years</td></tr><tr><td>• 70 to 74 years</td></tr><tr><td>• 75 to 79 years</td></tr><tr><td>• 80 to 84 years</td></tr><tr><td>• 85 years and over</td></tr><tr><td rowspan="2">Sex</td><td>• Male</td><td rowspan="2">2</td></tr><tr><td>• Female</td></tr><tr><td rowspan="7">Race</td><td>• White alone</td><td rowspan="7">7</td></tr><tr><td>• Black or African American alone</td></tr><tr><td>• American Indian and Alaska native alone</td></tr><tr><td>• Asian alone</td></tr><tr><td>• native Hawaiian and other Pacific Islander alone</td></tr><tr><td>• some other race alone</td></tr><tr><td>• two or more races</td></tr><tr><td rowspan="2">Hispanic or Latino origin</td><td>• Not Hispanic or Latino</td><td rowspan="2">2</td></tr><tr><td>• Not Hispanic or Latino</td></tr></table>