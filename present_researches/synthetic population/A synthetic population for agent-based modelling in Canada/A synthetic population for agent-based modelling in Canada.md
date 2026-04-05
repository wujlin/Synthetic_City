OPE N 

Dat a Desc ript or 

# A synthetic population for agent-based modelling in Canada

Manon Prédhumeau  ✉ & Ed Manley 

In order to anticipate the impact of local public policies, a synthetic population refecting the characteristics of the local population provides a valuable test bed. While synthetic population datasets are now available for several countries, there is no open-source synthetic population for Canada. We propose an open-source synthetic population of individuals and households at a fne geographical level for Canada for the years 2021, 2023 and 2030. Based on 2016 census data and population projections, the synthetic individuals have detailed socio-demographic attributes, including age, sex, income, education level, employment status and geographic locations, and are related into households. A comparison of the 2021 synthetic population with 2021 census data over various geographical areas validates the reliability of the synthetic dataset. Users can extract populations from the dataset for specifc zones, to explore ‘what if’ scenarios on present and future populations. They can extend the dataset using local survey data to add new characteristics to individuals. Users can also run the code to generate populations for years up to 2042. 

# Background & Summary

Te trajectory of spatial and transportation modelling is undoubtedly towards more granular representations of behaviour. Facilitated by the growth in richer, fner-grained mobility data, increased use of individual-level modelling in transportation planning is widely recognised1 . Te predominant methodology in this area is Agent-Based Modelling (ABM), an approach which involves modelling heterogeneous individual agents who act and interact autonomously. ABM has been recently applied to urban planning2,3 , future transportation4–6 , policy evaluation7–9 or simulating disease outbreaks and interventions10,11. Frameworks for developing ABMs, such as MATSim12 and AIMSUN13 for transportation, and Repast and NetLogo, have support these applications. 

When applied to real-world cases, ABM can beneft from using realistic synthetic populations of agents14. A realistic synthetic population does not attempt to represent every real individual as an agent. But to qualify as realistic, the synthetic population must be composed of agents that have socio-demographic attributes that could be found in a real individual, with statistical distribution of characteristics similar to those of the real population. If the synthetic population involves relations between agents, such as household formation or a spatial dimension, the population must also have realistic statistical characteristics at these levels. Te synthetic population can then act as a test bed to evaluate the impact of public policies or to conduct experiments that would be costly, unethical, or infeasible with real population data. 

With these purposes in mind, several works proposed open-source synthetic populations; for the UK15, the US16,17, or ofen for more specifc geographic areas like for the Ile-de-France region18 (France), Tallinn19 (Estonia), American Samoa20 (US), California21 (US) or Australian capital cities22. Similarly, several works have produced synthetic populations for Canadian cities. A synthetic population has been developed for Halifax in order to simulate individuals’ decisions along their life-course23. A geospatial synthetic population has been developed for the island of Montreal in order to analyse the residential location choice of the new immigrant populations24. Te TASHA (Toronto Area Scheduling Model for Household Agents) model25, designed to study individual activity schedules and travel patterns for the Greater Toronto area, includes a synthetic population directly sampled from the Transportation Tomorrow Survey data. However, this travel survey is conducted only in the Greater Golden Horseshoe Area (south-central Ontario) and there is no equivalent at the national level. With a focus on the methodology rather than producing an open synthetic dataset, a synthetic population has been proposed for the Atlantic region for the year $\bar { 2 } 0 0 6 ^ { 2 \bar { 6 } , 2 7 }$ . Furthermore, the national statistical agency of Canada develops “Te Social Policy Simulation Database and Model (SPSD/M)”, a synthetic population dataset that is specifcally designed for analysing the tax and transfer policies at the province spatial level 

University of Leeds, School of Geography, Leeds, LS2 9JT, UK. $\boxtimes _ { \mathsf { e } }$ -mail: m.predhumeau@leeds.ac.uk 

(https://www150.statcan.gc.ca/n1/en/catalogue/89F0002X). Tey also produce Demosim, a model designed to generate population projections at fne-scale level, but the model is not available to external users. 

Although several works have developed synthetic populations for some regions of Canada, there is no up-to-date, open-source synthetic population for all of Canada. To overcome this gap, this paper details the creation of an individual-level synthetic population at a fne geographical scale for the all Canada, for the years 2023 and 2030. 

A commonly used data source for population synthesis is the population census. Canadian census data is released as aggregated statistics for various levels of geography and two Public Use Microdata Files (PUMFs), one for individuals and one for households. However because it is a complex and extensive process, population censuses are conducted only every fve years in Canada. Moreover, the census raw data need to be carefully processed by the national statistical agency to ensure confdentiality and accuracy before census results are released. Tis means that census data is published progressively between 9 months (for population counts) and 3 years (for households microdata) afer the census has taken place, and the data are therefore no longer up to date at the time of their publication. A solution adopted by the SPENSER model15 is to synthesise a base population using past UK census data and then project the population to represent the present or the future. Te method we used to generate the dataset is inspired by the SPENSER approach, but was adapted to the data available in Canada (no household projections, aggregate population data at a slightly higher level than in the UK, population projections by age and sex available only at the provincial level). 

Many methods have been proposed to generate synthetic populations28: 

Synthetic reconstruction like Iterative Proportional Fitting $\left( \mathrm { I P F } \right) ^ { 2 9 }$ and Iterative Proportional Updating $( \mathrm { I P U } ) ^ { 3 0 }$ , which combines sample data and aggregate local statistics to compute the weights refecting each sample individual’s representativeness in the local zone. Müller31 proposed a Hierarchical IPF method which sample the hierarchical PUMF to directly generate a synthetic population of households and individuals. However, this method assumes a representative sample of both households and individuals, and the hierarchical PUMF of the 2016 Canadian census contains only $1 \%$ of individuals which limits its representativeness regarding individuals. 

Combinatorial optimisation with algorithms such as hill climbing32 or simulated annealing33, which consists in duplicating real individuals from a sample and iteratively updating the synthetic population in order to better ft the real population. While the combinatorial optimisation approach has shown great potential, the optimization algorithms used can get stuck in local optima and have a high computational complexity for large populations. Most applications of combinatorial optimisation have therefore been restricted to small population sizes26 and the approach is not suitable for generating a complete Canadian population. 

Statistical learning using Markov chain Monte Carlo simulation (MCMC)34, Hidden Markov Model (HMM)35 or Bayesian network36,37, where individuals and attributes are sampled one afer another and dependent on previous states, with transitions built from partially known distributions. More recently, deep learning methods have also been proposed38, using a variational autoencoder to learn the joint distribution of all individuals in the sample. However, statistical learning methods fail to satisfy the conditional attributes distributions while satisfying the aggregated distributions of all variables simultaneously28. In these methods a post-processing step using a synthetic reconstruction method is required to accurately match the observed distributions at a small area level. 

Following the decision tree provided by Yameogo et al. 28 to identify the most suitable methods for generating a two-layered synthetic population, we decided to apply the synthetic reconstruction approach. Te most common synthetic reconstruction approach is synthetic reconstruction with $\mathrm { I P F } ^ { 2 9 }$ . Tis method uses sample data as a seed and assigns each individual in the population sample a weight such that the weighted population shows predefned marginal distributions for attributes aggregated at a small area level. IPF has the advantage to be fast, simple and deterministic39, but generates fractional weights instead of integer populations, which is an important limitation when the synthetic population is to be used in an ABM with a integer number of agents. A comparison of integerisation procedures40 showed that the ‘truncate, replicate, sample’ and ‘proportional probabilities’ methods were more accurate than the ‘simple rounding’, ‘inclusion threshold’ or ‘counter-weight’ methods. However, the integerisation process can still introduce a mismatch between the original and simulated marginal distribution40. To overcome this issue, a probabilistic resampling method called Quasirandom Integer Sampling (QIS)41 has been proposed. Tis method creates a discrete without-replacement distribution using the marginals and uses quasirandom sampling to draw the individuals. It guarantees that the randomly sampled population will exactly match the marginal data without integerisation step needed. Finally, a hybrid approach called Quasirandom Integer Sampling of IPF (QISI) combines IPF and QIS by constructing a distribution with IPF and then sampling the integral population without replacement. Tis approach provides a bridge between IPF and combinatorial optimisation, ofering a compromise between the efciency and accuracy of both techniques42. 

Similarly, the population projection to future years may be done in various ways: 

A dynamic projection23,43 consists in adding individuals through births, removing individuals from deaths, ageing the all population, and adjusting it through migrations. However, this type of approach requires extensive knowledge about transitions between each socio-demographic attribute state if we do not want the projected individuals to be “old babies” (i.e. to age individuals without evolving their other attributes). 

A static projection15 consists in using the base synthetic population as a sample and applying a reconstruction method like IPF or QISI to make the population ft the projected marginals. However this approach may be computationally expensive and thus not usable if the number of individual’s attributes and possible attributes states in the base population are important. 

<table><tr><td>Input</td><td>Format</td><td>Source</td></tr><tr><td>2016 Individual PUMF45</td><td>Microdata in Stata .dta format</td><td>Individuals File, 2016 Census of Population – Statistics Canada Catalogue no. 98M0001X (https://www150.statcan.gc.ca/n1/en/catalogue/98M0001X also available at https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/GDJRT8)</td></tr><tr><td>2016 Hierarchical PUMF46</td><td>Microdata in Stata .dta format</td><td>Hierarchical File, 2016 Census of Population – Statistics Canada Catalogue no. 98M0002X (https://www150.statcan.gc.ca/n1/en/catalogue/98M0002X also available at https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/PYYXXR)</td></tr><tr><td>2016 Census Profile by region47</td><td>Aggregate counts in .csv format</td><td>Census Profile for Canada, provinces, territories, CDs, CSDs and DAs - REGION only, 2016 Census – Statistics Canada Catalogue no. 98-401-X2016044 (https://www150.statcan.gc.ca/n1/en/catalogue/98-316-X2016001)</td></tr><tr><td>2016 Geographic Attribute File48</td><td>Geographic hierarchy in .csv format</td><td>Geographic Attribute File, 2016 Census – Statistics Canada Catalogue no. 92-151-2016001 (https://www150.statcan.gc.ca/n1/en/catalogue/92-151-X2016001)</td></tr><tr><td>2018 Population projections49</td><td>Projected population values in .csv format</td><td>Projected population, by projection scenario, age and sex, as of July 1 (x 1,000) – Statistics Canada Table 17-10-0057-01 (https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701)</td></tr></table>


Table 1. Input data and sources.


<table><tr><td>Id.</td><td>Weight</td><td>Age group</td><td>Highest degree</td><td>Household size</td><td>Labour force status</td><td>Province</td><td>Household primary maintainer</td><td>Sex</td><td>Total income</td></tr><tr><td>871649</td><td>37.037277</td><td>40 to 44</td><td>Program &gt; 2 years</td><td>4 persons</td><td>Employed - Worked in reference week</td><td>35</td><td>No</td><td>Female</td><td>6,000$</td></tr><tr><td>591795</td><td>37.037277</td><td>20 to 24</td><td>Secondary school diploma or equivalent</td><td>4 persons</td><td>Employed - Worked in reference week</td><td>35</td><td>No</td><td>Male</td><td>24,000$</td></tr><tr><td>838385</td><td>37.037277</td><td>40 to 44</td><td>Bachelor&#x27;s degree</td><td>4 persons</td><td>Not in the labour force - Last worked before 2015</td><td>35</td><td>No</td><td>Female</td><td>2,000$</td></tr></table>


Table 2. Extract from the 2016 Individual PUMF records.


<table><tr><td>Household Id.</td><td>Id.</td><td>Weight</td><td>Age group</td><td>Province</td><td>Household primary maintainer</td><td>Sex</td></tr><tr><td>6</td><td>61102</td><td>100.196885</td><td>0 to 9</td><td>24</td><td>No</td><td>Male</td></tr><tr><td>6</td><td>61103</td><td>100.196885</td><td>0 to 9</td><td>24</td><td>No</td><td>Male</td></tr><tr><td>7</td><td>71101</td><td>100.384035</td><td>20 to 24</td><td>35</td><td>Yes</td><td>Male</td></tr></table>


Table 3. Extract from the 2016 Hierarchical PUMF records.


Finally, resampling is a simple and efcient projection approach. Tis consists in using the base population and randomly duplicating or removing individuals from the population in order to ft the projected marginals. Tis method presents the advantages to be fast, to not be data-intensive and to keep the individual attributes consistent. Tis is a method that we developed afer noticing that 1) there was not enough information on the transitions between attributes states to apply a dynamic projection and 2) methods like QISI were not suitable if individuals had their small area of residence as an attribute, because this attribute has between 50 and 20160 possible values depending on the Province, which makes the application of the QISI method extremely slow. A limitation of the resampling method is that it assumes that the individual sets of attributes remain the same over time, which makes it more suitable for short- and medium-term forecasts, where changes in individual correlated attributes (salary by age, qualifcation by age, etc) are small, than for long-term forecasts. Te longer-term the predictions are the more uncertainty they involve. Forecasting to 2042 may thus involve a risk if done for certain scenarios (e.g. forecasting a non-marginal change in the population structure). 

Tis paper presents the construction and validation of a synthetic population for Canada. First, the QISI approach was used to generate a base synthetic population from the 2016 Canada census data. We used 2016 census PUMF data that are realistic at the individual level and 2016 census aggregated data that allow a geographically realistic distribution of the individuals. Ten, a resampling method was used to project the base population of 2016 to present (2023) and future (2030) years based on provincial population projections. Two designed algorithms were then used to assign individuals to households and to infer household types. In addition to 2023 and 2030, a population was synthesised for 2021 and compared to population data from the 2021 census, in order to validate the approach and dataset. Comparison results are presented at the national, city and dissemination area levels, to support the technical quality of the dataset. 

Te 2023 and 2030 synthetic populations have been developed for the RAIM (Responsible Automation for Inclusive Mobility) project. Te RAIM project is a British-Canadian collaboration to address how an on-demand autonomous vehicle system can meet the diverse needs of older populations and improve the lives of older travellers. Te RAIM research is applied in two regions: the city of Winnipeg (Manitoba, Canada) and the West Midlands (UK), through partnerships with local transport providers. As part of the project, an agent-based model will be developed and simulations will be conducted to predict how demand for an on-demand autonomous vehicle service varies given spatial, temporal, and population-level variation. Such simulations require individual-level population estimates to be built for the study regions at fne spatial scale. Data produced in this paper will be used as an input for the agent-based model to identify the need for autonomous on-demand transportation in the 

<table><tr><td>Characteristics</td><td>Total</td><td>Male</td><td>Female</td></tr><tr><td>Population, 2016</td><td>1,278</td><td></td><td></td></tr><tr><td>Private dwellings occupied by usual residents</td><td>440</td><td></td><td></td></tr><tr><td>Total - Age groups and average age of the population - 100% data</td><td>1280</td><td>560</td><td>715</td></tr><tr><td>0 to 4 years</td><td>90</td><td>40</td><td>45</td></tr><tr><td>5 to 9 years</td><td>90</td><td>45</td><td>50</td></tr><tr><td>10 to 14 years</td><td>100</td><td>45</td><td>50</td></tr><tr><td>15 to 19 years</td><td>85</td><td>40</td><td>45</td></tr><tr><td>20 to 24 years</td><td>95</td><td>50</td><td>50</td></tr><tr><td>25 to 29 years</td><td>75</td><td>25</td><td>50</td></tr><tr><td>30 to 34 years</td><td>95</td><td>45</td><td>50</td></tr><tr><td>35 to 39 years</td><td>90</td><td>40</td><td>50</td></tr><tr><td>40 to 44 years</td><td>105</td><td>45</td><td>60</td></tr><tr><td>45 to 49 years</td><td>105</td><td>40</td><td>60</td></tr><tr><td>50 to 54 years</td><td>100</td><td>45</td><td>55</td></tr><tr><td>55 to 59 years</td><td>60</td><td>25</td><td>35</td></tr><tr><td>60 to 64 years</td><td>50</td><td>15</td><td>35</td></tr><tr><td>65 to 69 years</td><td>50</td><td>20</td><td>30</td></tr><tr><td>70 to 74 years</td><td>35</td><td>15</td><td>25</td></tr><tr><td>75 to 79 years</td><td>25</td><td>10</td><td>10</td></tr><tr><td>80 to 84 years</td><td>10</td><td>5</td><td>10</td></tr><tr><td>85 years and over</td><td>5</td><td>0</td><td>5</td></tr><tr><td>Total - Private households by household size - 100% data</td><td>440</td><td></td><td></td></tr><tr><td>1 person</td><td>85</td><td></td><td></td></tr><tr><td>2 persons</td><td>120</td><td></td><td></td></tr><tr><td>3 persons</td><td>85</td><td></td><td></td></tr><tr><td>4 persons</td><td>85</td><td></td><td></td></tr><tr><td>5 or more persons</td><td>70</td><td></td><td></td></tr><tr><td>Total - Total income groups in 2015 for the population aged 15 years and over in private households - 100% data</td><td>1,000</td><td>430</td><td>565</td></tr><tr><td>Under $10,000 (including loss)</td><td>140</td><td>65</td><td>75</td></tr><tr><td>$10,000 to $19,999</td><td>190</td><td>65</td><td>125</td></tr><tr><td>$20,000 to $29,999</td><td>100</td><td>30</td><td>70</td></tr><tr><td>$30,000 to $39,999</td><td>95</td><td>35</td><td>60</td></tr><tr><td>$40,000 to $49,999</td><td>90</td><td>30</td><td>65</td></tr><tr><td>$50,000 to $59,999</td><td>70</td><td>40</td><td>35</td></tr><tr><td>$60,000 to $69,999</td><td>60</td><td>25</td><td>35</td></tr><tr><td>$70,000 to $79,999</td><td>45</td><td>15</td><td>25</td></tr><tr><td>$80,000 to $89,999</td><td>45</td><td>25</td><td>20</td></tr><tr><td>$90,000 to $99,999</td><td>30</td><td>20</td><td>15</td></tr><tr><td>$100,000 and over</td><td>90</td><td>55</td><td>30</td></tr><tr><td>Total - Highest certificate, diploma or degree for the population aged 15 years and over in private households - 25% sample data</td><td>1,015</td><td>440</td><td>580</td></tr><tr><td>No certificate, diploma or degree</td><td>170</td><td>90</td><td>80</td></tr><tr><td>Secondary (high) school diploma or equivalency certificate</td><td>305</td><td>110</td><td>200</td></tr><tr><td>Postsecondary certificate, diploma or degree</td><td>540</td><td>240</td><td>300</td></tr><tr><td>Total - Population aged 15 years and over by Labour force status - 25% sample data</td><td>1,020</td><td>440</td><td>575</td></tr><tr><td>In the labour force</td><td>745</td><td>345</td><td>400</td></tr><tr><td>Employed</td><td>690</td><td>320</td><td>370</td></tr><tr><td>Unemployed</td><td>55</td><td>25</td><td>35</td></tr><tr><td>Not in the labour force</td><td>270</td><td>100</td><td>170</td></tr></table>


Table 4. Extract from the 2016 Census Profle for a dissemination area.


city of Winnipeg. Te 2023 and 2030 Winnipeg synthetic populations have be complemented with additional attributes (driving licence, health status) from local surveys, and will be extended with individual’s daily activity patterns to produce an activity-based model and derive the older population travel demand. 

The synthetic population has been generated using only publicly available data and open-source code to ease replicability. The synthetic populations are provided as csv files for 2016 (base population), 2021 (validation population), 2023 (present population) and 2030 (future population). Synthetic populations for 2021, 2023 and 2030 are provided for 9 population growth scenarios. Moreover, the code used to generate the synthetic populations is also available together with the code that was employed for the validation and scripts to parallelize the code execution on a server. 

Users can extract populations from the dataset for specifc zones of interest (province, city, neighborhood) or for specifc sub-populations to gain insight into relationships at a given spatial scale or for a given group. Te synthetic 

<table><tr><td>DBuid</td><td>DAuid</td><td>PRuid</td><td>PRename</td><td>CSDuid</td><td>CSDname</td></tr><tr><td>10020117004</td><td>10020117</td><td>10</td><td>Newfoundland and Labrador</td><td>1002006</td><td>Division No. 2, Subd. F</td></tr><tr><td>10020117003</td><td>10020117</td><td>10</td><td>Newfoundland and Labrador</td><td>1002006</td><td>Division No. 2, Subd. F</td></tr><tr><td>47090190048</td><td>47090190</td><td>47</td><td>Saskatchewan</td><td>4709006</td><td>Wallace No. 243</td></tr></table>


Table 5. Extract from the 2016 Geographic Attribute File.


<table><tr><td>Ref_date</td><td>Geo</td><td>DGUID</td><td>Projection scenario</td><td>Sex</td><td>Age group</td><td>Value</td></tr><tr><td>2023</td><td>Manitoba</td><td>2016A000246</td><td>Projection scenario LG: low-growth</td><td>Females</td><td>0 to 4 years</td><td>40900</td></tr><tr><td>2023</td><td>Manitoba</td><td>2016A000246</td><td>Projection scenario LG: low-growth</td><td>Females</td><td>5 to 9 years</td><td>43900</td></tr><tr><td>2023</td><td>Manitoba</td><td>2016A000246</td><td>Projection scenario LG: low-growth</td><td>Females</td><td>95 to 99 years</td><td>2200</td></tr><tr><td>2023</td><td>Manitoba</td><td>2016A000246</td><td>Projection scenario LG: low-growth</td><td>Females</td><td>100 years and over</td><td>500</td></tr></table>


Table 6. Extract from the 2018 Population projections fle.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/97d4dee69e500f1362a43af88e7ab51bbfd9af3e69bd8c3f30330126d61f2a1c.jpg)



Fig. 1 4-step workfow for generating the synthetic population. Each of the 4 scripts (in blue) takes as input (in orange) fles from the 2016 census, from the population projections and an output from the previous script, as well as some parameters (in grey).


population can be used as an input into agent-based models to investigate the potential impact of local public policies on present and future populations. Te synthetic population can also be used to initialize an agent-based social simulation and study emergent phenomena that may result from local interactions. Users can enrich and extend the synthetic population dataset by linking it to other datasets. Tey can use their own data or public data, such as local surveys data, to add new characteristics to synthetic individuals. Users can link the synthetic population to OpenStreetMap data to add residential buildings to the households for example. If users are interested in a year other than 2023 and 2030, they can use the proposed scripts to project the 2016 synthetic population for years up to 2042 (latest date for which population projections are available for the provinces and territories). Once the 2021 census data is fully released, it will be possible to simply replace the input fles from 2016 census and generate a 2021 base population which might be projected in the following years to obtain more accurate future populations. 

# Methods

Zoning system. Te synthetic population generation uses the multi-level spatial zoning system defned by Statistics Canada44. On the top level, the study area comprises the whole Canada, which is divided in 10 provinces and 3 territories. Each province or territory is divided into census subdivisions (CSD), which is the general term for municipalities or areas treated as municipal equivalents for statistical purposes. All CSD are further divided into dissemination areas (DA), small geographic units each with an average population of 400 to 700 persons based on data from the previous census. Each DA is further divided into dissemination blocks (DB), but only census population and dwelling count data are available at this scale. DA are the smallest standard geographic areas for which all census data is disseminated. Te synthetic individuals are produced for the whole Canada and are localised at the DA scale. 

Inputs. Two publicly available data sources, outlined in Table 1 are used as input: 2016 census data and 2018 population projections. Tables 2–6 show example extracts of the input fles. 

2016 Census data. Te 2016 census data were released in various ways. For this work, we used 4 outputs from the 2016 census: 

Te Individual PUMF45. Tis microdata fle provides access to non-aggregated data on the characteristics of the individuals in the Canadian population. Te fle contains a $2 . 7 \%$ sample of the Canadian population and provides access to 930,421 anonymised individual records from the 2016 Census questionnaire. Each individual in this sample presents 123 variables, a unique identifer and an individual weighting factor. Individuals in the PUMF are localised at the provinces (and a group gathering the three territories) level to preserve confdentiality. 

Te Hierarchical PUMF46. Similarly to the individual PUMF, this fle provides access to non-aggregated data for a sample of $1 \%$ of the Canadian households. Te fle contains 343,330 individuals records related to 140,705 households, and thus enables the study of individuals in relation to their households. Each individual record is restricted to the provinces level and consists in 95 variables, a unique identifer, a household identifer and an individual weighting factor. 

Te Census Profle47. Tis fle contains aggregate population counts for various variables (age, sex, education, households, income, etc) and for various levels of geography, including provinces and territories, CSD and DA. We used the census profle with counts disseminated at the DA level. We used as input the census profle split into six fles region by region, in order to avoid loading a 5 Gb fle at once. 

Te Geographic Attribute File48. Te fle contains information at the DB level, based on 2016 Census standard geographic areas with correspondences from DB to higher levels. Te fle is thus useful for obtaining the complete geographic hierarchy of areas with the codes and names used for each level of the geographic hierarchy. For example, the codes for all DAs belonging to a CSD or a province can be obtained from this fle. 

It should be noted that the PUMF fles do not include people living in institutions or collective dwellings such as hospitals, nursing homes, penitentiaries or student residences. Tese people are estimated to represent $1 . 9 \%$ of the Canadian population according to 2016 Census, more than half of them living in nursing homes or residences for senior citizens. People living in collective dwellings are counted in the synthetic population but are assigned into private households and have attributes from the PUMF, i.e. attributes from people not living in collective dwellings. If the dataset is used to study people living in collective dwellings, it might therefore be necessary to adapt the synthetic population, especially when generating the households. 

Moreover, to protect the confdentiality of individuals, areas with a population of less than 40 persons are not present in the census profle data and census profle counts are randomly rounded either up or down to a multiple of $^ { \mathfrak { c } _ { \mathfrak { z } } }$ or ‘10.’ 

2018 Population projections. Te second data source is population projections for provinces and territories49. Te national statistical agency of Canada develops population projections by age and sex every 5 years for provinces and territories, based on various assumptions on the population growth. Te last projections were developed in 2018, for 2018 to 2043. Te population projections gives a perspective of the future Canadian population demography according to nine scenarios. Each scenario is built on assumptions about the main components of population growth (fertility, life expectancy at birth, interprovincial migration, immigration and emigration). Five medium-growth scenarios (M1, M2, M3, M4 and M5) refect diferent internal migration patterns observed in the past, low-growth (LG) and high-growth (HG) scenarios explore either lower or higher population growth than in the medium-growth scenarios, and fast-aging (FA) and slow-aging (SA) scenarios consider either faster or slower population aging than in the medium-growth scenarios. 

We generated a synthetic population for each projection scenario to ensure that the model can be applied to all possible use cases. For the dataset validation we used the LG scenario, which is based on the following assumptions: the fertility rate reaches 1.4 children per woman in 2042/2043; life expectancy at birth reaches 82.6 years for males and 86.6 years for females in 2042/2043; interprovincial migration is based on linear interpolation of recently observed migration rates to rates observed over a long period of time reached in 2030/2031, and rates that remain constant thereafer; the immigration rate reaches $0 . 6 5 \%$ in 2042/2043; the annual number of non-permanent residents reaches 1,259,300 in 2043; the net emigration rate reaches $0 . 1 7 \%$ in 2042/2043. 

All the input data sources used to generate the synthetic population are publicly accessible through Statistics Canada Catalogue and can be downloaded from the sources listed in Table 1. PUMF are published under the Statistics Canada Open Licence since October 2018. Tey can be ordered for free from Statistics Canada Catalogue45,46 or can be downloaded from Abacus50,51, a repository of open data hosted by UBC Library. Te input .csv fle for the population projections can be downloaded through the Statistics Canada Catalogue by selecting “Download options” and then “CSV - Download entire table “Projected population, by projection scenario, age and sex, as of July 1”. 

Workfow. Te overall workfow for generating the synthetic populations in this study is detailed in Fig. 1. Te population synthesis is composed of four sequential steps: (1) generation of a base synthetic population of individuals for 2016, (2) projection of the base synthetic population towards future years 2021, 2023 and 2030, (3) assignment of individuals into households and (4) assignment of households types. On Fig. 1, scripts for each step are in blue and in orange is shown external data sources and input/output data for each script. On the right of each script, script parameters and one example of parameters are given. Each workfow step is described as follows. 

Base synthetic population generation. Te frst step involves synthesising a population province by province for the base year 2016, at the DA level. Te QISI approach, which combines IPF and QIS is used to synthesise an integral population DA by DA. Population synthesis for one province is performed as described in Algorithm 1. 


Algorithm 1: Population synthesis algorithm


input :indivs_pumf: 2016 Individual PUMF  
input :census_file: 2016 Census Profile file for the region  
input :geo_att_file: 2016 Geographic Attribute File  
input :pr: Province code  
output :synth_pop_csv: 2016 Synthetic population file  
begin  
DA_codes ← load(geo_att_file, pr)  
individuals ← load(indivs_pumf, pr)  
census_pr ← load(census_file, pr)  
synth_pop ← []  
seed ← initialise(seed(individuals)  
for DA_code in DA_codes do  
census_ta ← load(census_pr, DA_code)  
if census_ta/population > 0 then  
margins ← initialise_margins(census_ta, census_pr)  
margins ← match_margins(margins, census_pr)  
synth_pop_ta ← qisi(seed, marginals)  
synth_pop_ta.area ← DA_code  
synth_pop += synth_pop_ta  
end  
end  
Write synth_pop to synth_pop_csv 

Seed initialisation. Te weighted individuals localised in the province from the 2016 Individual PUMF are used to initialise the seed. Because convergence problems can occur when one of the rows is zero and the marginal total is nonzero, we allowed the zero state in the seed to be occupied with a small probability. Te individuals’ variables in the seed are: age group, sex, highest degree, labour force status, household size, total income and household responsibility. 

Marginals initialisation. Te aggregate counts by DA for each variable are loaded from the 2016 Census Profle and are used as marginals (i.e. target totals) in the IPF procedure. Sometimes the subtotal for a variable is not available at the DA level. Ten the distribution of the variable at the province level is used to infer the DA subtotal. 

Te marginals loaded for each DA are: total population, total number of households, total population by sex, total population by age group, total population by age group and sex, total population by household size, total population by highest degree, total population by labour force status, total population by income group. 

Te Individual PUMF variables’ categories and the Census Profle variables’ categories do not always match; e.g. categories for age group in PUMF comprise $^ { \infty } 5$ to 6 years” and $^ { \infty } 7$ to 9 years” while Census Profle report counts for $^ { \mathfrak { a } } 5$ to 9 years”. We then used unifed variables categories. Te correspondence between categories used in the Individual PUMF, in the Census Profle, and in the synthetic population is detailed in Tables 7–13. 

Marginals matching. Te subtotals sum for each variable must be equal to the DA total population count in order to apply IPF. However, categories of some of the variables in the Census Profle report counts only for the population aged 15 years and over. In order to match the total population count, we added the count of population for the age group 0–14 years to the category ${ } ^ { \mathfrak { e } } \mathrm { N o }$ certifcate, diploma or degree” for the “Highest degree” variable, to the category “Not in labour force” for the “Labour force status” variable, and to the category $^ { \alpha } < \$ 20,000$ for the “Total income” variable. 

Moreover, due to missing data and randomly rounded variables to preserve confdentiality, variable totals do not always match the DA total population count. Total population counts by sex, by age group, by age group and sex, by household size, by highest degree, by labour force status, and by income group have therefore been adjusted to match the total population count. Te marginals matching process is done for each variable by iteratively increasing or decreasing the variable marginals following the province marginals distribution, until the variable marginals sum match to DA total population 

Quasirandom integer sampling of IPF (QISI). Te QISI algorithm frst constructs a probability distribution for individuals, constrained to the marginal sums in every dimension, using IPF. QISI then samples the integral population using Quasirandom Integer Sampling without replacement. We used the implementation from the humanleague package42, developed for micro-synthesising populations from marginal and seed data. 

Population projection. Population projections published by Canada’s national statistical agency are available by age and sex for each province or territory, for each year from 2018 to 2042, and for 9 population growth scenarios. We have projected the 2016 base synthetic population for the future years 2021, 2023 and 2030, province by province, according to each scenario. 

For each scenario, each province, and each projection year, we calculated the diference in population by age group and sex between 2016 and the projection year. Ten, for each age group and sex, we applied a resampling, by randomly duplicating or deleting individuals from the 2016 population in that age group and sex group to match the population of the projection year. Algorithm 2 details this approach. 


Algorithm 2: Population projection algorithm


input :synth_pop_csv: 2016 Synthetic population file  
input :projections_pop_file: 2018 Population projections file  
input :geo_att_file: 2016 Geographic Attribute File  
input :pr: Province code  
input :year: Projection year  
input :scenario: Projection scenario  
output:synth_popProj_csv: Projected synthetic population file  
begin  
synth_pop $\leftarrow$ load(synth_pop_csv,geo_att_file,pr)  
if year $>$ 2017 and year $<  2043$ then  
projections_pr $\leftarrow$ load(projections_pop_file,pr,year,scenario)  
age_grps $\leftarrow$ get_age_grps(synth_pop)  
for age in age_grps do  
for sex in [0,1] do  
projection[age,sex] $\leftarrow$ get_projections_by_age-sex(projections_pr,age,sex)  
diff[age,sex] $\leftarrow$ projection[age,sex]-count(synth_pop[age,sex])  
if diff[age,sex] $>0$ then  
if diff[age,sex] $>$ count(synth_pop[age,sex]) then  
| toDuplicate $\leftarrow$ synth_pop[age,sex].sample_with_replacement(diff[age,sex])  
else  
| toDuplicate $\leftarrow$ synth_pop[age,sex].sample(diff[age,sex])  
end  
synth_pop $+=$ toDuplicate  
else  
to_delete $\leftarrow$ synth_pop[age,sex].sample(diff[age,sex])  
synth_pop $=$ to_delete  
end  
end  
Write synth_popProj to synth_popProj_csv 

<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="22">Age group</td><td>1</td><td>0-4</td><td>0</td><td>0-4</td><td>0 to 4 years: Total</td></tr><tr><td>2</td><td>5-6</td><td rowspan="2">1</td><td rowspan="2">5-9</td><td rowspan="2">5 to 9 years: Total</td></tr><tr><td>3</td><td>7-9</td></tr><tr><td>4</td><td>10-11</td><td rowspan="2">2</td><td rowspan="2">10-14</td><td rowspan="2">10 to 14 years: Total</td></tr><tr><td>5</td><td>12-14</td></tr><tr><td>6</td><td>15-17</td><td rowspan="2">3</td><td rowspan="2">15-19</td><td rowspan="2">15 to 19 years: Total</td></tr><tr><td>7</td><td>18-19</td></tr><tr><td>8</td><td>20-24</td><td>4</td><td>20-24</td><td>20 to 24 years: Total</td></tr><tr><td>9</td><td>25-29</td><td>5</td><td>25-29</td><td>25 to 29 years: Total</td></tr><tr><td>10</td><td>30-34</td><td>6</td><td>30-34</td><td>30 to 34 years: Total</td></tr><tr><td>11</td><td>35-39</td><td>7</td><td>35-39</td><td>35 to 39 years: Total</td></tr><tr><td>12</td><td>40-44</td><td>8</td><td>40-44</td><td>40 to 44 years: Total</td></tr><tr><td>13</td><td>45-49</td><td>9</td><td>45-49</td><td>45 to 49 years: Total</td></tr><tr><td>14</td><td>50-54</td><td>10</td><td>50-54</td><td>50 to 54 years: Total</td></tr><tr><td>15</td><td>55-59</td><td>11</td><td>55-59</td><td>55 to 59 years: Total</td></tr><tr><td>16</td><td>60-64</td><td>12</td><td>60-64</td><td>60 to 64 years: Total</td></tr><tr><td>17</td><td>65-69</td><td>13</td><td>65-69</td><td>65 to 69 years: Total</td></tr><tr><td>18</td><td>70-74</td><td>14</td><td>70-74</td><td>70 to 74 years: Total</td></tr><tr><td>19</td><td>75-79</td><td>15</td><td>75-79</td><td>75 to 79 years: Total</td></tr><tr><td>20</td><td>80-84</td><td>16</td><td>80-84</td><td>80 to 84 years: Total</td></tr><tr><td>21</td><td>&gt;=85</td><td>17</td><td>&gt;=85</td><td>85 years and over: Total</td></tr><tr><td>88</td><td>Not available</td><td>ignored</td><td></td><td></td></tr></table>


Table 7. Correspondence of the categories for the “age group” attribute.


<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="2">Sex</td><td>1</td><td>Female</td><td>0</td><td>Female</td><td>Population, 2016: Female</td></tr><tr><td>2</td><td>Male</td><td>1</td><td>Male</td><td>Population, 2016: Male</td></tr></table>


Table 8. Correspondence of the categories for the “sex” attribute.


Household assignment. Te third step consists in assigning the synthetic individuals into households. Tis step is performed for each scenario, each year of projection and each province or territory, according to Algorithm 3. At this step, an age attribute is added to each synthetic individual when the synthetic population is loaded. Te age attribute is randomly drawn in the age group range of the individual. For the individuals aged 0 to 84, a uniform distribution over the age group range is used. For the individuals aged 85 and over, a geometric distribution over the age group range with a success probability $\mathtt { p } = 0 . 2$ is used, to refect the population rapid decline in this age group. 

Households initialisation. For each DA, we know the number of households that need to be assigned by the number of synthetic individuals who are identifed as primary household maintainer. For each DA, we then create one household by individual identifed as primary household maintainer. 

Households size determination. Ten, for each household, we get the household size from the primary maintainer attributes in order to know how many members need to be assigned to this household. If the household is one person, then the household only contains the primary maintainer and is complete. If the household is more than one person, then it needs to be completed with non-responsible individuals. 

Households completion. Each household is completed with non-responsible individuals. Te non-responsible individuals are grouped by household size attribute, so that they are assigned to a household with a corresponding size. Te non-responsible individuals are classifed by age group either as young (age ${ < } 1 9$ years) or as adult. Young individuals are assigned into households as a priority, to avoid ending up with a high (and so unrealistic) number of young individuals not assigned to any household. 

Te distribution of non-responsible individuals’ age group and sex by primary maintainer’s age group and sex is inferred from the Hierarchical PUMF, for each household size. A non-responsible individual is linked to an household by randomly sampling one individual among the non-responsible individuals, according to the distribution defned by census micro-data. For example, a 2-persons household with a primary maintainer male aged 80–84 is more likely to include a female aged 80 than a female aged 0–4. Tis allows to preserve the distribution of household structures from the 2016 Census. If household structure is key information for the 

<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="15">Highest degree</td><td>88</td><td>Not available</td><td rowspan="3">0</td><td rowspan="3">No certificate, diploma or degree</td><td rowspan="3">No certificate, diploma or degree: Total +0 to 14 years: Total</td></tr><tr><td>99</td><td>Not applicable (&lt;15 y/o)</td></tr><tr><td>1</td><td>No certificate, diploma or degree</td></tr><tr><td>2</td><td>Secondary (high) school diploma or equivalency certificate</td><td>1</td><td>Secondary school or equivalent degree</td><td>Secondary (high) school diploma or equivalency certificate: Total</td></tr><tr><td>3</td><td>Trades certificate or diploma other than Certificate of Apprenticeship or Certificate of Qualification</td><td rowspan="11">2</td><td rowspan="11">Postsecondary degree</td><td rowspan="11">Postsecondary certificate, diploma or degree: Total</td></tr><tr><td>4</td><td>Certificate of Apprenticeship or Certificate of Qualification</td></tr><tr><td>5</td><td>Program of 3 months to less than 1 year</td></tr><tr><td>6</td><td>Program of 1 to 2 years</td></tr><tr><td>7</td><td>Program of more than 2 years</td></tr><tr><td>8</td><td>University certificate or diploma below bachelor level</td></tr><tr><td>9</td><td>Bachelor&#x27;s degree</td></tr><tr><td>10</td><td>University certificate or diploma above bachelor level</td></tr><tr><td>11</td><td>Degree in medicine, dentistry, veterinary medicine or optometry</td></tr><tr><td>12</td><td>Master&#x27;s degree</td></tr><tr><td>13</td><td>Earned doctorate</td></tr></table>


Table 9. Correspondence of the categories for the “highest degree” attribute.


<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="15">Labour force status</td><td>1</td><td>Employed - Worked in reference week</td><td rowspan="2">0</td><td rowspan="2">Employed</td><td rowspan="2">Employed: Total</td></tr><tr><td>2</td><td>Employed - Absent in reference week</td></tr><tr><td>3</td><td>Unemployed - Temporary layoff -Did not look for work</td><td rowspan="8">1</td><td rowspan="8">Unemployed</td><td rowspan="8">Unemployed: Total</td></tr><tr><td>4</td><td>Unemployed - Temporary layoff -Looked for full-time work</td></tr><tr><td>5</td><td>Unemployed - Temporary layoff -Looked for part-time work</td></tr><tr><td>6</td><td>Unemployed - New job -Did not look for Work</td></tr><tr><td>7</td><td>Unemployed - New job -Looked for full-time Work</td></tr><tr><td>8</td><td>Unemployed - New job -Looked for part-time work</td></tr><tr><td>9</td><td>Unemployed - Looked for full-time work</td></tr><tr><td>10</td><td>Unemployed - Looked for part-time work</td></tr><tr><td>11</td><td>Not in the labour force - Last worked in 2016</td><td rowspan="5">2</td><td rowspan="5">Not in labour force</td><td rowspan="5">Not in labour force: Total +0 to 14 years: Total</td></tr><tr><td>12</td><td>Not in the labour force - Last worked in 2015</td></tr><tr><td>13</td><td>Not in the labour force - Last worked before 2015</td></tr><tr><td>14</td><td>Not in the labour force - Never worked</td></tr><tr><td>99</td><td>Not applicable (&lt;15 y/o)</td></tr></table>


Table 10. Correspondence of the categories for the “labour force status” attribute.


<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="8">Household size</td><td>8</td><td>Not available</td><td rowspan="2">0</td><td rowspan="2">1 person</td><td rowspan="2">1 person: Total</td></tr><tr><td>1</td><td>1 person</td></tr><tr><td>2</td><td>2 persons</td><td>1</td><td>2 persons</td><td>2 persons: Total</td></tr><tr><td>3</td><td>3 persons</td><td>2</td><td>3 persons</td><td>3 persons: Total</td></tr><tr><td>4</td><td>4 persons</td><td>3</td><td>4 persons</td><td>4 persons: Total</td></tr><tr><td>5</td><td>5 persons</td><td rowspan="3">4</td><td rowspan="3">5 persons or more</td><td rowspan="3">5 or more persons: Total</td></tr><tr><td>6</td><td>6 persons</td></tr><tr><td>7</td><td>7 persons or more</td></tr></table>


Table 11. Correspondence of the categories for the “household size” attribute.


<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="5">Total income</td><td>88,888,888</td><td>Not available</td><td>ignored</td><td></td><td></td></tr><tr><td>99,999,999</td><td>Not applicable (&lt;15 y/o)</td><td>0</td><td>&lt;20,000 $</td><td>Total income groups Under $10,000: Total + $10,000 to $19,999: Total + 0 to 14 years: Total</td></tr><tr><td rowspan="3">Rounded value of the amount received by the individual in 2015</td><td rowspan="3"></td><td>1</td><td>20,000 $ to 59,999 $</td><td>$20,000 to $29,999 + $30,000 to $39,999 + $40,000 to $49,999 + $50,000 to $59,999</td></tr><tr><td>2</td><td>60,000 $ to 99,999 $</td><td>$60,000 to $69,999 + $70,000 to $79,999 + $80,000 to $89,999 + $90,000 to $99,999</td></tr><tr><td>3</td><td>≥100,000 $</td><td>$100,000 and over</td></tr></table>


Table 12. Correspondence of the categories for the “total income” attribute.


<table><tr><td></td><td>Code in Individual PUMF</td><td>Description in Individual PUMF</td><td>Code in synthetic population</td><td>Category in synthetic population</td><td>Characteristic in Census profile</td></tr><tr><td rowspan="2">Primary household maintainer</td><td>0</td><td>Person is not primary maintainer</td><td>0</td><td>Is not primary maintainer</td><td>Population, 2016: Total - Private dwellings occupied by usual residents</td></tr><tr><td>1</td><td>Person is primary maintainer</td><td>1</td><td>Is primary maintainer</td><td>Private dwellings occupied by usual residents: Total</td></tr></table>


Table 13. Correspondence of the categories for the “primary household maintainer” attribute.


considered use case, the assignment process should be further refned. It could take into account the occupational status, education and income of individuals when assigning them into households, and include shared fats and elderly residences, for a more exhaustive representation of household relationships. 

When an individual is added to an household, his HID attribute gets equal to the household identifer and the individual is removed from the pool of unassigned individuals. 

Remaining individuals assignment. Big households (5 persons or more) in the DA are then completed with non-responsible individuals who need to be in big households and who were not assigned in the previous step. Finally, households that are not full are flled in with unassigned non-responsible individuals according to the distribution defned by census microdata. Afer the household assignment process, each individual has an additional age attribute and a HID attribute related to his household. In some DA, a small number of households will not be full or a small number of individuals will not be assigned to an household (because the households number and sizes do not exactly match the individuals count). Te unassigned individuals have an HID attribute equal to $- 1$ . 


Algorithm 3: Household assignment algorithm


input :synth_pop_csv: Synthetic population file  
input :hh_pumf: 2016 Hierarchical PUMF  
input :geo_att_file: 2016 Geographic Attribute File  
input :pr: Province code  
input :year: Year of projection  
output :synth_pop_hh_csv: Synthetic population assigned in households file  
begin  
DA_codes $\leftarrow$ load(geo_att_file, pr)  
synth_pop $\leftarrow$ load(synth_pop_csv, pr, year)  
households_distrib $\leftarrow$ get_hh_distrib_by_hhsize_age-sex(hh_pumf)  
for DA_code in DA_codes do  
synth_pop.da $\leftarrow$ load(synth_pop, DA_code)  
prihms $\leftarrow$ load(synth_pop.da, prihm = 1)  
nb_households.da $\leftarrow$ count(prihms)  
prihms.HID $\leftarrow$ assign_unique_HID()  
hhsizes $\leftarrow$ get_hhsizes(synth_pop.da)  
for size in hhsizes do  
non_prihms_childen*size] $\leftarrow$ load(nonPRIHMS,hhsize = size, age <= 19)  
nonPRIHMS_adults*size] $\leftarrow$ load(nonPRIHMS,hhsize = size, age > 19)  
end  
for prihm in prihms do  
if prihm.hhsize == 2 persons then 

17 add one individual from non_prihms_adults[2] or non_prihms_children[2]   
18 else if prihm.hhsize in [3 persons, 4 persons, 5 or more persons] then   
19 add one individual from nonPRIHMS_adults[prihm.hhsize]   
20 add (prihm.hhsize - 2) individuals from nonPRIHMS_children[prihm.hhsize]   
21 end   
22 end   
23 synth_pop_DA $\leftarrow$ complete/big_households(synth_pop.da,nonPRIhms[5])   
24 synth_pop_DA $\leftarrow$ complete_other.households(synth_pop.da,nonPRIhms)   
25 end   
26 Write synth_pop to synth_pop_hh_csv   
27 end 

Household type assignment. A fnal step consists in assigning a type to each household. Te household type is inferred from the number of members in the household and from their age. Tis step is performed for each scenario, each projection year and for each province or territory. 

Households census categorisation. Statistics Canada classifes households into 9 types: One-census-family household without additional persons: Couple without children/Couple with children/Lone parent family, One-census-family household with additional persons: Couple without children/Couple with children/ Lone parent family, Multiple-census-family household, Non-census-family households: One person household/Two or more person non-census-family household. A census family is defned as a married couple, a common-law couple or a lone parent with at least one child living in the same dwelling. Census family households contain at least one census family. Non-census-family households are either one person living alone or at least two persons who live together but do not constitute a census family. 

Households simplifed categorisation. We defned the following simplifed categories for the household type: “One-person household”, “Couples without children”, “Couples with children”, “One-parent-family” and “Other kind of household”. We assigned the four most classical household types $8 3 \%$ of individuals in the 2016 census): “One-person household”, “Couples without children”, “Couples with children”, and “One-parent-family”, following simplistic rules regarding individuals’ ages. Other household structures (shared accommodation, more complex family household, …) are considered as “Other kind of household”. Tis process is simplistic in the way that it does not take into account couples with a large age diference, step families with little age diference between an adult and one of the children, or individuals living in a household without a family relationship. 

Household type assignment process. Algorithm 4 describes the assignment process. Households composed of one individual are one-person households. Households composed of two members having more than 16years diference are assumed to be one-parent family households. Otherwise, if both members are aged more than 16, the household is presumed to be a couple without children. For households with 3 to 6 members, the following assumptions are applied. If the two oldest members are aged more than 16 and other members are less than 16, or if the two oldest members have more than 16 years diference with the last member, the household is a couple with children. Otherwise, if the oldest member has more than 16 years diference with other members, who are all less than 16, then the household is a one-parent family. All unassigned households afer this process are considered to be other kind of households. 


Algorithm 4: Household type assignment algorithm


input :synth_pop_csv: Synthetic population file  
input :pr: Province code  
input :year: Year of projection  
output:synth_pop_hh_csv: Synthetic population assigned in households file  
1 begin  
2 synth_pop $\leftarrow$ load(synth_pop_csv,pr, year)  
3 households $\leftarrow$ get_housesols(synth_pop)  
4 for household in households do  
5 members $\leftarrow$ get_members(household)  
6 size $\leftarrow$ count(members)  
7 if size == 1 then  
8 household.hhtype $\leftarrow$ One-person household  
9 else if size == 2 then  
10 if diff age > 16 y between members then  
11 |household.hhtype $\leftarrow$ One-parent family  
12 else if both are aged > 16y then  
13 |household.hhtype $\leftarrow$ Couple without children  
14 end  
15 else if size ∈ [3,6] then 

if ((size - 2) youngest aged <16 and 2 oldest aged >16)  
or (2 oldest are 16 years older than (size - 2) youngest then  
| household.hhtype $\leftarrow$ Couple with children  
else if (size - 1) youngest aged <16 and oldest is 16 years older than (size - 1) youngest then  
| household.hhtype $\leftarrow$ One-parent family  
end  
end  
end  
for household in households_not Assigned do  
| household.hhtype $\leftarrow$ Other kind of household  
end  
Write synth_pop to synth_pop_hh_csv 

# Data Records

Te synthetic population dataset for all Canada is public and freely available on Zenodo52. Te dataset is composed of 364 fles, organised into 13 folders, one by province or territory. Each folder is named afer the province or territory and contains the synthetic population at the DA level for the province (or territory) in .csv fles. Te synthetic population is available for the year 2016, and for each of the nine projection scenarios for the years 2021, 2023 and 2030. Te CSV fles’ names refer to the year for which the synthetic population is generated. For example the fle manitoba/syn_pop/FA/synthetic_pop_2023_hh_.csv contains the synthetic population for Manitoba for the year 2023 projected according to the fast-aging scenario (afer the household assignment and household type assignment). Each csv fle contains one line per individual in the following format: index, HID, sex, prihm, agegrp, age, area, hdgree, lfact, hhsize, totinc, hhtype. Te descriptions, codes and categories of individuals attributes in the synthetic population fle are listed in Table 14. 

# Technical Validation

In order to assess the reliability of the method and the synthetic dataset, we generated a synthetic population for 2021 and compared its characteristics to the characteristics of the actual 2021 population as reported by the 2021 census53. Te comparison was performed at several resolution levels: dissemination area, national and city levels. Te results are presented at the city level for three cities of diferent sizes to illustrate the approach reliability: Toronto (most populated city in Canada, 2.8 million inhabitants), Winnipeg (6th most populated city, 749 thousand inhabitants) and Sherbrooke (30th most populated city, 173 thousand inhabitants)54. 

At each resolution level, the population was evaluated on the 2021 census characteristics published at the time of writing, i.e.: population count, population count in private dwellings, population count by sex, population count by age range, population count by income range, households count, household count by size, and household count by type. Characteristics relative to education and labour have not been published by the national statistical agency for Canada at the time of writing and have therefore not been included in the evaluation. 

Tere is no consensus on the appropriate validation metrics for synthetic population14. Following recommendations from Lovelace and Dumont55, validation at the DA level was performed by calculating three commonly-used metrics: Pearson’s correlation coefcient (r), Normalised Standardised Root Mean Square Error (NRMSE) and Relative Absolute Error (RAE). Te metrics are defned as follows: 

$$
r = \frac {\sum_ {i = 1} ^ {n} \left(o b s _ {i} - \overline {{o b s}}\right) \left(s i m _ {i} - \overline {{s i m}}\right)}{\sqrt {\sum_ {i = 1} ^ {n} \left(o b s _ {i} - \overline {{o b s}}\right) ^ {2} \left(s i m _ {i} - \overline {{s i m}}\right) ^ {2}}} \tag {1}
$$

$$
N R M S E = \frac {\sqrt {\frac {1}{n} \sum_ {i} ^ {n} \left(o b s _ {i} - s i m _ {i}\right) ^ {2}}}{\max (o b s) - \min (o b s)} \tag {2}
$$

$$
R A E _ {i} = \frac {\left| o b s _ {i} - s i m _ {i} \right|}{o b s _ {i}} \quad \forall i \in [ 1, n ] \tag {3}
$$

with $n$ the total number of DAs, $( o b s _ { 1 } , o b s _ { 2 } , . . . , o b s _ { \mathrm { n - l } } , o b s _ { \mathrm { n } } )$ the observed counts for the attribute category under consideration and $( s i m _ { 1 } , s i m _ { 2 } , . . . , s i m _ { n - 1 } , s i m _ { \mathrm { n } } )$ the synthetic population counts for the attribute category. 

In addition to comparing the aggregated socio-demographic characteristics, we checked that the synthetic individuals were realistic. To do this, we calculated the proportion of synthetic individuals whose attribute set exactly matches one of the individuals from the 2016 census micro-data. 

Dissemination area level evaluation. Te validation metrics for each attribute category across all DAs are summarised in Table 15. Te metrics indicate a good ft between the synthetic population and the census population: the correlation is high $\left( \mathbf { r } > 0 . 9 \right)$ , the NRMSE is low $( < 1 \% )$ and the RAE is low $\leq 5 0 \%$ for $7 5 \%$ of the DAs) for almost all categories. Te RAE suggest that half of the DAs represent the observed population count within 

<table><tr><td>Variable</td><td>Definition</td><td>Categories</td></tr><tr><td>index</td><td>Individual identifier</td><td>Integer unique for the province</td></tr><tr><td rowspan="2">HID</td><td rowspan="2">Household identifier</td><td>Integer unique for the province</td></tr><tr><td>-1: not assigned to an household</td></tr><tr><td rowspan="2">sex</td><td rowspan="2">Sex</td><td>0: female</td></tr><tr><td>1: male</td></tr><tr><td rowspan="2">prihm</td><td rowspan="2">First person in the household identified as a household maintainer</td><td>0: not primary maintainer</td></tr><tr><td>1: primary maintainer</td></tr><tr><td rowspan="18">agegrp</td><td rowspan="18">Age group</td><td>0: 0 to 4 years</td></tr><tr><td>1: 5 to 9 years</td></tr><tr><td>2: 10 to 14 years</td></tr><tr><td>3: 15 to 19 years</td></tr><tr><td>4: 20 to 24 years</td></tr><tr><td>5: 25 to 29 years</td></tr><tr><td>6: 30 to 34 years</td></tr><tr><td>7: 35 to 39 years</td></tr><tr><td>8: 40 to 44 years</td></tr><tr><td>9: 45 to 49 years</td></tr><tr><td>10: 50 to 54 years</td></tr><tr><td>11: 55 to 59 years</td></tr><tr><td>12: 60 to 64 years</td></tr><tr><td>13: 65 to 69 years</td></tr><tr><td>14: 70 to 74 years</td></tr><tr><td>15: 75 to 79 years</td></tr><tr><td>16: 80 to 84 years</td></tr><tr><td>17: 85 years and over</td></tr><tr><td>age</td><td>Age in completed years</td><td>Integer ∈[0;120]</td></tr><tr><td>area</td><td>Dissemination area code</td><td>a 8-digit code: a 2-digit province code, followed by a 2-digit census division code, followed by a 4-digit area code.</td></tr><tr><td rowspan="3">hdgree</td><td rowspan="3">Highest certificate, diploma or degree</td><td>0: no certificate, diploma or degree</td></tr><tr><td>1: secondary school or equivalent level</td></tr><tr><td>2: postsecondary degree</td></tr><tr><td rowspan="3">lfact</td><td rowspan="3">Labour force status</td><td>0: employed</td></tr><tr><td>1: unemployed</td></tr><tr><td>2: not in labour force</td></tr><tr><td rowspan="5">hhsize</td><td rowspan="5">Number of individuals in the household</td><td>0: 1 person</td></tr><tr><td>1: 2 persons</td></tr><tr><td>2: 3 persons</td></tr><tr><td>3: 4 persons</td></tr><tr><td>4: 5 persons or more</td></tr><tr><td rowspan="4">totinc</td><td rowspan="4">Total income, receipts that tend to be of a regular and recurring nature, before income taxes and deductions</td><td>0: &lt; 20,000 $</td></tr><tr><td>1: 20,000 $ to 59,999 $</td></tr><tr><td>2: 60,000 $ to 99,999 $</td></tr><tr><td>3: ≥100,000 $</td></tr><tr><td rowspan="5">hhtype</td><td rowspan="5">Type of relation between household members</td><td>0: Couples without children</td></tr><tr><td>1: Couples with children</td></tr><tr><td>2: One-parent-family</td></tr><tr><td>3: One-person</td></tr><tr><td>4: Other kind of household</td></tr></table>


Table 14. Individual’s attributes in the synthetic population with their defnitions and possible categories.


a diference $\leq 9 \%$ , and $7 5 \%$ of the DAs represent the observed population count within a diference $\leq 1 4 . 5 5 \%$ . Synthetic population at the DA level is less reliable under important land-use change between censuses. Areas with very high RAE regarding population counts were manually checked with Google Maps data in order to try to understand the high error. We noticed that in these DAs important land-use changes may have occurred between censuses: construction/destruction of a residential building, reallocation of a building to a diferent use, or DAs where the population vary a lot on the season/day. For example for DA 35204599, the 2016 census counts 797 individuals in 272 households. Te synthetic population predicts 886 individuals in 313 households for 2021, 

<table><tr><td>Category</td><td>Pearson&#x27;s correlation coefficient r</td><td>NRMSE %</td><td>RAE % min/q1/median/q3/max</td></tr><tr><td>Population</td><td>0.951</td><td>0.705</td><td>0.0/4.58/9.0/14.55/4,760.0</td></tr><tr><td>Population private dwellings</td><td>0.950</td><td>0.704</td><td>0.0/4.76/9.39/15.37/8,820.0</td></tr><tr><td>Households</td><td>0.953</td><td>0.621</td><td>0.0/5.26/9.62/14.29/10,133.33</td></tr><tr><td>Males</td><td>0.951</td><td>0.704</td><td>0.0/4.86/9.82/16.3/2,728.0</td></tr><tr><td>Females</td><td>0.949</td><td>0.719</td><td>0.0/4.19/8.74/14.87/6,095.0</td></tr><tr><td>0 to 4 years</td><td>0.912</td><td>0.728</td><td>0.0/12.0/26.67/50.0/6,980.0</td></tr><tr><td>5 to 9 years</td><td>0.916</td><td>0.834</td><td>0.0/11.11/25.0/44.0/6,580.0</td></tr><tr><td>10 to 14 years</td><td>0.922</td><td>0.956</td><td>0.0/11.43/25.0/43.76/5,400.0</td></tr><tr><td>15 to 19 years</td><td>0.919</td><td>1.024</td><td>0.0/11.43/25.0/46.67/4,300.0</td></tr><tr><td>20 to 24 years</td><td>0.894</td><td>1.064</td><td>0.0/12.0/26.67/50.0/1,680.0</td></tr><tr><td>25 to 29 years</td><td>0.911</td><td>0.943</td><td>0.0/12.0/25.71/47.5/4,380.0</td></tr><tr><td>30 to 34 years</td><td>0.918</td><td>0.757</td><td>0.0/11.43/25.0/45.0/3,300.0</td></tr><tr><td>35 to 39 years</td><td>0.919</td><td>0.757</td><td>0.0/11.43/24.44/44.0/1,440.0</td></tr><tr><td>40 to 44 years</td><td>0.922</td><td>0.813</td><td>0.0/11.11/24.0/42.86/2,260.0</td></tr><tr><td>45 to 49 years</td><td>0.925</td><td>1.022</td><td>0.0/10.0/22.5/40.0/1,250.0</td></tr><tr><td>50 to 54 years</td><td>0.922</td><td>0.845</td><td>0.0/10.0/22.22/40.0/830.0</td></tr><tr><td>55 to 59 years</td><td>0.925</td><td>0.638</td><td>0.0/10.0/20.0/36.67/1,960.0</td></tr><tr><td>60 to 64 years</td><td>0.925</td><td>0.534</td><td>0.0/10.0/21.11/37.5/1,240.0</td></tr><tr><td>65 to 69 years</td><td>0.923</td><td>0.524</td><td>0.0/10.77/22.86/40.0/940.0</td></tr><tr><td>70 to 74 years</td><td>0.917</td><td>0.547</td><td>0.0/12.86/26.67/46.67/940.0</td></tr><tr><td>75 to 79 years</td><td>0.887</td><td>0.680</td><td>0.0/16.0/33.33/60.0/1,620.0</td></tr><tr><td>80 to 84 years</td><td>0.865</td><td>0.908</td><td>0.0/20.0/40.0/75.0/1,900.0</td></tr><tr><td>85 to 89 years</td><td>0.813</td><td>1.838</td><td>0.0/20.0/46.67/80.0/2,190.0</td></tr><tr><td>90 to 94 years</td><td>0.784</td><td>1.190</td><td>0.0/22.86/50.0/80.0/1,540.0</td></tr><tr><td>95 to 99 years</td><td>0.664</td><td>1.579</td><td>0.0/40.0/60.0/80.0/340.0</td></tr><tr><td>100 years and over</td><td>0.391</td><td>5.193</td><td>0.0/40.0/60.0/80.0/460.0</td></tr><tr><td>1 person</td><td>0.935</td><td>0.862</td><td>0.0/6.67/13.85/25.33/5,760.0</td></tr><tr><td>2 persons</td><td>0.947</td><td>0.512</td><td>0.0/6.15/13.33/23.33/880.0</td></tr><tr><td>3 persons</td><td>0.936</td><td>0.822</td><td>0.0/8.89/20.0/36.0/1,100.0</td></tr><tr><td>4 persons</td><td>0.943</td><td>0.732</td><td>0.0/10.0/20.0/40.0/960.0</td></tr><tr><td>5 persons or more</td><td>0.919</td><td>0.665</td><td>0.0/13.33/30.0/60.0/2,860.0</td></tr><tr><td>&lt;20,000 $</td><td>0.934</td><td>1.756</td><td>0.0/54.17/75.71/100.65/1,620.0</td></tr><tr><td>20,000 $ to 59,999 $</td><td>0.944</td><td>0.691</td><td>0.0/4.44/9.47/16.8/335.86</td></tr><tr><td>60,000 $ to 99,999 $</td><td>0.941</td><td>0.841</td><td>0.0/8.24/17.14/29.52/1,980.0</td></tr><tr><td>≥100,000 $</td><td>0.919</td><td>0.875</td><td>0.0/13.12/26.67/45.0/2,980.0</td></tr><tr><td>Couples without children</td><td>0.942</td><td>0.504</td><td>0.0/9.33/20.0/40.0/1,840.0</td></tr><tr><td>Couples with children</td><td>0.920</td><td>1.117</td><td>0.0/10.0/20.0/31.71/1,280.0</td></tr><tr><td>One-parent-family</td><td>0.819</td><td>1.380</td><td>0.0/20.0/37.5/64.0/1,420.0</td></tr><tr><td>One-person</td><td>0.935</td><td>0.861</td><td>0.0/7.0/15.0/26.67/5,760.0</td></tr><tr><td>Other kind of household</td><td>0.832</td><td>1.086</td><td>0.0/20.0/47.78/100.0/4,300.0</td></tr></table>


Table 15. Validation metrics for evaluating the 2021 synthetic population by comparing the dissemination area counts with the 2021 census population in each category. Te Pearson’s correlation coefcient r, the Normalized Relative Mean Square Error (NRMSE) and the Relative Absolute Error (RAE) are indicated. RAE statistics for all dissemination areas are given with the minimum, frst quartile, median, third quartile and maximum. Te values in bold are the biggest errors.


which seems realistic. However, the 2021 census counts 293 individuals in 3 private dwellings. A land-use check shows that this DA is primarily student housing, which may explain the variations in counts between censuses. 

Te synthetic population at the DA level is less reliable for the categories: “75–79 years”, “80–84 years”, “85–89 years”, $^ { \mathfrak { a } } 9 0 \mathrm { - } 9 4$ years”, ${ } ^ { \mathfrak { a } } \bar { 9 } 5 { \bar { } } - 9 9 $ years”, “100 years and over”, “Income $< 2 0 , 0 0 0 \$ 3$ ”, “Household with 5 persons or more”, “One-parent-family” and “Other kind of household”. 

For the “Income $< 2 0 , 0 0 0 \$ 3$ category, this is because the individuals incomes from 2016 are kept, without taking into account the salary increase. If the dataset is used with particular interest for salaries, a qualifcation-based salary increase should be applied to update the individual income attribute. Similarly, for the same age, people are more qualifed in 2021 than in 2016. Te qualifcation attribute will need to be adjusted with the 2021 census data once available. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/bbfcc756a7bc6359b207c29d0c53b09e1dfd292c6629dada16eba2b1717dbc4d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/94e1af2f168cf713c835b6a8f6ab518a37f3b891adeafd33bcd368a0a30d2cf4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/3bf4cfb4cf146e3bf744ea79052ac687233f6dde93cc9ae8d3eb8ee20b3f817c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/e6fcf41d6b8a9a8719f1496c1608d782efac65bb94fd603bf72113275dede938.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/df005900a04290c94f5a59aa04e8c068ab11a2411dd11fdd9b63ce50a34a5dad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/be83e15194d9c80bc5447b4cdd74ad57c06eebab005870d5cf088389a9c8c958.jpg)



Fig. 2 National level validation: comparison of the 2021 census population (in blue) with the 2021 synthetic population (in orange) on (a) the population and households counts, (b) the sex distribution, (c) the age distribution, (d) the income distribution, (e) the household size distribution and (f) the household type distribution. Relative errors are indicated in boxes for each category. 2021 population projections and estimates appear in green and red respectively.


For the other categories, those representing the lowest proportions of the population are the least reliable at the DA level. According to the 2021 census: age groups over 74 years old each represent between $0 . 0 3 \%$ (for 100 years and over) and $3 . 4 \%$ (for 75–79 years old) of the population, households of 5 persons or more represent $8 . 4 \%$ of households, one-parent-families and other kind of households represent $8 . 7 \%$ and $1 1 . 1 \%$ of households respectively. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/8d2886596b58b8ea4049ebbcaa4a9e9471349d74178b5078b63667bf3c9bf191.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/38a3bac27ee448c9887f54611a44bd4adf06a0aa838f93c87466a6460f4276a3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/df53df94158e6f0afb9d68b56deffdb29cbba65fb8f7181b2c176c8c5898309e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/f531e1b0b2d79e906420b7a37e20601017f71b12f9488608112c2de5eb04b7fb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/8d6946c557879dba2c40c8e8a57ab0a85d7197b060b6f2940f4cbd4497fb58d8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/bd729864ae02b17269ea749604e7c195154f0dc6663d61f277489d6d6d514375.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/182acedf264bc479780fd84cab508a956aa95b8812adca45a3f11174ab90df7e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/6b636e4b4f78c73d2802d29465b1c5166c27aa5aa229c5bcef81acab90f58b0e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/9eca7b619d5387da40e01ce080195b6576b808e1da4d120f0650e4080274c561.jpg)



Fig. 3 City level validation: comparison of the 2021 census population (in blue) with the 2021 synthetic population (in orange) for Sherbrooke, Toronto and Winnipeg on (a–c) the population and households counts, (d–f) the sex distribution and (g–i) the age distribution. Relative errors are indicated in boxes for each category.


Tese categories have the highest RAE. Te error is partly explained by the fact that census profle counts are randomly rounded either up or down to a multiple of ‘5’. Te average absolute error of any value is 2.5 but the smaller the count, the larger this error as a percentage of its value. Te average relative error for a population count of 1000 is $0 . 2 5 \%$ , but if the count is 10 (as it is ofen the case for the low proportion categories at the DA level), the error is $2 5 \%$ . 

Finally, at the DA level, $9 4 . 3 \%$ of the individuals are realistic on average, with $7 5 \%$ of DAs having more than $9 5 . 4 \%$ realistic individuals. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/868b8e7e8b6ae8c016eecabb8c23b5748a82c5707cd24ceae512cc171983cd91.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/ea56ebe690d91eb9c58788e15bda619cfd0e5fa648327c777036c1a4db56799c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/51ad95e3ed6fadc70a7cacc968cc7e7472ba85e69b33c5e7f64ca3d956312686.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/b34b90e24f596a8a6d521c24e0df86dcbae7d928f75ed1afbd957c396be578a5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/15d23fc3914254bd6d2293b5dea5f1f5d3547c7a52d7d2e0084439b935b4f3ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/b18aaebd96370fddce9b4ed9866fbdd063d86ddfae93eb8c59e615d564a9195a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/74fc1f171a4eda324636ed8e56972a2f7fd2869cec95298998b6e3284a9b6849.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/e70464990158e7377b166a8e6fadc8f9b135f4c8cdd0aae76d5cdaab9e5bc731.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/faf14764e518981e8b45af007b2ddd3f87cc2442cde07401bac1d62790e4c513.jpg)



Fig. 4 City level validation: comparison of the 2021 census population (in blue) with the 2021 synthetic population (in orange) for Sherbrooke, Toronto and Winnipeg on (a–c) the income distribution, (d–f) the household size distribution and $\left( \mathbf { g - i } \right)$ the household type distribution. Relative errors are indicated in boxes for each category.


National level evaluation. Figure 2 presents a comparison of the 2021 synthetic population and the 2021 census data at the national level. Histograms show comparisons regarding the population count, population count in private dwellings, sex distribution, age distribution, income distribution, households count, household size distribution, and household type distribution. Te relative error for each category appears in boxes on the histograms. 

Population counts and distributions by sex and by age in the 2021 synthetic population show little diference from the 2021 census and are similar to the 2021 projections and estimates. Te census population counts are not adjusted for undercoverage or overcoverage, so the population projections and estimates difer from the census and are generally higher and closer to reality. Tis diference is refected in the synthetic population and accounts for part of the diference with the 2021 census. For instance, the $+ 5 . 9 9 \%$ error for the age range $^ { \infty } 2 0 { - } 2 4 ^ { \infty }$ means that while census 2021 reports $6 \%$ of population is aged 20–24 years old, our model predicted $6 . 3 6 \%$ . Te 2021 synthetic population provides a good prediction of the distribution for the household types and sizes. Similarly to the DA level, the prediction is less reliable regarding the income distribution because this attribute distribution evolved from 2016 to 2021 and has not been adjusted. Finally, at the national level, $9 5 . 7 \%$ of all synthetic individuals present realistic sets of attributes. 

City level evaluation. Figures 3, 4 present a comparison of the 2021 synthetic population and the 2021 census data at the city level for Sherbrooke, Toronto and Winnipeg. Histograms and relative errors are shown for each attribute and each category. 

Te fgures show that the 2021 synthetic population counts and distributions present a good ft with statistics from 2021 census. Te synthetic population reproduces well the cities’ specifcities: for example, a high proportion of 25–34 years old in Toronto and a high number of one-person households in Sherbrooke. Moreover, Sherbrooke’s synthetic population has $9 7 . 3 \%$ of realistic individuals on average $9 5 \%$ of DAs with $> 9 0 . 2 \%$ of realistic individuals), Toronto’s synthetic population has $9 6 . 3 \%$ realistic individuals on average $9 5 \%$ of DAs with ${ > } 9 2 . 4 \%$ of realistic individuals), and Winnipeg’s synthetic population has $9 5 . 5 \%$ realistic individuals on average $9 5 \%$ of DAs with ${ > } 8 9 \%$ of realistic individuals). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/bcc9cec4070c2f8c93fe76691d42977f0d96c68ecc0b8eb8a306a95be7b28841.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/5a22f2230fe0885e6676efa44f68ca1ebbc859b33aa8d7e0e59f4123b89ad326.jpg)



Fig. 5 Sherbrooke synthetic population density by dissemination area for 2023 and 2030 relative to 2016.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/d113860bc4410939444e6471daf9da0e08173631a61e4876cdda897148e1b818.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/f67c9abbfadf985910de381d516b21a18c0fbfeabd230bf6f9af6e6cfa9188e8.jpg)



Fig. 6 Toronto synthetic population density by dissemination area for 2023 and 2030 relative to 2016.


Finally, in order to illustrate the 2023 and 2030 synthetic populations, the evolution of the synthetic population density by DA from 2016 to 2023 and from 2016 to 2030 is presented for each city in Figs. 5–7. Te DAs boundaries are the ones from 2016 census. A population densifcation can be observed in almost all areas, with greater densifcation in already dense areas. Tis is due to the way the population is projected in the future years. If the projection predicts an increase in the province population, then some synthetic individuals are drawn randomly from the 2016 province’s synthetic population to be duplicated (according to the age and sex projections) in order to expand the province synthetic population. A highly populated DA in 2016 is therefore more likely to have its individuals duplicated than a sparsely populated DA. 

# Usage Notes

The synthetic population can be used directly to initialise agent-based models. Synthetic populations for specifc zones can be extracted from the population dataset by identifying the zone’s geographical code in the 2016 Geographic Attribute File, getting the corresponding DA codes and fltering the synthetic individuals which have their “area” attribute within the selected DA codes. DAs boundaries can be geolocalised using the 2016 census boundaries fle provided by Statistics Canada56. Figures 5–7 give an example of the DA spatial resolution for three cities. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/949d17036305a094050450c863aad29e730f390e7729884e4edd10eb46ec9087.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/309eaaea02e0706d09e49fa6568b5115d3a419d217096bb2992c4bd8c1ce26ab.jpg)



Fig. 7 Winnipeg synthetic population density by dissemination area for 2023 and 2030 relative to 2016.


For proper use of the data, it is important to note that the projected populations are independent from one year to another. Tis means, for example, that the individual with index 2 in the Manitoba population for 2021 is not the same as index 2 in the Manitoba population for 2023. 

Te generation process is partly stochastic which induces some limitations. We provide only one instance of the synthetic population by year and scenario and while the model seems stable, additional analyses of the variance between instantiations should be performed. We provide the code for users who would like to generate multiple instances of the model and perform a sensitivity analysis. In addition, if the users want to generate a synthetic population themselves (for a diferent projection year or using diferent methods of assigning households or household types), the scripts developed for this work are provided. Te scripts workfow with the input fles and parameters of each script are described in Fig. 1. 

If the population to be synthesise covers a large area, HPC facilities are required to run the scripts in a reasonable time. Te synthetic population was generated on ARC4, which is part of the High Performance Computing facilities at the University of Leeds, UK. ARC4 is a Linux-based HPC cluster, based on the CentOS 7 distribution, supporting Son of Grid Engine to run parallel batch jobs. Te generation script can be parallelised to generate DAs 250 by 250 for each province. Te projection script is fast and does not need to be parallelised. Te household assignment and household type assignment can be parallelised to generate DAs 1,000 by 1,000 for each province. Shell scripts to run Python scripts in parallel on HPC facilities are provided with the code, as well as additional Python scripts to merge output fles that were generated in parallel. Te parallelisation process is documented to guide the user in its execution. 

# Code availability

Te python scripts (python 3.10) developed for the generation and validation of the synthetic dataset are publicly and freely accessible on Zenodo57. Te scripts use the following python packages: pandas (1.4.4), numpy (1.23.2), pyreadstat (1.19), scipy (1.9.1) for the Pearson’s correlation coefcient computation, scikit-learn (1.1.2) for the RMSE computation, and matplotlib (3.5.3) to generate the charts. All these python packages are available from the Python Package Index: https://pypi.org/. Te humanleague package (2.1.10) providing the QISI and IPF implementations is available from the Python Package Index and on Zenodo42. 

Received: 26 September 2022; Accepted: 17 February 2023; 

Published: xx xx xxxx 

# References



1. Kagho, G. O., Balać, M. & Axhausen, K. W. Agent-based models in transport planning: current state, issues, and expectations. In Te 9th International Workshop on Agent-based Mobility, Traffic and Transportation Models, Methodologies and Applications (ABMTRANS), 726–732, (2020). 





2. Pagani, A., Ballestrazzi, F., Massaro, E. & Binder, C. R. ReMoTe-S. Residential mobility of tenants in Switzerland: an agent-based model. Journal of Artifcial Societies and Social Simulation 25, 4 (2022). 





3. Li, F., Li, Z., Chen, H., Chen, Z. & Li, M. An agent-based learning-embedded model (ABM-learning) for urban land use planning: A case study of residential land growth simulation in Shenzhen, China. Land Use Policy 95, 104620 (2020). 





4. Oh, S. et al. Assessing the impacts of automated mobility-on-demand through agent-based simulation: a study of Singapore. Transportation Research Part A: Policy and Practice 138, 367–388 (2020). 





5. Balać, M., Rothfeld, R. L. & Hörl, S. Te Prospects of on-demand urban air mobility in Zurich, Switzerland. 2019 IEEE Intelligent Transportation Systems Conference (ITSC) 906–913 (2019). 





6. Chouaki, T. & Puchinger, J. Agent based simulation for the design of a mobility service in the Paris-Saclay area. In 23rd EURO Working Group on Transportation Meeting, EWGT 2020, 16–18 September 2020, Paphos, Cyprus (2021). 





7. Noeldeke, B., Winter, E. & Ntawuhiganayo, E. B. Representing human decision-making in agent-based simulation models: agroforestry adoption in rural Rwanda. Ecological Economics 200, 107529 (2022). 





8. Maggi, E. & Vallino, E. Price-based and motivation-based policies for sustainable urban commuting: an agent-based model. Research in Transportation Business & Management 39, 100588 (2021). 





9. Furtado, B. A. PolicySpace2: modeling markets and endogenous public policies. Journal of Artifcial Societies and Social Simulation 25, 8 (2022). 





10. Baccega, D. et al. An agent-based model to support infection control strategies at school. Journal of Artifcial Societies and Social Simulation 25, 2 (2022). 





11. Retzlaf, C. O. et al. Fear, behaviour, and the COVID-19 pandemic: a city-scale agent-based model using socio-demographic and spatial map data. Journal of Artifcial Societies and Social Simulation 25, 3 (2022). 





12. Horni, A., Nagel, K. & Axhausen, K. W. Te Multi-Agent Transport Simulation MATSim (London: Ubiquity Press, 2016). 





13. Casas, J., Ferrer, J. L., Garcia, D., Perarnau, J. & Torday, A. Trafc Simulation With Aimsun (Springer New York, 2010). 





14. Chapuis, K., Taillandier, P. & Drogoul, A. Generation of synthetic populations in social simulations: a review of methods and practices. Journal of Artifcial Societies and Social Simulation 25, 6 (2022). 





15. Lomax, N., Smith, A. P., Archer, L., Ford, A. & Virgo, J. An open-source model for projecting small area demographic and land-use change. Geographical Analysis 54, 599–622 (2022). 





16. Wheaton, W. et al. Synthesized population databases: a US geospatial database for agent-based models. Methods report (RTI Press) (2009). 





17. Sexton, W., Abowd, J. M., Schmutte, I. M. & Vilhuber, L. Synthetic population housing and person records for the United States. Zenodo. https://doi.org/10.5281/zenodo.556121 (2017). 





18. Hörl, S. & Balać, M. Synthetic population and travel demand for Paris and Île-de-France based on open and publicly available data. Transportation Research Part C: Emerging Technologies 130, 103291 (2021). 





19. Agriesti, S., Roncoli, C. & Nahmias-Biran, B.-H. Assignment of a synthetic population for activity-based modeling employing publicly available data. ISPRS International Journal of Geo-Information 11 (2022). 





20. Xu, Z. et al. A synthetic population for modelling the dynamics of infectious disease transmission in American Samoa. Scientifc Reports 7, 16725 (2017). 





21. Balać, M. & Hörl, S. Synthetic population for the state of California based on open-data: examples of San Francisco Bay area and San Diego County. In 100th Annual Meeting of the Transportation Research Board (TRB) (2021). 





22. Lim, P. P. Population synthesis for travel demand modelling in Australian capital cities. Ph.D. thesis, Institute for Social Science Research, Te University of Queensland (2020). 





23. Fatmi, M. R. & Muhammad, A. H. Baseline synthesis and microsimulation of life-stage transitions within an agent-based integrated urban model. In 8th International Conference on Ambient Systems, Networks and Technologies, ANT-2017 and the 7th International Conference on Sustainable Energy Information Technology, SEIT 2017 (2017). 





24. Perez, L., Dragicevic, S. & Gaudreau, J. A geospatial agent-based model of the spatial urban dynamics of immigrant population: A study of the island of Montreal, Canada. PLOS ONE 14, 1–23 (2019). 





25. Miller, E. J. & Roorda, M. J. Prototype model of household activity-travel scheduling. Transportation Research Record 1831, 114–121 (2003). 





26. Hafezi, M. H. & Habib, M. A. Synthesizing population for microsimulation-based integrated transport models using Atlantic Canada micro-data. In Te 1st International Workshop on Information Fusion for Smart Mobility Solutions (IFSMS’14), 410–415 (2014). 





27. Hafezi, M. H. & Habib, M. A. Development and evaluation of an algorithm to produce the population in regional level and dissemination area level. In Canadian Transportation Research Forum 50th Annual Conference - Another 50 Years: Where to From Here?//Un autre 50 ans: qu’en est-il à partir de maintenant? 15 (2015). 





28. Yameogo, B. F., Gastineau, P., Hankach, P. & Vandanjon, P. O. Comparing methods for generating a two-layered synthetic population. Transportation Research Record 2675, 136–147 (2020). 





29. Stephan, F. F. An iterative method of adjusting sample frequency tables when expected marginal totals are known. Te Annals of Mathematical Statistics 13, 166–178 (1942). 





30. Ye, X., Konduri, K. C., Pendyala, R. M., Sana, B. & Waddell, P. Methodology to match distributions of both household and person attributes in generation of synthetic populations. In 88th Annual Meeting of the Transportation Research Board (2009). 





31. Müller, K. A generalized approach to population synthesis. Ph.D. thesis, ETH Zurich (2017). 





32. Williamson, P., Birkin, M. & Rees, P. H. Te estimation of population microdata by using data from small area statistics and samples of anonymised records. Environment and Planning A: Economy and Space 30, 785–816 (1998). 





33. Harland, K., Heppenstall, A., Smith, D. & Birkin, M. Creating realistic synthetic populations at varying spatial scales: a comparative critique of population synthesis techniques. Journal of Artifcial Societies and Social Simulation 15, 1 (2012). 





34. Farooq, B., Bierlaire, M., Hurtubia, R. & Flötteröd, G. Simulation based population synthesis. Transportation Research Part B: Methodological 58, 243–263 (2013). 





35. Saadi, I., Mustafa, A., Teller, J., Farooq, B. & Cools, M. Hidden Markov model-based population synthesis. Transportation Research Part B: Methodological 90, 1–21 (2016). 





36. Sun, L. & Erath, A. A Bayesian network approach for population synthesis. Transportation Research Part C: Emerging Technologies 61, 49–62 (2015). 





37. Zhou, M., Li, J., Basu, R. & Ferreira, J. Creating spatially-detailed heterogeneous synthetic populations for agent-based microsimulation. Computers, Environment and Urban Systems 91, 101717 (2022). 





38. Garrido, S., Borysov, S. S., Pereira, F. C. & Rich, J. Prediction of rare feature combinations in population synthesis: application of deep generative modelling. Transportation Research Part C: Emerging Technologies 120, 102787 (2020). 





39. Lovelace, R., Birkin, M., Ballas, D. & van Leeuwen, E. Evaluating the performance of iterative proportional ftting for spatial microsimulation: new tests for an established technique. Journal of Artifcial Societies and Social Simulation 18, 21 (2015). 





40. Lovelace, R. & Ballas, D. ‘Truncate, replicate, sample’: a method for creating integer weights for spatial microsimulation. Comput. Environ. Urban Syst. 41, 1–11 (2013). 





41. Smith, A., Lovelace, R. & Birkin, M. Population synthesis with quasirandom integer sampling. Journal of Artifcial Societies and Social Simulation 20, 14 (2017). 





42. Smith, A., Russell, T. & Lovelace, R. virgesmith/humanleague: v2.1.10. Zenodo. https://doi.org/10.5281/zenodo.6371111 (2022). 





43. Bae, J. W., Paik, E., Kim, K., Singh, K. & Sajjad, M. Combining microsimulation and agent-based model for micro-level population dynamics. In International Conference on Computational Science 2016, ICCS 2016 (2016). 





44. Statistics Canada. Hierarchy of standard geographic areas for dissemination, 2016 Census. https://www12.statcan.gc.ca/censusrecensement/2016/ref/dict/fgures/f1_1-eng.cfm (2016). 





45. Statistics Canada. Individuals File, 2016 Census of Population (Public Use Microdata Files) (98M0001X). https://www150.statcan. gc.ca/n1/en/catalogue/98M0001X (2019). 





46. Statistics Canada. Hierarchical File, 2016 Census of Population (Public Use Microdata Files) (98M0002X). https://www150.statcan. gc.ca/n1/en/catalogue/98M0002X (2019). 





47. Statistics Canada. Census Profle for Canada, provinces, territories, CDs, CSDs and DAs - REGION only, 2016 Census – Statistics Canada Catalogue no. 98–401-X2016044. https://www150.statcan.gc.ca/n1/en/catalogue/98-316-X2016001 (2016). 





48. Statistics Canada. Geographic Attribute File, 2016 Census – Statistics Canada Catalogue no. 92-151-2016001. https://www150.statcan. gc.ca/n1/en/catalogue/92-151-X2016001 (2016). 





49. Statistics Canada. Projected population, by projection scenario, age and sex, as of July 1 (x 1,000) – Statistics Canada Table 17–10-0057-01. https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701 (2018). 





50. Statistics Canada. 2016 Census Public Use Microdata File (PUMF). Individuals File. Abacus Data Network https://hdl.handle. net/11272.1/AB2/GDJRT8 (2019). 





51. Statistics Canada. 2016 Census Public Use Microdata File (PUMF): Hierarchical fle. Abacus Data Network https://hdl.handle. net/11272.1/AB2/PYYXXR (2019). 





52. Prédhumeau, M. & Manley, E. Synthetic population for Canada at the DA level for 2016, 2021, 2023 and 2030. (2.1.0). Zenodo. https://doi.org/10.5281/zenodo.7572117 (2023). 





53. Statistics Canada. Census Profle. 2021 Census of Population. Statistics Canada Catalogue number 98-316-X2021001. https://www12. statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm?Lang=E (2022). 





54. Statistics Canada. Table 98–10-0002-01 Population and dwelling counts: Canada and census subdivisions (municipalities). https://doi. org/10.25318/9810000201-eng (2022). 





55. Lovelace, R., Dumont, M., Ellison, R. & Zaloznik, M. Spatial Microsimulation ith R (Chapman and Hall/CRC, 2016). 





56. Statistics Canada. 2016 Census - Boundary fles. https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/boundlimit-2016-eng.cfm (2016). 





57. Prédhumeau, M. & Manley, E. maprdhm/synpopCanada: v2.0.0. Zenodo. https://doi.org/10.5281/zenodo.7569219 (2023). 



# Acknowledgements

Tis research has been conducted as part of the RAIM project (Responsible Automation for Inclusive Mobility: Using AI to Develop Future Transport Systems that Meet the Needs of Ageing Populations), funded by the ESRC-Canada AI initiative (ES/T012587/1). 

# Author contributions

M.P.: conceptualisation, methodology, sofware, validation, visualisation, writing - original draf, review and editing. E.M.: conceptualisation, validation, writing - review and editing, supervision, project administration, funding acquisition. 

# Competing interests

Te authors declare no competing interests. 

# Additional information

Correspondence and requests for materials should be addressed to M.P. 

Reprints and permissions information is available at www.nature.com/reprints. 

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afliations. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-27/d857b994-f12c-4d86-9702-76eb8f371b01/58a618b188ea0f0e9385145313b1ede140fc89ceec0ad40ca92ab7b7bee7c5a3.jpg)


Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or 

format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. Te images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/. 

$^ ©$ Te Author(s) 2023 