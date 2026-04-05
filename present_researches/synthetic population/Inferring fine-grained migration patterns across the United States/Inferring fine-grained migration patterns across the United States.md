# Inferring fine-grained migration patterns across the United States

Received: 16 June 2025 

Accepted: 16 December 2025 

Published online: 26 December 2025 

Check for updates 

Gabriel Agostini 1 , Rachel Young2,3,4, Maria Fitzpatrick5 , Nikhil Garg1 & Emma Pierson 3 

Fine-grained migration data illuminate demographic, environmental, and health phenomena. However, United States migration data have serious drawbacks: public data lack spatial granularity, and higher-resolution proprietary data suffer from multiple biases. To address this, we develop a method that fuses high-resolution proprietary data with coarse Census data to create MIGRATE: annual migration matrices capturing flows between 47.4 billion US Census Block Group pairs—approximately four thousand times the spatial resolution of current public data. Our estimates are highly correlated with external ground-truth datasets and improve accuracy relative to raw proprietary data. We use MIGRATE to analyze national and local migration patterns. Nationally, we document demographic and temporal variation in homophily, upward mobility, and moving distance—for example, rising moves into top-income-quartile block groups and racial disparities in upward mobility. Locally, MIGRATE reveals patterns such as wildfire-driven out-migration that are invisible in coarser previous data. We release MIGRATE as a resource for migration researchers. 

Fine-grained migration data, which record the number of people relocating from one geographic area to another, are essential for understanding a range of social, environmental, and health phenomena. Migration data illuminate responses to environmental disasters and climate change1–4 , responses to economic stresses and opportunities5–8 , patterns of social change9 , consequences ofconflicts10,11, effects of the COVID-19 pandemic12, housing instability13,14, opportunities5-8, urban-suburban migration patterns15,16, and political polarization17. 

But migration datasets within the United States have serious limitations. The most fine-grained publicly available national datasets track migration at the county level18. While widely used, these datasets lack sufficient spatial resolution to study a number of phenomena where previous research has revealed important variation at the subcounty level. For example, research on flood-risk-induced migration uses models at the much more granular Census block level19, arguing that this spatial granularity is necessary due to the highly localized nature of flood risk20. Other research on climate-induced vulnerability, eviction, or displacement similarly models risk at a sub-county (Census 

Tract) level14,21 or even at the household level22. Research on housing instability often models Census Block Groups (CBGs)23 or even specific building complexes24. All of these applications testify to the need for highly granular data to facilitate accurate study of many important migration-related phenomena. Additionally, movers within the same county have accounted for more than half of migratory flows in the United States in all years from 2006 to $2 0 1 9 ^ { 2 5 }$ 5. Publicly available, county-level migration datasets are too coarse to study these important migration patterns. 

Proprietary migration datasets offer greater granularity. Data aggregators like Infutor26 combine many data sources—including voter files, property deeds, credit files, and phone books27–29—to attempt to infer address histories for individual-level movers. Such datasets have been widely used because they are extremely temporally and spatially fine-grained5,13,28,30–32. A long literature which provides more recent or granular migration estimates using non-traditional data sources, including social media and digital advertisement data, testifies to the power of new data sources4,10,11,33,34. Previous work also suggests that 

1 Cornell Tech, New York City, NY, USA. 2 University of Minnesota, Minneapolis, MN, USA. 3 University of California, Berkeley, Berkeley, CA, USA. 4 Princeton University, Princeton, NJ, USA. 5 Cornell University, Ithaca, NY, USA. e-mail: emmapierson@berkeley.edu 

address history data do contain valuable signal which correlates with external datasets, as validated by cross-referencing the data with hurricanes or public housing foreclosures13, and with marginal population counts in the region of interest27. However, proprietary address history datasets have three major disadvantages. First, they are not publicly available, limiting their utility to researchers. Second, they require extensive computational pipelines, and substantial computational resources to clean and map to standardized geographical areas to facilitate subsequent analysis. Third, and most fundamentally, they combine multiple imperfect data sources using proprietary algorithms, and thus contain noise and biases. For example, Infutor and other consumer record datasets have been shown to overrepresent higher-income and majority-group populations in some settings35. 

To address the limitations of existing migration data, we create and release Migration Inference for GRAnular Trend Estimation (MIGRATE): fine-grained migration estimates that combine the strengths of both aforementioned types of data by harmonizing biased but fine-grained proprietary data from the data aggregator Infutor with reliable but coarse Census data. To produce our estimates, we develop a data fusion method36,37 based on iterative proportional fitting (IPF)38–40 to reconcile the raw Infutor migration data with the more reliable Census constraints. The output of our method is a set of yearly inferred United States migration matrices at the CBG level from 2010 to 2019. Our matrices capture migration flows between 47.4 billion pairs of CBGs, making MIGRATE approximately 4600 times more granular than the county-county flows publicly available on the 5-year level, and 18 million times more granular than the state-state flows publicly available on the 1-year level. We comprehensively validate MIGRATE by comparing to external data sources, showing that it correlates well with ground-truth population counts and migration flows, and improves accuracy and reduces demographic bias compared to raw Infutor data, which overcounts rural, older, white, and home-owning populations. We then use MIGRATE to analyze both national and local migration. Nationally, we reveal both temporal and demographic variation in migration homophily, upward mobility, and moving distance. We find, for example, that people are increasingly likely to move to top-income-quartile CBGs, but also provide evidence of racial disparities: movers from plurality Black CBGs are less likely, and movers from plurality Asian CBGs more likely to move to topincome-quartile CBGs, and these disparities persist even when controlling for income of origin CBG. Locally, we show that MIGRATE can illuminate important migration patterns, including dramatic increases in out-migration in response to California wildfires, that are invisible in county-level data. To provide a foundation for more precise migration research in the social, environmental, urban, and health sciences, we release MIGRATE for non-profit research use at our website. 

# Results

# Creating MIGRATE

Here, we provide a high-level summary of our method for inferring fine-grained migration matrices by harmonizing Census and Infutor data. In the “Methods” Section, we provide full details of processing the Infutor dataset, processing Census data, and harmonizing both data sources. 

We first preprocess the raw Infutor data into yearly migration matrices. The raw Infutor data consists of sequences of addresses for individual people. We use these address sequences to compute the number of people moving between each pair of geographic areas. In particular, we estimate the matrices $\boldsymbol { E } ^ { ( t ) }$ , where entries $\bar { E } _ { i j } ^ { ( t ) }$ represent the number of people who reside in $\mathbf { C B G } j$ some time during year t and resided in CBG i a year prior. We correct populations to account for birth, deaths, and international migration; full details in the “Methods” Section. (For a subset of individuals, Infutor also provides estimated demographic information—e.g., age and gender—but we do not use 

this in our analysis, both because it is missing for roughly half the population, and because it contains biases35,41). 

We provide a conceptual overview of the preprocessing pipeline here and full details in “Methods”. We first construct cleaned monthly address histories for each individual in the Infutor dataset by reconciling inconsistencies in address start/end dates, discarding unreliable records (e.g., postal boxes), and modeling uncertainty when multiple addresses are active. These monthly address distributions are then aggregated in a way that simulates American Community Survey (ACS) responses about residence one year prior, producing annual migration flow estimates between pairs of addresses. Finally, to convert the address-address migration matrices to CBG-CBG migration matrices, each address is mapped to one or more CBGs. When the raw Infutor data contains a precise street address $( 9 0 . 4 0 \%$ of all addresses in the dataset), we map each address to a unique 2010 CBG with a state-ofthe-art geocoder42,43. We obtain matches for $9 2 . 1 8 \%$ of these addresses and confirm by hand inspection of 200 addresses that this yields highly accurate results. The remaining addresses are incomplete or imprecise (e.g., contain only ZIP codes); we probabilistically map these addresses to multiple CBGs, with the weight of each CBG proportional to the population in the CBG intersecting the ZIP code. We are able to map $9 9 . 2 1 \%$ of all addresses in Infutor to CBGs. The final output of this preprocessing procedure is the Infutor yearly migration matrices $\boldsymbol { E } ^ { ( t ) }$ , which capture migration counts between pairs of CBGs as estimated from Infutor data alone. We provide summary statistics for the raw Infutor data and the processed migration matrices for the 2010–2019 time period in Table 1 and additional descriptive statistics in Table 2. 

Having produced the raw Infutor yearly migration matrices ${ \cal { E } } ^ { ( t ) }$ , we next reconcile them with more reliable but less granular Census data by rescaling selected entries of $\boldsymbol { E } ^ { ( t ) }$ ). We rescale to different Census datasets in sequence, rather than simultaneously, because the datasets are not totally consistent with each other. Specifically, we first rescale the entries of $\cdot \boldsymbol { E } ^ { ( t ) }$ to match CBG populations from the Census and 5-year ACS estimates; then the 1-year ACS counts of movers and non-movers by state; then the 1-year ACS state-to-state flows; and finally the 1-year county populations from the Population Estimates Program (PEP). The motivation for this ordering is that the 1-year county populations are more precisely estimated than the other datasets, and we thus prioritize matching to them. While CBG populations and state flows are estimated by the ACS and often contain large sampling errors, county populations are estimated by PEP using more precise large-scale administrative datasets. We verify that incorporating each of these datasets improves our performance on the validations discussed below, and that the performance is not overly sensitive to the order in which we match to the datasets (see Supplementary Table 1). 

To perform the final rescaling (matching 1-year county populations), we apply an iterative procedure based on the classical IPF 


Table 1 | Summary statistics for raw Infutor data and Census data during the period of interest (2010–2019)


<table><tr><td>Individuals with active records (any time in 2010–2019)</td><td>374,217,253</td></tr><tr><td>Individuals with active records in a given year (average across 2010–2019)</td><td>231,270,602</td></tr><tr><td>US population in 2010</td><td>309,327,143</td></tr><tr><td>US population in 2019</td><td>328,329,953</td></tr><tr><td>Address records per active individual (mean)</td><td>2.67</td></tr><tr><td>Address records per active individual (median)</td><td>2</td></tr></table>


We define “individuals with active records” as those whose [earliest Infutor date observed, latest Infutor date observed] interval intersects a given time period. Comparing the average yearly number of active individuals in the Infutor data to the Census population (second row) shows that Infutor under-counts the population. (Table 2 provides the number of active individuals in each year from 2010 to 2019). Active individuals have on average between two and three addresses, translating to one or two moves during the decade. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/3ca5844a5742087beec9a07313c5b9f34e5afcd9f1dce60d6f80528607f28f8a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/edb537e24039dcd68e5d628be366b73b1446567f3eb96d589c50460f0043fc3f.jpg)



Fig. 1 | MIGRATE estimates. We estimate annual migration flows between all pairs of Census Block Groups (CBGs) from 2010 to 2019. a Average MIGRATE estimates of out-migration rates across the entire United States. b, c MIGRATE estimates of out-migration rates within New York City. MIGRATE estimates reveal granular spatial patterns invisible in publicly available county-to-county data (inset plot b). Out-migration rates for CBGs with fewer than 100 people are omitted.


algorithm38,39: specifically, we scale blocks of rows and columns to agree with the annual county populations in years t−1 and t, respectively, alternating between scaling rows and scaling columns until the procedure converges. Only the final rescaling is performed using full IPF; the other rescalings, to CBG populations, 1-year counts of movers and non-movers, and 1-year state-to-state flows, are performed only once at the beginning of the process, as opposed to iteratively, to avoid overfitting to these more noisily estimated datasets. The resulting migration matrices constitute our MIGRATE estimates. In the “Methods” Section, we provide additional details on these scalings. 

Naive implementations of our harmonization algorithm impose prohibitive computation times due to the size of the matrices involved. We rely on the sparsity of the problem to dramatically reduce both memory and time requirements, allowing our procedure to converge within 2 hours using 16G of memory and 8 cores. Our general approach —i.e., rescaling submatrices to match Census constraints—can straightforwardly be adapted to accommodate other sources of Census data. 

Figure 1 depicts the MIGRATE estimates. Figure 1a plots the average out-migration rates for all CBGs in the United States in the 2010–19 period, highlighting the spatial granularity of the estimates. This granularity often reveals important patterns which are invisible at the county-county level, as we illustrate by zooming in on New York City and comparing the county-level rates obtained from ACS flows (Fig. 1b) to the CBG-level out-migration rates inferred by MIGRATE (Fig. 1c). 

# Comparing MIGRATE to Census data

MIGRATE estimates are well-correlated with all available Census measures of population counts and flows (Fig. 2a–c). By design, MIGRATE estimates perfectly match state and county populations (because the last step in our harmonization procedure is to match to county populations). MIGRATE estimates also achieve Pearson correlation $\rho { = } 0 . 9 9 7$ with 5-year Census Tract populations, and $\rho { = } 0 . 9 9 6$ with 5-year CBG populations (Fig. 2a). MIGRATE estimates of the number of movers between each pair of states and each pair of counties are also highly correlated with ACS estimates $\scriptstyle ( \rho = 0 . 9 9 8$ and 0.957 for states and counties respectively; Fig. 2b). We exclude people who remain within the same state or county from this calculation because most people do not move, which artificially inflates the correlation; Supplementary Table 1 reports the correlation without this exclusion, which is nearly perfect. Finally, MIGRATE estimates of state and county in-migration rates (i.e., the number of people moving into an area as a fraction of the area’s population) are well-correlated with ACS estimates (Fig. 2c; $\rho { = } 0 . 9 8 7$ and 0.715 for state and county in-migration 

rates, respectively, weighting by area population). Supplementary Table 1 reports correlations with additional Census quantities (e.g., inmigration counts as opposed to rates), which remain high. We note that even for Census quantities that are used in our harmonization procedure, like CBG populations, we would expect correlations between Census and MIGRATE to be high but not perfect, because the Census datasets are not totally consistent with each other. 

Overall, these validations demonstrate that MIGRATE estimates are highly correlated with ground-truth Census data. This is not merely because of variation in state or county populations, since it remains true when examining in-migration rates, which normalize for population, as well as when examining Census Tracts and CBGs, which do not vary as significantly in population. Nor is it due merely to the fact that most people do not move, since we examine correlations for movers specifically, as well as correlations with in-migration rates. Finally, it is not merely a consequence of overfitting to Census data, because it remains true on held-out datasets that are not used in our harmonization procedure, a standard check for statistical estimation methods44. Specifically, MIGRATE estimates remain highly correlated with county-county mover counts and county in-migration rates, which are not used in our estimation procedure. As a further held-out validation, we verify that our estimation procedure yields highly correlated estimates with each Census data source, even when we remove it from the datasets used for estimation. For example, we remove CBGlevel Census populations from our estimation datasets, and verify that our resulting estimates remain highly correlated with data below county level $\scriptstyle ( \rho = 0 . 8 8 8$ with Census Tract populations and $\rho { = } 0 . 8 5 6$ with CBG populations). See Supplementary Table 1 for full results. 

# Reduction in error relative to Infutor data

We show that MIGRATE estimates also increase agreement with Census datasets compared to using the Infutor data alone. We note that Infutor systematically undercounts the population, as shown in Table 1; to compare to Infutor as generously as possible, and in particular to ensure that MIGRATE does not reduce error due to trivial rescalings, we multiply each raw Infutor matrix $\boldsymbol { E } ^ { ( t ) }$ by a scaling factor so it matches the national yearly Census population prior to conducting these comparisons. Figure 2d–f report the reduction in error that MIGRATE estimates achieve relative to Infutor data. We compute root mean squared error (RMSE) between (1) MIGRATE estimates and ground-truth Census data and (2) Infutor data and ground-truth Census data, and report the reduction in RMSE from using MIGRATE estimates. We report average reduction in RMSE across all data releases between 2010 and 2019; error bars represent standard deviation across these releases. 

MIGRATE estimates eliminate error in state and county populations because they are constructed to match county-level estimates. They also reduce error in Census Tract and CBG populations by an average of $8 5 . 9 \%$ and $8 3 . 6 \%$ respectively (Fig. 2d); in state-to-state movers and county-to-county movers by $8 7 . 5 \%$ and $4 2 . 3 \%$ (Fig. 2e); and in state in-migration and county in-migration rates by $8 7 . 3 \%$ and $5 1 . 8 \%$ (Fig. 2f). These gains persist even when using held-out datasets that are not used in estimating MIGRATE (namely, county-to-county flows and county in-migration rates). We also repeat the validation above where we successively remove types of data (e.g., CBG-level populations) from our estimation procedure, and verify that our estimation procedure still reduces error in reproducing those held-out data types (Supplementary Table 1). 

Collectively, these validations demonstrate that MIGRATE estimates increase agreement with Census datasets, including held-out datasets, compared to raw Infutor data. We provide error reductions for additional metrics (all flows, non-movers, and in-migration counts as opposed to rates) in Supplementary Table 1, and conduct additional validations on synthetic data (Supplementary Fig. 1). 


aPopulation counts


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/f51a39556d41f3c617a07da2e739180f4c19559f6bd97373451f2e09493c8063.jpg)



d


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/2046db615d9d47c88c3849708dba455c0d100757d1ad22086c3778b5c07afdf3.jpg)



bPopulation flows (area movers only)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/aa635ebe745932e61986796893fcbacb0e02f8ea2e1b5806f0252b100779e686.jpg)



e


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/52fa6cb33dd4e1335a5d0634da14173cee95096ff077f50735fd42573dd9a2af.jpg)



cIn-migration rate (per 1,000)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/4bdcf3b66c85874fff5372857e28f8a3cf88e95d799716e33791f2ab1c21b55d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/ba434a0f5e34b470315430403a1a20435eb3d3f6f506cbfbc893314dca63c5cf.jpg)



Fig. 2 | Validating the MIGRATE estimates. a–c MIGRATE estimates (y-axis) are highly correlated with Census data (x-axis), including a Census populations at the Census Tract and Census Block Group (CBG) level, b movers between each pair of states and each pair of counties (excluding people who remain within the same state or county), and c state and county in-migration rates (i.e., the number of people moving into an area as a fraction of the area’s population). d–f MIGRATE estimates increase agreement with Census datasets relative to raw Infutor data for population counts, movers between states and counties, and in-migration rate, respectively. We compute root mean squared error (RMSE) between (1) MIGRATE estimates and Census data and (2) Infutor data and Census data, and report the



reduction in RMSE from using MIGRATE estimates. Bars show the mean reduction in RMSE across all data release years $\scriptstyle { n = 5 }$ for 5-year population and county-level migration datasets, $n = 9$ for 1-year state-level migration datasets); error bars plot standard deviation across years. To compare our 1-year MIGRATE estimates to 5-year ACS estimates of population and county-level migration, we average MIGRATE estimates across the same 5-year period each ACS data product covers, using only ACS data products whose time period completely overlaps with the 2010–2019 MIGRATE range. For in-migration rates, all metrics are weighted by state or county population, and points are sized by population.


# Reduction in bias relative to Infutor data

The Infutor data displays geographic and demographic biases (Fig. 3). Specifically, it overrepresents populations in the counties in the northeastern United States while underrepresenting populations in the southwest (Fig. 3a). For this analysis, as above, we multiply each Infutor matrix $\boldsymbol { E } ^ { ( t ) }$ by a scaling factor so it matches the national yearly Census population to avoid deviations due to trivial rescalings. Errors plotted are averages of county population errors across all 10 years of data from 2010 to 2019. MIGRATE estimates remove all such countylevel errors by construction. 

Errors in the Infutor data also correlate with demographics (Fig. 3b). Infutor data overrepresents White, home-owning, richer, and older populations: there is a positive Spearman correlation between the relative error in county population estimates and the population share of a county in each of these groups. These biases are consistent with biases found in previous research35, as we discuss in detail in the Supplementary Information, and could propagate into biases in downstream analyses of inequality and other topics. 

MIGRATE estimates reduce biases in Infutor data. Figure 3c compares the demographic biases of the MIGRATE estimates to the biases of the Infutor data. To quantify bias, we compare (1) the groundtruth number of people within each group (from Census data) to (2) the number of people within each group estimated from Infutor or MIGRATE, which we compute as $\textstyle \sum _ { i } p _ { i } \cdot n _ { i } ,$ where i indexes CBGs, $p _ { i }$ is the proportion of people in a CBG within a given group (e.g., the proportion of Black residents) based on Census data, and $n _ { i }$ is the number of people in a CBG in Infutor or MIGRATE. This quantifies how much Infutor and MIGRATE overcount or undercount a given demographic group, relative to ground-truth Census data. The raw Infutor data substantially overcounts rural, older, white, and home-owning populations, and undercounts younger, Black, Asian, Hispanic, renter, and below-the-poverty-line populations; the MIGRATE estimates almost entirely eliminate these biases. We present analyses for additional demographic groups, including immigrant populations, in the Supplementary Information. 

# Analysis of national migration patterns

We use MIGRATE to analyze national patterns of migration from 2010 to 2019. The spatial granularity of the data affords an opportunity to study demographic variation in migration patterns—for example, 

differences in migration between higher-income and lower-income CBGs—that may be obscured by county or state-level data, which show far less demographic variation. We hence divide CBGs into 10 (overlapping) categories—plurality white, Asian, Black, and Hispanic; urban versus rural; and bottom, second, third, and top income quartile—and stratify the migration statistics we compute by these ten categories. To determine the category of each CBG every year, we use the most recent ACS 5-year demographic estimates. For example, to classify CBGs by their plurality race group to study moves in the 2010–2011 period, we use the ACS 2006–2010 race and ethnicity estimates. 

Figure 4a plots flows between the ten categories—for example, the proportion of movers from top-income-quartile CBGs who move to top-income-quartile CBGs. This reveals substantial homophily in migration: out-movers from all ten CBG categories are more likely than movers as a whole to move to CBGs of the same category. However, the strength of this homophily varies across categories. Movers from plurality Black, Asian, and Hispanic CBGs exhibit strong homophily: they are $5 . 5 \times , 1 4 . 6 \times$ , and $4 . 3 \times$ likelier than movers as a whole to move to CBGs with the same plurality race group (Supplementary Fig. 4a reports relative rates for all groups). We also observe income homophily, particularly for movers within top and bottom quartile CBGs: movers from bottom-income-quartile CBGs are nearly twice as likely as movers as a whole to move to CBGs in the bottom income quartile $34 \%$ versus $1 8 \%$ for movers as a whole); movers from top-incomequartile CBGs are $1 . 7 \times$ likelier than movers as a whole to move to topincome-quartile CBGs $5 3 \%$ versus $32 \%$ ). In Supplementary Fig. 4b, we confirm that this homophily does not occur merely because many moves are local and demographics are spatially correlated: we also observe homophily when restricting to long-distance (out-of-county) moves. Figure 4a also shows that people are likelier to move to CBGs in higher-income quartiles: $32 \%$ of moves are to top-income-quartile CBGs, while only $1 8 \%$ are to bottom-income-quartile CBGs. This trend becomes more pronounced over the decade we study (Supplementary Fig. 5d) and is only partially explained by the fact that top-incomequartile CBGs account for a larger share of the population $( 2 9 \% )$ than bottom-income-quartile CBGs $( 2 1 \% )$ . There are also racial disparities: movers from plurality Asian CBGs are $1 . 7 \times$ likelier than movers as a whole to move to top-income-quartile CBGs; movers from plurality Black CBGs are $2 . 0 \times$ likelier than movers as a whole to move to bottomincome-quartile CBGs. Figure 4b investigates whether these racial 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/ca6deecc7f4bace28c92f22e50830a4f3ae4d8870113719292d09c355623d979.jpg)



bCounty population error in Infutor by demographics


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/0e6386e495eeff8600b2be7e1d849f7c72ef92622c01dab26f866452148ffaa7.jpg)



cBias in demographic subpopulations


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/dd9027221cbd50d26070b0afbd410ef3f2ef894e85ef87cc05fa4e43918f2ba5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/0fde57445533644bf2634e0b6e6474ad88ec1b5fda137053dd1a99912b212631.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/346aa650a65b7ec7262f62952b8f1a59022cd74db93cb701e4bae12a8b426d79.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/e889a91f3ce8ea69607d0db4385b56bc260dfe505a1ca3117447de4682f81b01.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/ac77ef5753bc8625c44f4daf9cffe00dc5a582f3a7cebab9f936d6f67580ec86.jpg)



Fig. 3 | Assessment of demographic bias in the raw Infutor data and the MIGRATE estimates. Infutor data displays biases that MIGRATE estimates greatly reduce. a Average errors in county populations in Infutor data relative to Census data; orange denotes counties where Infutor underrepresents the population, and purple denotes counties where Infutor overrepresents it. MIGRATE estimates remove all county-level errors by construction. b Spearman correlation between county demographics (x-axis) and error in Infutor estimates (y-axis). Infutor’s error



is correlated with racial, socioeconomic, and other demographic characteristics. c Comparison of demographic bias in Infutor (purple) and MIGRATE (green). MIGRATE greatly reduces biases for all demographic subgroups. Bars compare demographic subpopulations estimated from Infutor or MIGRATE to ground-truth Census data, averaged over $n = 5$ population data releases (American Community Survey 5-year estimates, 2015 through 2018). Error bars represent standard deviations across these releases.


disparities persist when controlling for origin CBG median income: in particular, plotting the share of movers moving to a higher-income CBG when controlling for the mover’s origin CBG median income decile. This reveals that the probability of moving to a higher-income CBG varies substantially by race even conditional on origin CBG income: movers from plurality Asian CBGs are more likely than movers as a whole, and movers from plurality Black CBGs less likely, to move to higher-income CBGs. For example, movers from fifth-income-decile, plurality Asian CBGs have a $71 \%$ chance of moving to a higher-income CBG; movers from fifth-income-decile, plurality Black CBGs have a $53 \%$ chance. Supplementary Fig. 5 shows that the same racial disparities occur when controlling for CBG median income percentile, as opposed to decile, or when examining the probability of moving to a top- or bottom-income-quartile CBG, as opposed to moving to a higherincome CBG. Overall, we find robust and substantial racial disparities in income of destination CBG that persist even conditional on income of origin CBG. 

Finally, Fig. 4c provides statistics on the distance of moves: $3 7 \%$ of movers move less than 5 miles; $40 \%$ from 5 to 50 miles; and $23 \%$ more than 50 miles. Figure 4c also highlights demographic variation in these statistics, revealing that movers from plurality white CBGs, rural CBGs, and higher-income CBGs are likelier to move long distances (more than 50 miles). In the Supplementary Information, we report additional migration distance statistics stratified by geographic boundary (i.e., whether movers remain within the same tract, county, or state) and show that migration distance increases over the decade we study. 

Overall, these results demonstrate that fine-grained migration data illuminates important demographic variation in homophily, upward mobility, and migration distance. Future research could stratify migration flows by additional characteristics available in Census data: for example, one might study the migration patterns of immigrants by analyzing flows from areas with larger proportions of immigrants, or larger proportions of residents with a given country of birth—both of these are datasets available at the sub-county level via ACS. However, we note that such analyses require significant heterogeneity across CBGs (or other Census areas) in the demographic trait being studied; for example, it would not be possible to reliably 

disaggregate migration patterns by gender using this method, since gender proportions remain relatively stable across Census areas. 

# Analysis of local migration patterns

In addition to using MIGRATE to analyze national migration trends, we use it to study local migration patterns: specifically, migration in response to wildfires in California. Natural disasters, including wildfires, are known drivers of human mobility3,45. There were over 3000 fire events in California from 2010 to 2019, which cumulatively affected nearly 27,504 square kilometers—approximately $6 . 5 \%$ of the California land area. In many US states, including California, wildfire risk increased from 2010 to $2 0 1 9 ^ { 4 6 }$ , and is expected to further increase due to climate change47. Researchers rely on a variety of data sources, such as administrative records and building codes, to study vacancy and movement following these disasters48 

Post-wildfire migration estimates can inform disaster response and long-term planning in multiple ways. First, migration data is crucial for policymaking in the aftermath of a wildfire. It helps guide the allocation of housing, public health, and other resources to support displaced residents while minimizing strain on housing markets in receiving areas45,49–51. Data on out-migration rates from affected areas can also inform decisions about whether and how to rebuild in fireprone regions52,53. Second, high-resolution migration data is important for future wildfire planning. As populations relocate in response to shifting wildfire risks, migration patterns can refine policymakers’ estimates of future risk and guide the regulation of housing and insurance markets54–58. While some households may move to reduce their exposure, others may become “trapped" in high-risk areas due to financial, social, or logistical barriers59–61. Identifying these communities through high-resolution migration data can help ensure that government support reaches those who need it most53,62. 

We use high-resolution fire perimeter data from the California Department of Forestry and Fire Protection63. Fire perimeters are typically much smaller than county boundaries, suggesting that analyzing fire impacts may require the granularity of the MIGRATE estimates as opposed to the relatively coarse county-to-county flows. We analyze the two most destructive fires in California from 2010 to 2019 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/27db29c10bbea8c59d5e9a1789557199c350f65a33ec1132c5e07352635ee711.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/042520ed70c04961e16e5fafdb50c223e2fb85639bb2d483ff0c2fa5c22181a1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/b039864f8402ac2cc1e3da505337918b0c01c8f428af7e7005ea4f71bee5ef9d.jpg)



Fig. 4 | National migration statistics. a Flows between ten types of Census Block Groups (CBGs)—plurality white, Asian, Black, and Hispanic; urban versus rural; and bottom, second, third, and top income quartile. Rows correspond to the origin CBG, and columns to the destination CBG; for example, the top left entry indicates that $9 0 \%$ of movers from plurality white CBGs move to plurality white CBGs. The



final two rows report the proportion of all movers moving to CBGs of each type, and the population share living in CBGs of each type. We report averages across all years. b Probability of moving to a higher-median-income CBG, conditional on income decile of origin CBG, and plurality race of origin CBG. c Distance moved stratified by CBG type.


(as quantified by the number of structures lost to the fire): the Tubbs Fire in October 2017, and the Camp Fire in November 2018 (Fig. 5). The Tubbs Fire occurred in Napa County and Sonoma County and destroyed at least 5636 residential or commercial structures64. The Camp Fire, about a year later, destroyed over 18,804 structures and damaged nearly 1000 more in the Northern California county of Butte65. The Camp Fire remains the most destructive wildfire in California as of early 2025; the Tubbs Fire has only been surpassed by the January 2025 Eaton and Palisades fires66. 

MIGRATE estimates reveal dramatic levels of out-migration for CBGs within the fire perimeters (Fig. 5a) in the year following the fires, often exceeding $50 \%$ . In contrast, CBGs outside the fire perimeters experience much lower out-migration rates. We systematically quantify these differences in Fig. 5b, which compares CBGs within the fire perimeter to three sets of less-affected CBGs outside the perimeter: (1) other CBGs neighboring the perimeter; (2) other CBGs within the 

affected counties; and (3) others within California as a whole. The outmigration rate in the year following the Camp Fire for CBGs within the fire perimeter is $46 \%$ , at least $3 . 1 \times$ that of less-affected CBG groups; the out-migration rate in the year following the Tubbs Fire for CBGs within the perimeter is $3 7 \%$ , at least $2 . 8 \times$ that of less-affected CBG groups. Our estimates of out-migration following the Camp Fire are similar to those in prior work45. (The fact that the out-migration rate is not $100 \%$ is likely due to a number of factors, including the often-preferred option of rebuilding and returning56,57 and is consistent with past studies finding a much more significant uptick in short-term vacancy than in long-term vacancy48). 

In contrast, publicly available county-level migration data obscure these dramatic out-migrations; Fig. 5c shows that rates of outmigration in affected counties remain essentially flat. This happens because the county-level data is both too spatially and too temporally coarse: the county boundaries include many CBGs unaffected by the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/7460f4361861f3a3cf1f4ba60c1b1b530213b94a7eac693901d014e4a1b36ef0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/d24020a5ae522e58e0dce4f463dded56f022555968bbbf4c691b1f269064baff.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/64fd457e8086a3e58da66ff490df6b2eb599bdde9651af45571e2abf1f1bcf33.jpg)



Fig. 5 | Migration in response to California wildfires. a Out-migration following the Camp fire (2018; top) and Tubbs fire (2017; bottom). Red boundaries plot fire perimeters; black lines plot county boundaries; Census Block Groups (CBGs) are colored by domestic out-migration rate in MIGRATE. Out-migration rates exceed $50 \%$ in many CBGs within the fire perimeters. b Out-migration rates in different groups of CBGs over time according to MIGRATE estimates. Out-migration rates in



the year after the fire are higher in CBGs within the fire perimeter (red line) than in groups of CBGs outside the fire perimeter (other lines), including those neighboring the fire perimeter, those in affected counties, or those within California. c Out-migration rates in the American Community Survey (ACS) 5-year county-tocounty data remain relatively constant over time.


fire, and the estimates are aggregated over 5 years. Further, many people displaced by the Camp and Tubbs Fires moved to other CBGs within the affected counties (and thus would not be counted as movers in the county-level data): $7 7 \%$ of movers from CBGs within the Tubbs fire perimeter and $54 \%$ of movers from CBGs within the Butte Fire perimeter remained within the affected counties. 

ACS 5-year CBG population estimates similarly obscure the dramatic levels of out-migration due to their lack of temporal granularity. In the year following the fires, MIGRATE estimates reveal population declines $2 6 0 \%$ larger in magnitude for the Tubbs fire and $40 \%$ larger in magnitude for the Camp fire than those visible at the 5-year ACS level. (For this analysis, we compare to the relative change in the ACS 5-year dataset released the year following the fire from the ACS 5-year dataset the year before: for example, for the 2018 Camp Fire, we compare the 2015–2019 ACS estimates to the 2014–2018 ACS estimates.) The ACS 5-year population estimates, unlike MIGRATE estimates, also provide no insight into the destinations of out-movers. 

In the Supplementary Information, we provide two additional analyses of local migration patterns that are impossible using countylevel data: we analyze socioeconomic variation in New York City outmover destinations, and migration patterns for residents who live in public housing provided by the New York City Housing Authority as part of the city’s affordable housing policy. Collectively, these analyses showcase how MIGRATE estimates reveal important, policy-relevant local migration patterns that traditional data sources conceal. 

# Discussion

We produce MIGRATE, a dataset of spatially and temporally finegrained migration matrices capturing annual flows between pairs of CBGs from 2010 to 2019. Our estimates are approximately 4600 times more spatially granular than publicly available county-to-county 5-year migration data, and 18 million times more spatially granular than stateto-state 1-year migration data. MIGRATE estimates correlate highly with external Census ground-truth datasets and reduce error and demographic bias relative to raw proprietary data. Our estimates are available for non-profit research use at our website. We discuss the measures taken to protect the privacy of all individuals in the dataset in the Supplementary Information. 

We use MIGRATE to analyze both local and national patterns of migration. We show that MIGRATE can reveal important local migration patterns, including dramatic increases in out-migration in response to California wildfires, that are invisible in county-level data. Nationally, we analyze demographic and temporal variation in migration patterns. We find that people tend to move to CBGs of the same 

type across dimensions of race, income, and rural/urban status—for example, movers from plurality Black, Asian, and Hispanic CBGs are $1 5 . 5 \times$ , $1 4 . 6 \times$ , and $4 . 3 \times$ likelier to move to CBGs with the same plurality race group, consistent with prior work documenting the persistence of racial segregation67–69. We document demographic and temporal variation in moving distance: movers from plurality white, rural, and higher-income CBGs are likelier to travel long distances, and moving distance increases over the decade we study, consistent with prior work25. 

An important finding from our national analysis is racial and temporal variation in upward mobility. Movers are likelier to move to top-income-quartile CBGs than bottom-income-quartile CBGs, a trend that becomes more pronounced over the decade we study. We find large racial disparities in the likelihood of moving to higher-income CBGs, or top-income-quartile CBGs, even when controlling for income of origin CBG: movers from plurality Asian CBGs are likelier to move to a higher-income CBG, and movers from plurality Black CBGs are less likely. These findings are consistent with prior work documenting racial disparities in neighborhood attainment70–73 and generalize these findings to larger and more recent samples and additional racial groups. Overall, we document demographic and temporal variation in national migration trends using a recent, large-scale, and highly granular dataset. 

While we validate MIGRATE estimates comprehensively against external data sources, practitioners making use of the data should be mindful of two limitations. First, while we use multiple ground-truth Census data sources to reduce the biases we document in the raw Infutor data, uncorrected biases likely remain. For example, while we correct biases in CBG populations, we cannot correct sub-CBG-level population biases, or biases in CBG-CBG flows; practitioners should thus not assume data at this level contain no residual biases. Data at very fine-grained levels will also be noisier than data at more aggregated levels; while we provide practitioners with fine-grained data to maximize flexibility, we recommend aggregating up to less finegrained levels if granularity is not required for analysis. Second, while we show that MIGRATE estimates are highly correlated with external Census datasets (including held-out datasets that are not used to produce our estimates), these validations have imperfections. As we further describe in the “Methods” Section, Census datasets themselves have biases; are not perfectly consistent with each other; and can possess significant margins of error, particularly at fine spatial scales. To mitigate these concerns, we conduct our harmonization process using datasets with relatively low margins of error and also prioritize fitting to Census datasets that are more precisely estimated (both 

detailed in the “Methods” Section). Another challenge in validation is the lack of a ground-truth CBG-CBG migration matrix against which to validate; publicly available Census datasets do not afford this level of granularity, and non-Census data sources (e.g., voter files) have significant biases and measure substantively different populations than the one we seek to model41,74,75. Overall, MIGRATE should be viewed as a migration data source that like all data possesses limitations but that nonetheless represents a significant improvement on widely used publicly available data sources (due to its granularity) or proprietary data sources (due to reductions in error and bias). 

We hope these improvements will enable a wide range of further migration-related analyses. As our analyses illustrate, MIGRATE reveals patterns which cannot be observed in publicly available, county-level migration data. We release MIGRATE as a resource to facilitate more precise study of many migration-related phenomena across the social, health, urban, and environmental sciences. 

# Methods

# Processing Infutor data

Infutor provides data in tabular form. Each row in the data provides data for one individual, which includes a list of known addresses, along with the date when the individual is first observed at the address (which we refer to as the effective date below), and the first and last date that the individual is observed in the data (which we refer to as the listed start date and thelisted end date, respectively). Dates are listed as month and year. For example, the data for one individual might consist of two addresses: “1 Main Street, Everytown, USA, 10000 (January 2010); 2 Cornelia Street, New York, New York, USA (December 2017)” followed by an initial date of January 2008 and an end date of December 2017. We describe the steps taken to (1) identify and clean Infutor data records within the scope of MIGRATE, (2) process the Infutor data into matrices documenting yearly flows between address pairs, and (3) map these address-level matrices to the CBG-level matrices $\boldsymbol { E } ^ { ( t ) }$ mentioned in the main text. Table 2 summarizes the raw and processed data. 

Cleaning address histories. We first create a sequence of monthly addresses for each individual in Infutor: i.e., the address at which they are living each month. This requires us to resolve any inconsistencies in the dates provided by Infutor and model uncertainty in address histories. 

We define the interval of activity during which each individual is active (observed) in the Infutor data. For each individual, we define their reconciled start date as the minimum of their first effective date, and their listed start date. We similarly define their reconciled end date as the maximum of their last effective date, and their listed end date. (For example, if the effective date list for an individual was [January 2013, January 2016], their listed start date was January 2014, and their listed end date was January 2017, their reconciled start date would be January 2013, and their reconciled end date would be January 2017.) Finally, we define their interval of activity as [reconciled start date −1 year, reconciled end date $^ { + 1 }$ year]. We use 1-year padding to reflect the annual granularity of the final dataset we produce and avoid discarding data in each yearly estimation process. This padding is also consistent with our treatment of deaths and emigrants, who will be recorded as non-movers in MIGRATE (and thus keep residence in the address they held during their last year alive in the United States). 

Having defined each individual’s interval of activity, we define the address at which the individual is located for each month in that interval. This requires us to resolve inconsistencies in the address dates provided by Infutor, which we do as follows. We discard any addresses lacking an effective date, unless they are the only address for the individual—in such cases, we consider the reconciled start date also to be the effective date for that address. We discard any postal box addresses if there is a non-postal box address for an individual with an 


Table 2 | Yearly breakdown of data summaries


<table><tr><td>Year</td><td>(1) Active records</td><td>(2) US population</td><td>(3) Processed CBG moves</td></tr><tr><td>2010</td><td>269,228,776</td><td>309,327,143</td><td>15,180,982</td></tr><tr><td>2011</td><td>260,064,921</td><td>311,583,481</td><td>11,793,806</td></tr><tr><td>2012</td><td>259,451,204</td><td>313,877,662</td><td>12,544,199</td></tr><tr><td>2013</td><td>252,933,953</td><td>316,059,947</td><td>13,829,056</td></tr><tr><td>2014</td><td>256,453,630</td><td>318,386,329</td><td>13,071,070</td></tr><tr><td>2015</td><td>241,295,957</td><td>320,738,994</td><td>14,022,249</td></tr><tr><td>2016</td><td>242,298,262</td><td>323,071,755</td><td>13,605,804</td></tr><tr><td>2017</td><td>193,479,001</td><td>325,122,128</td><td>12,684,184</td></tr><tr><td>2018</td><td>177,629,604</td><td>326,838,199</td><td>13,505,890</td></tr></table>


Population and addresses accounted by the Infutor and Census datasets during the analysis period (2010–2019). (1) Individuals have active records in a given year if the year falls within their interval of activity in the dataset. Comparison of these values to (2) the Census population shows that Infutor under-counts the population at every given year. (3) The number of moves between Census Block Groups (CBGs) parsed in our dataset. 


effective date within a year, since we assume that non-postal-box addresses are more reliable indicators of where someone lives. We then forward fill the remaining addresses so that each address spans span the period between its own effective date and the effective date of the next address. If the reconciled start date and reconciled end date for the individual differ from all the address effective dates, we fill these gaps with the first and last available addresses chronologically. If the individual has multiple addresses with the same effective date, we uniformly split the individual between the addresses by saying there was an equal probability of residence in any of them. At the end of this process, each individual is associated with a monthly probability distribution over addresses for each month in their interval of activity. 

Processing address histories into annual address-address migration matrices. When then aggregate the monthly address histories for each individual to the annual level in a way which is consistent with the ACS interview process. We will ultimately reconcile the Infutor data with ACS geographic mobility data and thus want these datasets to be consistently processed. Our goal is to simulate the sampling process of ACS, in which any individual can be asked, throughout the year, where they lived one year ago (the specific wording of the ACS interview question is, “Where did this person live 1 year ago?”). To do so, we loop over the twelve months of the year and compute how the individual would answer this survey question if they were asked in each month, yielding flows between a pairs of addresses (where the individual currently resides, and where they resided a year prior). 

We allow monthly flows to be probabilistic. For each pair of months in which the individual was certainly seen at single addresses, they would notify the ACS of that move (or stay) with probability one. However, if in one of the months there was uncertainty due to conflicting effective dates, the corresponding flow will also be uncertain; the individual might notify ACS they moved from any of the addresses they resided in the previous year during the month, or to any of the addresses they reside in the current year. We multiply probabilities of residence in each different address to obtain the probability of a flow; if the monthly address distribution remains constant in both years, we assume permanence (i.e., the individual did not move) and report that individual possibly stayed in each listed address with its own residence probability. Accounting for this uncertainty yields a list with 3-tuples of addresses and a probability (ADDR1, ADDR2, $p _ { 1  2 } )$ ) per month; weighting these tuples per month (according to the number of days in the month) yields an yearly distribution. 

We then aggregate these yearly distributions over the entire Infutor data, we create a list of expected ACS responses for the full population. That is, if multiple people reported a flow between ADDR1 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-13/d0dcabbe-b2f4-4265-888c-0971db67c759/582cd465f7e96d9c2e51f37d633de9847859684748af8210f91d91286c1def7e.jpg)



Fig. 6 | Flowchart detailing the process of mapping addresses to Census Block Groups (CBGs). We start with all Infutor addresses and classify them into five address types. We remove addresses which lie within US territories but not US states (around $0 . 2 6 \%$ of the addresses). Incomplete addresses—those that do not have a precise street address line—as well as PO boxes and rural routes are mapped probabilistically to CBGs according to their ZIP code. The vast majority $( 9 0 . 4 \% )$ of addresses are clean, complete street addresses, which are sent to the Census


and ADDR2, we add all their individual probabilities and report a single combined flow between ADDR1 and ADDR2. This represents the expected number of individuals with that flow reported to ACS. Finally, we create an yearly set of matrices $A ^ { ( t ) }$ with dimension NADDRESSES × NADDRESSES (i.e., a square matrix on the number of unique Infutor addresses) where each entry $A _ { i j } ^ { ( t ) }$ corresponds to the expected number of individuals moving between addresses i and j from year t−1 to year t. These matrices will be further aggregated based on the location of each address. e 

Processing address-address migration matrices into CBG-CBG migration matrices. We transform the address-level migration matrices $A ^ { ( t ) }$ into CBG-level matrices $\boldsymbol { E } ^ { ( t ) }$ by mapping each address either to a precise CBG where possible, or to a distribution over CBGs when only a ZIP code is available. We use the 2010 CBG boundaries for all addresses to maintain geographic consistency. Figure 6 details this process. 

Our address mapping pipeline is divided in two parts: mapping addresses via geocoding, and mapping addresses with a ZIP code-to-CBG crosswalk. We initially attempt to use one of two state-of-the art geocoders to map addresses to latitudes and longitudes: the publicly available Census Bureau geocoder43 and ESRI’s ArcGIS geocoder42. We first attempt to geocode an address via the Census geocoder; if this algorithm does not find a match, we submit the address to ArcGIS. If neither algorithm finds a match, we map the address to Census Tracts intersecting its ZIP code, weighting probabilities based on the share of residential ZIP code addresses represented by the population of each Census Tract. Addresses representing rural routes and postal boxes are directly mapped via ZIP code, as well as incomplete addresses. We use the HUD ZIP-to-tract crosswalk, with the date closest available to the last seen date of an address to account for the fact that ZIP code definitions may vary and are not nested within Census geographies76. We then distribute the probability to each CBG within the tracts, again weighting by their relative population shares. Success rates for each of these steps are high and detailed in Fig. 6. 

We use the results of this mapping process to create an auxiliary matrix $\mathcal { G }$ of dimensions $N _ { \mathrm { A D D R E S S E S } } \times N _ { \mathrm { C B G S } }$ . Each row of $\mathcal { G }$ defines a probability distribution over CBGs: an address i belongs to $_ { \mathrm { C B G } j }$ with probability $\mathcal { G } _ { i j }$ . If the address has been precisely mapped (i.e., geocoded via the Census or ArcGIS geocoders), the corresponding row of 

geocoder, which is able to map $8 1 . 2 6 \%$ of these addresses to a Census Block Group. If the Census geocoder fails to provide a match, we reattempt matching using ESRI's ArcGIS geocoder; this achieves a lower match rate of $5 8 . 2 4 \%$ , in part because the sample it is applied to is more difficult to parse. In total, we are able to map $9 9 . 2 1 \%$ of all addresses in the original Infutor data: $8 3 . 3 3 \%$ to a precise latitude and longitude, and $1 5 . 8 9 \%$ to a ZIP code. 

$\mathcal { G }$ has a single entry equal to 1, with other entries 0. We use the matrix $\mathcal { G }$ to compute the CBG-to-CBG matrix $\boldsymbol { E } ^ { ( t ) }$ via the following equation: 

$$
E ^ {(t)} = \mathcal {G} ^ {T} \cdot \left[ A ^ {(t)} - \operatorname {d i a g} \left(A ^ {(t)}\right) \right] \cdot \mathcal {G} + \operatorname {d i a g} \left[ \mathcal {G} ^ {T} \cdot \operatorname {d i a g} \left(A ^ {(t)}\right) \right] \tag {1}
$$

where the diag  operator either converts a vector into a diagonal ð Þ-matrix or extracts the diagonal of a matrix as a vector. The first term in this equation maps movers to the appropriate entries of the migration matrix: for example, a precisely mapped move from CBG i to CBG j will increment $E _ { i j } ^ { ( t ) }$ by 1, and a move from ZIP code a to ZIP code $^ { b }$ will distribute mass (summing to 1) evenly across the sub-block whose rows lie within ZIP code a and columns lie within ZIP code b. The second term in this equation maps stayers to the appropriate diagonal entries of the migration matrix: for example, a precisely mapped stayer in CBG i will increment $E _ { i i } ^ { ( t ) }$ by 1, and someone who stays within ZIP code a will distribute mass (summing to 1) evenly across the diagonal entries corresponding to CBGs within that ZIP code. 

Summarizing the processed Infutor data. Table 2 reports a yearly breakdown of relevant Infutor data statistics throughout this pipeline. The number of individuals with active records in the data represents the number of individuals with an interval of activity containing each year at the start of the data processing; the number of CBG moves corresponds to the sum of off-diagonal entries of each matrix $E ^ { ( t + 1 ) }$ and measures the “effective size” of each year’s data: how many movers we actually account for at the beginning of the estimation procedure. 

The number of active records drops in recent years because individuals are only marked as “active” when they have a recorded move; many people simply haven’t had a recent move, so they no longer appear as active even though they remain in the population. This means that Infutor is more likely to underrepresent non-movers in recent years. We employ several measures to mitigate this effect: we pad each individual’s reconciled end date by one year when constructing their active interval, and also rescale Infutor data to match rates of non-movers in Census data. 

We also verify that the decrease in Infutor active records in recent years, which is simply an expected consequence of how active records are defined, does not reflect a broader and more worrisome decrease in Infutor data quality over time. To do this, we examine variation over 


Table 3 | Validation of Infutor and MIGRATE estimates over time for selected representative metrics


<table><tr><td colspan="11">Root mean squared error (RMSE) compared to Census data</td></tr><tr><td>Quantity</td><td>Dataset</td><td>2011</td><td>2012</td><td>2013</td><td>2014</td><td>2015</td><td>2016</td><td>2017</td><td>2018</td><td>2019</td></tr><tr><td rowspan="2">CBG population</td><td>Infutor</td><td>-</td><td>-</td><td>-</td><td>507</td><td>519</td><td>531</td><td>556</td><td>581</td><td>-</td></tr><tr><td>MIGRATE</td><td>-</td><td>-</td><td>-</td><td>80</td><td>91</td><td>91</td><td>91</td><td>88</td><td>-</td></tr><tr><td rowspan="2">State-to-state movers</td><td>Infutor</td><td>3031</td><td>3684</td><td>3549</td><td>3372</td><td>3607</td><td>3315</td><td>2801</td><td>2518</td><td>2431</td></tr><tr><td>MIGRATE</td><td>365</td><td>403</td><td>400</td><td>399</td><td>434</td><td>410</td><td>343</td><td>361</td><td>383</td></tr><tr><td rowspan="2">State in-migration rate 
(per 1000)</td><td>Infutor</td><td>12.4</td><td>14.9</td><td>14.4</td><td>13.4</td><td>14.1</td><td>12.4</td><td>9.77</td><td>6.78</td><td>5.56</td></tr><tr><td>MIGRATE</td><td>1.68</td><td>1.73</td><td>1.92</td><td>1.48</td><td>1.29</td><td>1.18</td><td>1.07</td><td>1.03</td><td>1.12</td></tr><tr><td colspan="11">Pearson correlation compared to Census data</td></tr><tr><td>Quantity</td><td>Dataset</td><td>2011</td><td>2012</td><td>2013</td><td>2014</td><td>2015</td><td>2016</td><td>2017</td><td>2018</td><td>2019</td></tr><tr><td rowspan="2">CBG population</td><td>Infutor</td><td>-</td><td>-</td><td>-</td><td>0.824</td><td>0.826</td><td>0.827</td><td>0.822</td><td>0.818</td><td>-</td></tr><tr><td>MIGRATE</td><td>-</td><td>-</td><td>-</td><td>0.996</td><td>0.995</td><td>0.995</td><td>0.996</td><td>0.996</td><td>-</td></tr><tr><td rowspan="2">State-to-state movers</td><td>Infutor</td><td>0.950</td><td>0.956</td><td>0.953</td><td>0.951</td><td>0.954</td><td>0.950</td><td>0.948</td><td>0.936</td><td>0.919</td></tr><tr><td>MIGRATE</td><td>0.998</td><td>0.997</td><td>0.997</td><td>0.998</td><td>0.997</td><td>0.998</td><td>0.998</td><td>0.998</td><td>0.998</td></tr><tr><td rowspan="2">State in-migration rate</td><td>Infutor</td><td>0.895</td><td>0.916</td><td>0.911</td><td>0.884</td><td>0.885</td><td>0.896</td><td>0.871</td><td>0.863</td><td>0.809</td></tr><tr><td>MIGRATE</td><td>0.980</td><td>0.981</td><td>0.976</td><td>0.987</td><td>0.991</td><td>0.992</td><td>0.993</td><td>0.994</td><td>0.992</td></tr></table>


Year corresponds to year of the Census data release. 5-Year estimates of Census Block Group Populations are only validated against matrices from 2014 to 2019, which span the full 5-year period. Metrics for in-migration rates are weighted by population area. More details on computation of these metrics can be found on our validations section. 


time in the quality metrics discussed in the Results Section—the RMSE and correlation between Infutor and Census data for population counts, mover counts, and in-migration rates. Results are shown in Table 3. While some quality metrics improve slightly over time, and others worsen slightly, overall, we do not find consistent evidence that processed Infutor data quality decreases over time. 

Our harmonization process further improves the temporal stability of quality metrics, as can be seen in the MIGRATE rows of Table 3; the correlations between MIGRATE and Census data remain stable and high for all years. 

# Processing Census data

The U.S. Census Bureau runs multiple programs with the aim of counting and estimating demographic and socioeconomic population data. Each of these programs may release data at different spatiotemporal granularity and, even if geographies or timespans agree, the data might lack internal consistency. We use a series of steps to clean, select, and process Census datasets. We describe our procedures for reconciling Census geographies, selecting which Census datasets to use as constraints, and how we process the Census data to deal consistently with births, deaths, emigration, and immigration. 

Reconciling Census geographies. The United States is hierarchically divided into states, counties, Census Tracts, CBGs, and Census Blocks, with each level subdividing its parent geography. Geographic boundaries are not necessarily consistent over time, and ensuring that population counts reflect the same geographic area across multiple years is essential when working with longitudinal data. For example, to produce multi-year estimates, the ACS maps every reported address to the corresponding geography in the current year77. We use the 2010 Census boundaries for all addresses to maintain geographic consistency. 

While the vast majority of geographies remain intact in the interdecennial period (between Censuses, 2010–2020), there are some geography changes we must address. When geographies merge or split throughout the decade, we resolve the change by keeping only the coarser geographic boundary—i.e., the combined area which is the union of all finer-grained areas. With this approach, we can aggregate population statistics from the fine-grained areas into the coarse areas. There was one such county change in 2010–2019 affecting Bedford County in Virginia78 and a few Census Tract or CBG changes 

documented yearly in the Census Bureau website (e.g., ref. 79 for 2010–2011). When we aggregate ACS estimates that contain margins of error, we also aggregate margins of error in the L2 norm—as proposed by the ${ \bf A } { \bf C } { \bf S } ^ { 8 0 }$ . For our purposes, area renames are irrelevant. After these adjustments, MIGRATE considers a universe of 217, 740 CBGs grouped into 3142 counties. 

Selecting Census datasets. We rely primarily on three Census datasets: the ACS 5-year CBG populations; ACS 1-year state-to-state flows (from which we use raw flows as well as counts of non-movers and movers); and PEP 1-year county populations. PEP makes use of administrative records on births, deaths, and net migration to directly estimate county-level populations; the numbers reported contain the population estimate for July 1st of every year. ACS estimates, on the other hand, are a product of year-round surveying aggregated at the correspondent time scale, then released with associated margins of errors that define $90 \%$ confidence intervals80. Our dataset selection and methodology accounts for these different precision levels across the board. 

We would like to avoid matching to datasets that were too noisily estimated. To decide which ACS datasets to use in our harmonization process, we examine their level of sampling error. We quantify the sampling error by the coefficient of variation (CV): the ratio of the standard deviation to the estimate value. Table 4 reports the mean CV in ACS estimates. In general, the number of non-movers has very low CVs (mean below $2 \%$ for counties and below $1 \%$ for states); population estimates have considerably lower CVs than flow estimates; state-tostate flows have relatively high CVs (mean around $45 \%$ every year), and county-to-county flows have extremely high CVs (mean almost $90 \%$ ). We opt not to match to county-county flows due to the imprecision of raw flow estimates, although we do use them as a validation. Inmigration rates have much tighter CVs than the corresponding flows (mean around $1 5 \%$ and $5 \%$ for counties and states, respectively), and we also use those as validations. 

We perform a synthetic simulation study to verify that the datasets we do match to—state-to-state flows, and CBG population estimates—have sufficiently low sampling error to be reliably used. In particular, we resample each dataset using its reported margins of error, and compute correlations between the resampled data and the original data. To also assess mobility rates, we summarize state-tostate flows via the net migration rate. CBG populations and net state migration rates remain highly correlated with themselves after 


Table 4 | Average coefficient of variation (CV) of the non-zero estimates in each American Community Survey (ACS) dataset (in $\%$ )


<table><tr><td></td><td>2010</td><td>2011</td><td>2012</td><td>2013</td><td>2014</td><td>2015</td><td>2016</td><td>2017</td><td>2018</td><td>2019</td></tr><tr><td>CBG populations</td><td>15.76</td><td>15.67</td><td>15.42</td><td>15.33</td><td>15.17</td><td>14.94</td><td>14.95</td><td>15.08</td><td>15.21</td><td>15.59</td></tr><tr><td>Tract populations</td><td>6.41</td><td>6.24</td><td>6.03</td><td>5.95</td><td>5.79</td><td>5.60</td><td>5.50</td><td>5.54</td><td>5.56</td><td>5.66</td></tr><tr><td>County flows</td><td>-</td><td>91.03</td><td>89.97</td><td>89.47</td><td>89.41</td><td>88.06</td><td>87.33</td><td>87.30</td><td>87.01</td><td>87.01</td></tr><tr><td>County non-movers</td><td>-</td><td>1.78</td><td>1.72</td><td>1.69</td><td>1.68</td><td>1.65</td><td>1.64</td><td>1.65</td><td>1.63</td><td>1.65</td></tr><tr><td>County in-migration rate</td><td>-</td><td>16.37</td><td>15.79</td><td>15.55</td><td>15.48</td><td>15.03</td><td>14.96</td><td>15.25</td><td>15.34</td><td>15.60</td></tr><tr><td>State flows</td><td>-</td><td>47.61</td><td>45.47</td><td>45.14</td><td>44.23</td><td>45.33</td><td>45.76</td><td>46.31</td><td>45.62</td><td>47.31</td></tr><tr><td>State non-movers</td><td>-</td><td>0.41</td><td>0.39</td><td>0.39</td><td>0.38</td><td>0.40</td><td>0.40</td><td>0.38</td><td>0.37</td><td>0.39</td></tr><tr><td>State in-migration rate</td><td>-</td><td>4.95</td><td>4.68</td><td>4.79</td><td>4.65</td><td>4.70</td><td>4.76</td><td>4.72</td><td>4.85</td><td>5.02</td></tr></table>


CVs were computed as the ratio of the standard error (obtained by dividing the margin of error by 1.645) by the estimate80. We did not use flows released in 2010 (as they would report moves from 2009). 


resampling (average Pearson correlation of $\rho { = } 0 . 9 7 6$ and $\rho = 0 . 7 6 6$ respectively). 

Based on these analyses, we harmonize the entries of each yearly matrix $\boldsymbol { E } ^ { ( t ) }$ with the following data: (1) CBG populations (both from 2010 Census and 5-year ACS); (2) ACS 1-year counts of movers and nonmovers by state; (3) ACS 1-year state-to-state flows; and (4) PEP 1-year county populations. Population count data at the sub-county level was obtained via IPUMS National Historical Geographic Information System, along with socieoconomic and demographic data used for our bias analyses81; population flows and movers data was obtained directly via ${ \mathsf { A C S } } ^ { 8 2 }$ ; population count data at county level was obtained via the $\mathsf { P E P } ^ { 8 3 }$ . 

Accounting for natural population increase and international migration. Using Census datasets to constrain MIGRATE estimates requires ensuring that the Census datasets reflect the same population accounted for in the Infutor matrix $\boldsymbol { E } ^ { ( t ) }$ . Each entry $E _ { i j } ^ { ( t ) }$ represents the expected number of people alive with a residence in area i at year t−1 and with a residence in area $j$ at year $t ,$ and diagonal entries $\dot { E } _ { i i } ^ { ( t ) }$ also include individuals who resided in area i at year t−1 but died or emigrated. We process the Census datasets so they reflect the same population, by accounting for changes in the populations at year t due to births, deaths, and international migration. Specifically, when using Census populations from year t to scale entries of $\boldsymbol { E } ^ { ( t ) }$ , we want to remove from each Census area population the natural increase in population (i.e., births minus deaths) and the net international migration (i.e., immigrants minus emigrants); when using Census flows from year t−1 to year t, we want to add counts of deaths and emigrants back into the diagonal as non-movers. We adjust the Census constraints individually for every matrix $\boldsymbol { E } ^ { ( t ) }$ . 

To do this adjustment, we make use of the PEP components of population change and ACS international immigration estimates. PEP releases estimates for the number of births, deaths, net international migration, and net domestic migration yearly at both a county and a state level83. To build yearly population estimates, PEP directly adds to the previous year’s estimates the natural increase, net international migration, and net domestic migration. (PEP produces estimates that are geographically consistent: county-level estimates must aggregate precisely to state-level estimates, which must aggregate to nationallevel estimates83). Whereas PEP estimates only report the net international migration, the ACS 1-year state-to-state flows also estimate the yearly number of immigrants per state (average CV around $12 \%$ across all years), which allows us to estimate the number of emigrants per state. 

# Harmonizing Infutor and Census data

Having processed the Infutor and Census data, the final step in producing MIGRATE is to harmonize both data sources. Our 

harmonization process consists of scaling selected entries of the $\boldsymbol { E } ^ { ( t ) }$ to match non-zero entries of Census datasets. We do not scale any entries to zero because the Census zeroes are noisily estimated, and subsequent multiplicative re-scalings cannot correct erroneous zero scalings. We detail below, in order, the scalings we perform. 

Harmonizing with CBG population data. First, we harmonize $\boldsymbol { E } ^ { ( t ) }$ with CBG populations from the 5-year ACS estimates and the 2010 Census. We scale each row of each matrix $\boldsymbol { E } ^ { ( t ) }$ such that the row sums—i.e., the CBG populations—are consistent with the constraints implied by the ACS 5-year CBG populations and the 2010 Census. For example, when we take the average sum of each row across the five matrices $\{ E ^ { ( 2 0 1 1 ) }$ , E(2012), $E ^ { ( 2 0 1 3 ) }$ , $E ^ { ( 2 0 1 4 ) }$ ), E(2015)}, this should be equivalent to the population for the corresponding CBG in the 2011–2015 ACS. We also impose a boundary constraint to match the 2010 population to that reported by the Decennial Census. Imposing these constraints corresponds to solving the non-negative least squares optimization problem 

$$
\begin{array}{l l} \text {m i n i m i z e} & \| A \cdot \vec {x} - \vec {b} \| \\ \text {s . t .} & \vec {x} \geq 0 \end{array} \tag {2}
$$

where 

$$
A = \left( \begin{array}{c c c c c c c c c c} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ \frac {1}{2} & \frac {1}{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ \frac {1}{3} & \frac {1}{3} & \frac {1}{3} & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ \frac {1}{4} & \frac {1}{4} & \frac {1}{4} & \frac {1}{4} & 0 & 0 & 0 & 0 & 0 & 0 \\ \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 & 0 & 0 & 0 & 0 \\ 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 & 0 & 0 & 0 \\ 0 & 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 & 0 & 0 \\ 0 & 0 & 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} & \frac {1}{5} \end{array} \right) \quad \vec {b} = \left( \begin{array}{l} C e n s u s _ {2 0 1 0} \\ A C S _ {2 0 0 6 - 1 0} \\ A C S _ {2 0 0 7 - 1 1} \\ A C S _ {2 0 0 8 - 1 2} \\ A C S _ {2 0 0 9 - 1 3} \\ A C S _ {2 0 1 0 - 1 4} \\ A C S _ {2 0 1 1 - 1 5} \\ A C S _ {2 0 1 2 - 1 6} \\ A C S _ {2 0 1 3 - 1 7} \\ A C S _ {2 0 1 4 - 1 8} \\ A C S _ {2 0 1 5 - 1 9} \end{array} \right)
$$

for each row (i.e., each CBG). The vector $\vec { x }$ represents the estimated population for the CBG in each year from 2009 to 2019, and we impose the constraint that ${ \vec { x } } \geq 0$ to ensure populations are nonnegative. We use these estimated CBG populations to harmonize our flow matrices, by scaling each row to match the estimated CBG population in the relevant year. 

Harmonizing with yearly data on state movers and non-movers. Next, we harmonize ${ \boldsymbol E } ^ { ( t ) }$ with the counts of movers and non-movers by state from 1-year ACS data. To do this, we aggregate the off-diagonal 

entries of columns corresponding to each state and scale them to match the count of movers by state. We also aggregate the diagonal entries of columns corresponding to each state and scale them to match the count of non-movers by state. These scalings (which treat the population remaining within each CBG as equivalent to the population of non-movers) assume that very few people move to another location within the same CBG, an assumption substantiated by previous research: the median distance people moved in the US was about 10–15 miles in the 2010–19 period, which is far more than the average 2010 CBG radius (1.7 miles) and close to the 98th percentile of CBG radii84. 

Mathematically, let $S _ { k } ^ { ( t ) }$ be population of state $k$ in year t and $R _ { k } ^ { ( t ) }$ the population of state $k$ in year t which did not move in the previous year. Then we multiply every diagonal entry $E _ { x x } ^ { ( t ) }$ where CBG $x$ lies in state $r$ by 

$$
\frac {R _ {r} ^ {(t)}}{\sum_ {i} E _ {i i} ^ {(t)} \cdot \mathbb {1} \left\{\text {C B G} i \in \text {s t a t e} r \right\}} \tag {3}
$$

and then multiply every off-diagonal entry $E _ { x y } ^ { ( t ) } \left( x \neq y \right)$ where CBG $x$ lies in state $r$ and CBG y lies in state $s$ by 

$$
\frac {S _ {s} ^ {(t)} - R _ {s} ^ {(t)}}{\sum_ {i , j : i \neq j} E _ {i j} ^ {(t)} \cdot \mathbb {1} \left\{\mathrm {C B G} j \in \text {s t a t e} s \right\}} \tag {4}
$$

That is, we scale the off-diagonal entries so that the columns corresponding to each state (capturing the movers into that state) match the Census population who live in the state and did not live there the year prior $( S _ { s } ^ { ( t ) } - R _ { s } ^ { ( t ) } )$ . 

Harmonizing with yearly state-to-state flow data. We then harmonize $\boldsymbol { E } ^ { ( t ) }$ with all state-to-state flows from 1-year ACS data. To do this, we aggregate the entries of the matrix $\boldsymbol { E } ^ { ( t ) }$ according to the origindestination state pair they belong to. For example, all flows from CBGs within Delaware to CBGs within Nevada are aggregated and scaled so that they sum to the total flow from Delaware to Nevada in ACS data. We do not scale to match zero flows. 

Mathematically, let $F _ { r s } ^ { ( t ) }$ be the number of movers between states r and $s$ from year t−1 to year t. Then we multiply every entry $E _ { x y } ^ { ( t ) }$ where CBG $x$ lies in state $r$ and $\mathbf { C B G } y$ lies in state s by 

$$
\frac {F _ {r s} ^ {(t)}}{\sum_ {i , j} E _ {i j} ^ {(t)} \cdot \mathbb {1} \left\{\text {C B G} i \in \text {s t a t e} r \right\} \cdot \mathbb {1} \left\{\text {C B G} j \in \text {s t a t e} s \right\}} \tag {5}
$$

whenever $F _ { r s } ^ { ( t ) } { \neq } 0$ . We scale both people who move between states $( r \neq s )$ and who remain within the same state $( r = s )$ . 

Harmonizing with yearly county population data. Finally, we harmonize $\boldsymbol { E } ^ { ( t ) }$ with county populations from 1-year PEP data. We choose to match ${ \cal { E } } ^ { ( t ) }$ to PEP populations last due to their superior reliability. In addition, PEP estimates are used internally by the Census Bureau as controls for many other datasets85, and ensuring MIGRATE estimates agree with PEP can lead to better downstream agreement in other datasets. 

To match to PEP county populations, we apply a classical IPFbased algorithm, alternating between two steps until convergence: (1) we aggregate the entries of $\boldsymbol { E } ^ { ( t ) }$ by columns corresponding to each county and scale them to match the PEP county population at year t and (2) we aggregate the entries of ${ \cal { E } } ^ { ( t ) }$ by rows corresponding to each county and scale them to match the PEP county population at year t−1. 

More formally, let $P _ { c } ^ { ( t ) }$ be the population of a given county c at a given year t. We start with the matrix $M ^ { 0 } : = E ^ { ( t ) }$ . We then perform the following updates for iterations $n = 1 ,$ 2, …N. For odd $n$ , we scale the blocks of columns of the matrix to match the county populations in 

year t. Specifically, for each CBG $y$ lying within county $q$ , we multiply entries $M _ { x y } ^ { n - 1 }$ by the scaling factor 

$$
\frac {P _ {q} ^ {(t)}}{\sum_ {i , j} M _ {i j} ^ {n - 1} \cdot \mathbb {1} \left\{\mathrm {C B G} j \in \text {c o u n t y} q \right\}} \tag {6}
$$

For even n, we scale blocks of rows of the matrix to match the county populations in year t−1. Specifically, for each CBG $x$ lying within county $p$ , we multiply entries $M _ { x y } ^ { n - 1 }$ by the scaling factor 

$$
\frac {P _ {p} ^ {(t - 1)}}{\sum_ {i , j} M _ {i j} ^ {n - 1} \cdot \mathbb {1} \left\{\mathrm {C B G} i \in \text {c o u n t y} p \right\}} \tag {7}
$$

Our algorithm runs for 6000 iterations, which we verify is sufficient for convergence. Specifically, we track the L1 distance between the resulting matrices in two subsequent iterations. 

# Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article. 

# Data availability

MIGRATE is available upon request for non-profit research use at our website. To mitigate any privacy risks, interested researchers must agree to a data usage agreement pledging not to re-identify individuals in the data, and to adhere to privacy-protecting measures when storing data and presenting results. Manual review of their application should be completed within 10 business days, and will last for the duration of the proposed research project. 

# Code availability

Code to reproduce our research findings is available on GitHub: https://github.com/gsagostini/MIGRATE. 

# References



1. Hoffmann, R., Abel, G., Malpede, M., Muttarak, R. & Percoco, M. Drought and aridity influence internal migration worldwide. Nat. Clim. Change 14, 1245–1253 (2024). 





2. Kaysen, R. Open Houses in Los Angeles Take on an Eerie Feeling. Chap. Real Estate (The New York Times, accessed 24 January 2025, 2025). 





3. McConnell, K. et al. Rare and highly destructive wildfires drive human migration in the U.S. Nat. Commun. 15, 6631 (2024). 





4. Alexander, M., Polimis, K. & Zagheni, E. The impact of Hurricane Maria on out-migration from Puerto Rico: evidence from Facebook data. Popul. Dev. Rev. 45, 617–630 (2019). 





5. Diamond, R., Guren, A. & Tan, R. The Effect of Foreclosures on Homeowners, Tenants, and Landlords (National Bureau of Economic Research, accessed 24 January 2025, 2020) https://doi.org/10. 3386/w27358. https://www.nber.org/papers/w27358. 





6. Wilkerson, I. The Warmth of Other Suns: The Epic Story of America’s Great Migration (Knopf, New York, 2010). 





7. Niva, V. et al. World’s human migration patterns in 2000-2019 unveiled by high-resolution data. Nat. Hum. Behav. 7, 2023–2037 (2023). 





8. Boucherie, L., Maier, B.F., Lehmann, S. Decoupling geographical constraints from human mobility. Nat. Human Behav. https://doi. org/10.1038/s41562-025-02282-7 (2025). 





9. DePillis, L. For L.G.B.T.Q. People, Moving to Friendlier States Comes With a Cost. Chap. Business (The New York Times, accessed 24 January 2025, 2024). 





10. Chi, G., Abel, G. J., Johnston, D., Giraudy, E. & Bailey, M. Measuring global migration flows using online data. Proc. Natl. Acad. Sci. USA 122, 2409418122 (2025). 





11. Leasure, D. R. et al. Nowcasting daily population displacement in Ukraine through social media advertising data. Popul. Dev. Rev. 49, 231–254 (2023). 





12. Ramani, A., Alcedo, J. & Bloom, N. How working from home reshapes cities. Proc. Natl. Acad. Sci. USA 121, 2408930121 (2024). 





13. Phillips, D. C. Measuring housing stability with consumer reference data. Demography 57, 1323–1344 (2020). 





14. Gershenson, C., Jin, O., Haas, J. & Desmond, M. Fracking evictions: housing instability in a fossil fuel boomtown. Soc. Nat. Resour. 37, 347–364 (2024). 





15. Golding, S. A. & Winkler, R. L. Tracking urbanization and exurbs: migration across the rural–urban continuum, 1990–2016. Popul. Res. Policy Rev. 39, 835–859 (2020). 





16. Reia, S. M., Rao, P. S. C., Barthelemy, M. & Ukkusuri, S. V. Spatial structure of city population growth. Nat. Commun. 13, 5931 (2022). 





17. Kaysen, R. & Singer, E. Millions of Movers Reveal American Polarization in Action. Chap. The Upshot (The New York Times, accessed 24 January 2025, 2024). 





18. U.S. Census Bureau. County-to-County Migration Flows (Section, Government, accessed 15 May 2025, 2024) https://www.census. gov/topics/population/migration/guidance/county-to-countymigration-flows.html. 





19. Shu, E. G. et al. Integrating climate change induced flood risk into future population projections. Nat. Commun. 14, 7870 (2023). 





20. Craig, D.G. America’s Great Climate Migration has Begun. Here’s What You Need to Know (Columbia Magazine, accessed 05 December 2024, 2024). 





21. Tedesco, M. et al. Socio-economic, physical, housing, eviction, and risk dataset, version 2 (SEPHER 2.0), preliminary release https://doi. org/10.7927/r6yw-xw73 (2021). 





22. Rising, J., Tedesco, M., Piontek, F. & Stainforth, D. A. The missing risks of climate change. Nature 610, 643–651 (2022). 





23. Freedman, A. A. et al. Living in a block group with a higher eviction rate is associated with increased odds of preterm delivery. J. Epidemiol. Community Health 76, 398–403 (2022). 





24. Rutan, D. Q. & Desmond, M. The concentrated geography of eviction. ANNALS Am. Acad. Pol. Soc. Sci. 693, 64–81 (2021). 





25. Kerns-D’Amore, K., McKenzie, B. & Locklear, L.S. Migration in the United States: 2006 to 2019. Technical Report ACS-53, https:// www.census.gov/content/dam/Census/library/publications/2023/ acs/acs-53.pdf (American Community Survey, 2023). 





26. Infutor Data Solutions, L.L.C. Infutor Data Solutions - Batch Documentation (accessed 20 March 2025) https://batchdocs.infutor. com/ (2019). 





27. Diamond, R., McQuade, T. & Qian, F. The effects of rent control expansion on tenants, landlords, and inequality: evidence from San Francisco. Am. Econ. Rev. 109, 3365–3394 (2019). 





28. Phillips, D. C. & Sullivan, J. X. Personalizing homelessness prevention: evidence from a randomized controlled trial. J. Policy Anal. Manag. 43, 1101–1128 (2024). 





29. Boar, C. & Giannone, E. Consumption Segregation. Technical report (National Bureau of Economic Research, 2023). 





30. Bernstein, S., Diamond, R., Jiranaphawiboon, A., McQuade, T. & Pousada, B. The Contribution of High-Skilled Immigrants to Innovation in the United States. Technical report (National Bureau of Economic Research, 2022). 





31. Asquith, B., Mast, E. & Reed, D. Supply Shock Versus Demand Shock: The Local Effects of New Housing in Low-income Areas (SSRN 3507532, 2019). 





32. Qian, F. & Tan, R. The Effects of High-Skilled Firm Entry on Incumbent Residents. Working Paper, 21–039 (Stanford Institute for Economic Policy Research (SIEPR), 2021). 





33. Alexander, M., Polimis, K. & Zagheni, E. Combining social media and survey data to nowcast migrant stocks in the United States. Popul. Res. Policy Rev. 41, 1–28 (2022). 





34. Rampazzo, F., Bijak, J., Vitali, A., Weber, I. & Zagheni, E. Assessing timely migration trends through digital traces: a case study of the UK before Brexit. Int. Migr. Rev. 59, 119–140 (2025). 





35. Ramiller, A., Song, T., Parker, M. & Chapple, K. Residential mobility and big data: assessing the validity of consumer reference datasets. Cityscape 26, 227–240 (2024). 





36. Imbens, G. W. & Lancaster, T. Combining micro and macro data in microeconometric models. Rev. Econ. Stud. 61, 655–680 (1994). 





37. Guan, A., Reitsma, M., Sahoo, R., Salomon, J. & Wager, S. Data fusion for high-resolution estimation https://arxiv.org/abs/2508. 14858 (2025). 





38. Deming, W. E. & Stephan, F. F. On a least squares adjustment of a sampled frequency table when the expected marginal totals are known. Ann. Math. Stat. 11, 427–444 (1940). 





39. Chang, S., Koehler, F., Qu, Z., Leskovec, J. & Ugander, J. Inferring Dynamic Networks From Marginals With Iterative Proportional Fitting (ICML, 2024) 





40. Chang, S. et al. Mobility network models of covid-19 explain inequities and inform reopening. Nature 589, 82–87 (2021). 





41. Dong, E., Schein, A., Wang, Y. & Garg, N. Addressing discretizationinduced bias in demographic prediction. PNAS Nexus 4, 027 (2025). 





42. Hernangómez, D. arcgeocoder: geocoding with the ArcGIS REST API service. https://doi.org/10.32614/CRAN.package. arcgeocoder (2024). 





43. U.S. Census Bureau. Census Geocoder User Guide https://www2. census.gov/geo/pdfs/maps-data/data/Census_Geocoder_User_ Guide.pdf (U.S. Government Publishing Office, 2024). 





44. Hastie, T., Tibshirani, R. & Friedman, J. The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd edn. https:// hastie.su.domains/ElemStatLearn/ (Springer, New York, 2009). 





45. McConnell, K. et al. Effects of wildfire destruction on migration, consumer credit, and financial distress. FRB Cleveland Work. Paper https://www.clevelandfed.org/publications/working-paper/2021/ wp-2129-effects-of-wildfire-destruction-on-migration (2021). 





46. Modaresi Rad, A. et al. Human and infrastructure exposure to large wildfires in the United States. Nat. Sustain. 6, 1343–1351 (2023). 





47. Barbero, R., Abatzoglou, J. T., Larkin, N. K., Kolden, C. A. & Stocks, B. Climate change presents increased potential for very large fires in the contiguous United States. Int. J. Wildland Fire 24, 892 (2015). 





48. Din, A. Graphic detail: what do visualizations of administrative address data show about the camp fire. Cityscape: J. Policy Dev. Res. 24, 261–271 (2022). 





49. Rosenthal, A., Stover, E. & Haar, R. J. Health and social impacts of California wildfires and the deficiencies in current recovery resources: an exploratory qualitative study of systems-level issues. PLoS ONE 16, 0248617 (2021). 





50. Hopfer, S., Jiao, A., Li, M., Vargas, A. L. & Wu, J. Repeat wildfire and smoke experiences shared by four communities in southern California: local impacts and community needs. Environ. Res.: Health 2, 035013 (2024). 





51. Isaac, F., Toukhsati, S. R., Klein, B., Di Benedetto, M. & Kennedy, G. A. Differences in anxiety, insomnia, and trauma symptoms in wildfire survivors from Australia, Canada, and the United States of America. Int. J. Environ. Res. Public Health 21, 38 (2023). 





52. McWethy, D. B. et al. Rethinking resilience to wildfire. Nat. Sustain. 2, 797–804 (2019). 





53. Mach, K. J. & Siders, A. Reframing strategic, managed retreat for transformative climate adaptation. Science 372, 1294–1299 (2021). 





54. Boomhower, J., Fowlie, M., Gellman, J. & Plantinga, A. How are Insurance Markets Adapting to Climate Change? Risk Classification 





and Pricing in the Market for Homeowners Insurance. Technical Report w32625 (National Bureau of Economic Research, Cambridge, MA, accessed 27 February 2025, 2024) https://doi.org/10. 3386/w32625. 





55. Patrick Baylis, J. B. Mandated vs. Voluntary Adaptation to Natural Disasters: The Case of U.S. Wildfires. Technical Report w29621 (National Bureau of Economic Research, Cambridge, MA, accessed 27 February 2025, 2021) https://www.nber.org/system/files/ working_papers/w29621/w29621.pdf. 





56. Kramer, H. A., Butsic, V., Mockrin, M. H., Ramirez-Reyes, C., Alexandre, P. M. & Radeloff, V. C. Post-wildfire rebuilding and new development in California indicates minimal adaptation to fire risk. Land Use Policy 107, 105502 (2021). 





57. Alexandre, P. M., Mockrin, M. H., Stewart, S. I., Hammer, R. B. & Radeloff, V. C. Rebuilding and new housing development after wildfire. Int. J. Wildland Fire 24, 138–149 (2014). 





58. Lee, J., Costa, R. & Baker, J. W. Post-disaster housing recovery estimation: data and lessons learned from the 2017 Tubbs and 2018 Camp Fires. Int. J. Disaster Risk Reduct. 114, 104912 (2024). 





59. Bohra-Mishra, P., Oppenheimer, M. & Hsiang, S. M. Nonlinear permanent migration response to climatic variations but minimal response to disasters. Proc. Natl. Acad. Sci. USA 111, 9780–9785 (2014). 





60. Cattaneo, C. & Peri, G. The migration response to increasing temperatures. J. Dev. Econ. 122, 127–146 (2016). 





61. Benveniste, H., Huybers, P., Proctor, J. Global Climate Migration is a Story of Who, Not Just How Many https://doi.org/10.2139/ssrn. 4925994 (SSRN, 2024). 





62. Hino, M., Field, C. B. & Mach, K. J. Managed retreat as a response to natural hazard risk. Nat. Clim. Change 7, 364–370 (2017). 





63. California Department of Forestry and Fire Protection Historical Fire Perimeters (accessed 14 March 2025) https://www.fire.ca.gov/ what-we-do/fire-resource-assessment-program/fireperimeters (2024). 





64. California Department of Forestry and Fire Protection Tubbs Fire (Central LNU Complex) (accessed 14 March 2025) https://www.fire. ca.gov/incidents/2017/10/8/tubbs-fire-central-lnu-complex (2025). 





65. California Department of Forestry and Fire Protection Camp Fire (accessed 14 March 2025) https://www.fire.ca.gov/incidents/2018/ 11/8/camp-fire/ (2025). 





66. California Department of Forestry and Fire Protection Top 20 Most Destructive California Wildfires (accessed 14 March 2025) https:// www.fire.ca.gov/our-impact/statistics (2025). 





67. Quillian, L. Why is black–white residential segregation so persistent?: evidence on three theories from migration data. Soc. Sci. Res. 31, 197–229 (2002). 





68. Boustan, L. P. Racial Residential Segregation in American cities. Technical report (National Bureau of Economic Research, 2013). 





69. Dawkins, C. J. Recent evidence on the continuing causes of blackwhite residential segregation. J. Urban Aff. 26, 379–400 (2004). 





70. South, S. J., Pais, J. & Crowder, K. Metropolitan influences on migration into poor and nonpoor neighborhoods. Soc. Sci. Res. 40, 950–964 (2011). 





71. Pais, J., South, S. J. & Crowder, K. Metropolitan heterogeneity and minority neighborhood attainment: spatial assimilation or place stratification? Soc. Probl. 59, 258–281 (2012). 





72. South, S. J., Huang, Y., Spring, A. & Crowder, K. Neighborhood attainment over the adult life course. Am. Sociol. Rev. 81, 1276–1304 (2016). 





73. Quillian, L. A comparison of traditional and discrete-choice approaches to the analysis of residential mobility and locational attainment. ANNALS Am. Acad. Pol. Soc. Sci. 660, 240–260 (2015). 





74. Fraga, B. L., Holbein, J. B. & Skovron, C. Using Nationwide Voter Files to Study the Effects of Election Laws. Technical report, Working Paper (University of Virginia, 2018). 





75. Kim, S.-y.S., Fraga, B. When do voter files accurately measure turnout? How transitory voter file snapshots impact research and representation (accessed 20 March 2025). APSA Preprints https:// doi.org/10.33774/apsa-2022-qr0gd (2022). 





76. Wilson, R. & Din, A. Understanding and enhancing the U.S. department of housing and urban development’s ZIP code crosswalk files. Cityscape 20, 277–294 (2018). 





77. U.S. Census Bureau. 2024 ACS and PRCS Design and Methodology Version 4.0 https://www2.census.gov/programs-surveys/acs/ methodology/design_and_methodology/2024/acs_design_ methodology_report_2024.pdf (U.S. Government Publishing Office, 2024). 





78. U.S. Census Bureau. Substantial Changes to Counties and County Equivalent Entities: 1970-Present (Section, Government, accessed 19 March 2025, 2021) https://www.census.gov/programs-surveys/ geography/technical-documentation/county-changes.html. 





79. U.S. Census Bureau. 2011 Geography Changes (Section, Government, accessed 19 March 2025, 2021) https://www.census.gov/ programs-surveys/acs/technical-documentation/table-andgeography-changes/2011/geography-changes.html. 





80. U.S. Census Bureau. Understanding and Using American Community Survey Data: What All Data Users Need to Know (U.S. Government Publishing Office, 2020). 





81. Schroeder, J. et al. IPUMS National Historical Geographic Information System: Version 20.0. https://doi.org/10.18128/D050.V20.0 (IPUMS, Minneapolis, MN. [dataset], 2025). 





82. U.S. Census Bureau. American Community Survey Migration Flows [dataset] (accessed September 2025) https://www.census.gov/ data/developers/data-sets/acs-migration-flows.html (2024). 





83. U.S. Census Bureau. Population and Housing Unit Estimates (Section, Government, accessed 15 March 2025, 2025) https://www. census.gov/popest. 





84. Lautz, J. Long-distance Movers: Why Did They Move and How? (accessed 20 March 2025) https://www.nar.realtor/blogs/ economists-outlook/long-distance-movers-why-did-they-moveand-how (2023). 





85. U.S. Census Bureau. Population Controls for the 2021 ACS (Section, Government, accessed 15 March 2025, 2022) https://www.census. gov/programs-surveys/acs/technical-documentation/user-notes/ 2022-10.html. 



# Acknowledgements

Throughout this research, E.P. was supported by a Google Research Scholar award, an AI2050 Early Career Fellowship, NSF CAREER #2142419, a CIFAR Azrieli Global scholarship, a gift to the LinkedIn-Cornell Bowers CIS Strategic Partnership, the Survival and Flourishing Fund, Coefficient Giving, and the Zhang Family Endowed professorship. N.G. was supported by NSF CAREER IIS-2339427, and NASA, Cornell Tech Urban Tech Hub, Google, Meta, and Amazon research awards. 

# Author contributions

G.A. performed the experiments. All authors (G.A., R.Y., M.F., N.G., and E.P.) conceived and designed the experiments. M.F. provided the raw dataset. All authors contributed to the interpretation of the results, the analyses of the data, and the writing of the manuscript. 

# Competing interests

The authors declare no competing interests. 

# Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-025-68019-2. 

Correspondence and requests for materials should be addressed to Emma Pierson. 

Peer review information Nature Communications thanks Roberto Ponce-Lopez and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. A peer review file is available. 

Reprints and permissions information is available at http://www.nature.com/reprints 

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations. 

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/ licenses/by/4.0/. 

$\circledcirc$ The Author(s) 2025 