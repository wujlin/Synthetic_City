Visits
🏛️
This dataset is only available in institutional licenses


Overview


Visits is Veraset's point of interest (POI) product, providing reliable, device-level POI data from over 4 million high-interest locations across the US. They merge raw GPS signals with precise polygon places to help you understand when certain devices visited which relevant POI. The dataset includes:

2.5B visits from 300 million pseudonymous devices
4 million U.S. points of interest
~6% of the U.S. population represented
Data Information	Value
Refresh Cadence	Monthly
Historical Coverage	2019-Present
Geographic Coverage	US
Schema


Name	Description
UTC_TIMESTAMP	Timestamp in UTC in seconds since January 1, 1970
LOCAL_TIMESTAMP	Timestamp of when the visit began (local time)
ID_TYPE	Indicates whether device is Android or iOS
CAID	A hash that uniquely and anonymously identifies the device
LOCATION_NAME	The name of the point of interest visited
TOP_CATEGORY	The top-level categorization of this point of interest
SUB_CATEGORY	A more specific categorization of this point of interest
STREET_ADDRESS	The street number and street of the point of interest
CITY	The city in which this point of interest is located
STATE	The state (as postal code abbreviation) in which this point of interest is located
ZIPCODE	ZIPCODE
GEOHASH_5	Contains the 5-digit geohash within which this POI or home visit is in
CENSUS_BLOCK_GROUP	Contains the census block group within which this POI or home visit is in
NAICS_CODE	The 6-digit NAICS code associated with sub_category
BRANDS	For dealerships selling new cars, a list of brands sold by that dealership
MINIMUM_DWELL	Minimum duration of visit (minutes), calculated by a device pinging multiple times within a POI
SAFEGRAPH_PLACE_ID	Unique and consistent ID that is tied to this POI. If this is a home visit, then = "home"
PLACEKEY	Unique and consistent ID that is tied to this POI as provided by the placekey service
Key Concepts


Visit Attribution


From 2019 to June 1, 2024, a visit was counted when a device received two pings that were 4 minutes apart (to create clustering for visit attribution). Starting on June 1, 2024, visits are counted with a single ping and with no dwell time requirements (this allows for things like grab-and-go orders).

Categorization


Veraset uses SafeGraph as it's POI data provider and as such uses the same categorization methodology. THis includes NAICS code, category, sub-category, and placekey. For more information about SafeGraph, visit the SafeGraph Places doc.

Coverage


Veraset users SafeGraph as it's POI provider. While POI's without visits will not appear in the dataset, the SafeGraph Summary Statistics is a good estimate as to what may be included in the Veraset Visits data.

Broadly, Veraset has consistent coverage across 5,886,791 census block groups (CBGs), representing approximately ~70% of all block groups nationwide—including those without recorded populations. When looking only at CBGs with a recorded population, their coverage rises to roughly 90%.


