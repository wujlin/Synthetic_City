Home/Work Visits
🏛️
This dataset is only available in institutional licenses
Overview

The Home/Work visits datasets include visits to a devices work and home location. The visits are resolved to GEOHAS-5due to privacy requirements.

Data Information	Value
Refresh Cadence	Monthly
Historical Coverage	2019-2024
Geographic Coverage	US
Schema

Field Name	Description
UTC_TIMESTAMP	Timestamp in UTC in seconds since January 1, 1970.
LOCAL_TIMESTAMP	Timestamp of when the visit began (local time).
ID_TYPE	Indicates whether device is Android or iOS.
CAID	A hash that uniquely and anonymously identifies the device.
LOCATION_NAME	The name of the point of interest visited.
TOP_CATEGORY	The top-level categorization of this point of interest.
SUB_CATEGORY	A more specific categorization of this point of interest.
STREET_ADDRESS	The street number and street of the point of interest.
CITY	The city in which this point of interest is located.
STATE	The state (as postal code abbreviation) in which this point of interest is located.
ZIPCODE	Postal zip code.
GEOHASH_5	5-digit geohash for the POI or home visit (if licensed).
CENSUS_BLOCK_GROUP	Census block group within which this POI or home visit is located.
NAICS_CODE	6-digit NAICS code associated with sub_category.
BRANDS	List of brands sold by dealerships for new cars.
MINIMUM_DWELL	Minimum duration of visit (minutes).
SAFEGRAPH_PLACE_ID	Unique and consistent ID tied to this POI.
PLACEKEY	Unique and consistent ID provided by the placekey service.