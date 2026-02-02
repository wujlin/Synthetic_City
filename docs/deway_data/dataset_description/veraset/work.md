Work
🏛️
This dataset is only available in institutional licenses
Overview

The Veraset Work dataset maps the unique devices to their respective work locations.

Data Information	Value
Refresh Cadence	Monthly
Historical Coverage	2023 - Present
Geographic Coverage	US
Schema


Name	Description
CAID	A hash that uniquely and anonymously identifies the device from which the location record originated.
ID_TYPE	Indicates whether device is Android or iOS
GEOHASH_5	Contains the 5-digit geohash within which this POI or home visit is in
COUNTRY	Country
ISO_COUNTRY_CODE	2 digit country code
REGION	State in the US or region globally
CITY	City
ZIPCODE	Five digit numeric code that identifies a collection of mailing addresses, US only
CENSUS_BLOCK_GROUP	Contains the census block group within which this POI or home visit is in
Key Concepts

Geographic Resolution and WFH (Work From Home) Calculation

Veraset uses clustering at a smaller area geo_hash when creating the home and work datasets. The Work From Home (WFH) attribute is then calculated by comparing these two clusters:

If a device’s work location matches its home location (i.e., both fall within the same geo_hash), the device is flagged as WFH = true.
Otherwise, it is flagged as WFH = false.