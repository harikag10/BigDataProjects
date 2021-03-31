#Bigquery View to join Sevir and Storm data

SELECT 
sd.*,
sv.event_type as catalog_event_type, 
sd.EVENT_TYPE as storm_event_type,
sv.file_index,
sv.file_name,
sv.height_m,
sv.img_type,
sv.llcrnrlat,
sv.llcrnrlon,
sv.minute_offsets,
sv.pct_missing,
sv.proj,
sv.size_x,
sv.size_y,
sv.data_max,sv.data_min,
sv.time_utc,
sv.id as sevir_id,
sv.width_m FROM `bigdata-7245.sevir.sevir_catalog` sv left join 
`bigdata-7245.Storm.storm_details` sd on  sv.id=sd.eventid_new

#Bigquery view to join Storm fatalities and details data


SELECT 
sd.*,
sf.FAT_DAY,
sf.FAT_TIME,
sf.FAT_YEARMONTH,
sf.FATALITY_AGE,
sf.FATALITY_DATE,
sf.FATALITY_ID,
sf.FATALITY_LOCATION,
sf.FATALITY_SEX,
sf.FATALITY_TYPE
FROM `bigdata-7245.Storm.storm_fatalities` sf
left join `bigdata-7245.Storm.storm_details` sd
on  sf.EVENT_ID = sd.EVENT_ID


#Bigquery View to join Storm locaions and details data

SELECT sd.*,
sl.LAT2,
sl.LATITUDE,
sl.LOCATION,
sl.LOCATION_INDEX,
sl.LON2,
sl.LONGITUDE,
sl.RANGE,
sl.YEARMONTH
 FROM `bigdata-7245.Storm.storm_locations` sl
left join `bigdata-7245.Storm.storm_details` sd
on  sl.EVENT_ID = sd.EVENT_ID