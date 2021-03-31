from ftplib import FTP
import s3fs
import pandas as pd



s3 = s3fs.S3FileSystem(anon=False)
ftp_path = "/pub/data/swdi/stormevents/csvfiles/"
s3_path = "prudhvis7245/storm_all" #S3 bucket name



ftp = FTP('ftp.ncdc.noaa.gov','anonymous')
ftp.login()
ftp.cwd(ftp_path)
data=[]
ftp.dir(data.append)



filelist = []
for line in data:
col = line.split()
filelist.append(col[8])

df=pd.DataFrame({​​​​'a':filelist}​​​​)



final_list=list(df[(df['a'].str.contains('StormEvents')) & (df['a'].str.contains('d2019'))]['a'])



for i in final_list:
ftp.retrbinary('RETR ' + i, s3.open("{​​​​}​​​​/{​​​​}​​​​".format(s3_path, i), 'wb').write)