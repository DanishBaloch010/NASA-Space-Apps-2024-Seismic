this is the sorted lengths of each file
[342378, 572263, 572383, 572386, 572386, 572390, 572395, 572395, 572395, 572395, 572395, 572395, 572395, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572399, 572407, 572407, 572407, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572411, 572415, 572415, 572415, 572415, 572415, 572415, 572415, 572415, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572418, 572420, 572423, 572423, 572423, 572423, 572423, 572423, 572423, 572423, 572423, 572427]
total 75 files

one file is missing in the dataset which is './data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1971-04-13HR00_evid00029.csv'
both the .csv and .mseed versions of this file is missing so we can ignore this. 
so we are left with 75 of total number of files from lunar dataset.
the lowest length of a file is of 342378 indexes.
and the highest is 572427
most files are 570000 indexes long (99% of files except one)

The average length is: 569340.1866666666 (this comes by taking the average of the above array)

and the sampling rate of each file is 6.625.
so it means every 6.625 approximately 7, which means one second.
notice that in each graph the size is always 800000 
here is how it comes up:
when we divide the average lengths of files with sampling rate we get 80000 seconds
569340.1866666666 / 6.625  = 85938.1413836478 seconds , which is near to 80000 , it is accurate as graphs show average ending value.

now we have to find a way that detects anomaly at each second(after each 7 indexes) to determine if the data is weird on that particular chunk(1 sec, 7 indexes) or not.
