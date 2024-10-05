this folder contains a single and code to detect anomaly(earthquake) on that signal. 

the process is totally automatic.

run this code on 'xa.s12.00.mhz.1970-06-15HR00_evid00008.csv'

the path is mentioned in the code, change it as you want.

but test it on the same file. because it is an easy signal.

the idea is simple, we feed the whole signal to the model and the model tells us which data points are anamolic.  

then we collect the indexes of those anamolic datapoints and fetch all the timestamps of those indexes from the original csv. 

when we have all the timestamps then we take the average of timestamps to get a predicted earthquake timestamp.
