fTDD: A Fusion of Time-Domain Descriptors for Improved Myoelectric Hand Control
============
fTDD is a feature extraction algorithm for the classification of ***any kind of signals***, although this was mainly developed for myoelectric, a.k.a, Electromyogram (EMG), signal feature extraction for prostheses control. The algorithm employs the ***orientation*** between a set of descriptors of muscular activities and a nonlinearly mapped version of them. It incorporates information about the Electromyogram (EMG) signal power spectrum characteristics derived from each analysis window while correlating that with the descriptors of the n'th previous windows for robust activity recognition. The proposed idea can be 
summarized in the following three steps: 

* Extract power spectrum moments from the current analysis window and its nonlinearly scaled version in time-domain through Fourier transform relations, 
* Compute the orientation between the two sets of moments, and 
* Apply data fusion on the resulting orientation features for the current and previous time windows and use the result as the final feature set. 

Refer to the paper for more details. 

![Alt text](fTDD.png?raw=true "fTDD")

As this is a matlab function (adding a python version soon), then usage is really simply, just call this function by submitting the signals matrix (denoted as variable x) as input

	feat = getfTDDfeat_Online(x,steps,winsize,wininc)

## Inputs
	x 	columns of signals
	steps 	variable denoting the number of steps away from the current window (for example a number from 3 to 25)
	winsize window size (length of x)
	wininc	spacing of the windows (winsize)

## Outputs

	feat	fused time domain features (6 features per channel)


Pay attention to features normalization
-------
For the online version I have included the two normlization methods. However, if you have your own normalization method then please comment lines 90 and 91 in "getfTDDfeat_Online.m" and add your own.


References
------
	[1] R. N. Khushaba, A. Al-Ani, A. Al-Timemy, A. Al-Jumaily, "A Fusion of Time-Domain Descriptors for Improved Myoelectric Hand Control", ISCIT2016 Conference, Greece, 2016.
 	[2] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees", IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
 	[3] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features", Neural Networks, vol. 55, pp. 42-58, 2014.

