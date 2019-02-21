%
% getfTDDfeat: A Fusion of Time-Domain Descriptors.
%
% feat = getfTDDfeat(x,winsize,wininc,datawin,dispstatus)
%
% Author Rami Khushaba 
%
% This function computes the fused time-domain features of the signals in x,
% x could be made of many columns (or just one), each representing a channel/sensor. 
% For example if you get 5 sec of data from 8 channels/sensors at 1000 Hz 
% then x should be 5000 x 8. A windowing scheme is used here to extract features 
%
% The signals in x are divided into multiple windows of size
% winsize and the windows are space wininc apart.
%
%
% Inputs
%    x: 		columns of signals
%    steps:     variable denoting the number of steps away from the current window (for example a number from 3 to 25, read the paper)
%    winsize:	window size (length of x)
%    wininc:	spacing of the windows (winsize)
%    datawin:   window for data (e.g. Hamming, default rectangular)
%               must have dimensions of (winsize,1)
%    dispstatus:zero for no waitbar (default)
%
% Outputs
%    feat:      fused time domain features (6 features per channel)
%             
%
% Modifications
% 23/06/2004   template imported from http://www.sce.carleton.ca/faculty/chan/index.php?page=matlab
% 17/11/2013   RK: Spectral moments first created.
% 15/01/2016   RK: created fTDD method
% 21/02/2019   RK: moving this to GitHub

% References
% [1] R. N. Khushaba, A. Al-Ani, A. Al-Timemy, A. Al-Jumaily, "A Fusion of Time-Domain Descriptors for Improved Myoelectric Hand Control", ISCIT2016 Conference, Greece, 2016.
% pdf here: https://pdfs.semanticscholar.org/4730/98af39c66b0a4b541860a1f4617c036d8249.pdf
% [2] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees", 
%     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
% [3] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features", 
%     Neural Networks, vol. 55, pp. 42-58, 2014. 

function feat = getfTDDfeat(x,steps,winsize,wininc)


datasize = size(x,1);
Nsignals = size(x,2);
numwin = floor((datasize - winsize)/wininc)+1;

% allocate memory
feat = zeros(numwin,Nsignals*6);

st = 1+steps*wininc;
en = winsize+steps*wininc;
for i = 1:numwin-steps
   
   curwin = x(st:en,:);
   
   % steps1: Extract features from original signal and a nonlinear version
   % of the previous window
   ebp = KSM1(x((st-steps*wininc):(en-steps*wininc),:));
   efp = KSM1(log(x((st-steps*wininc):(en-steps*wininc),:).^2+eps).^2);
   
   % steps2: Correlation analysis
   num = -2.*ebp.*efp;
   den = efp.*efp+ebp.*ebp;
   
   % steps1: Extract features from original signal and a nonlinear version of it
   ebp = KSM1(curwin);
   efp = KSM1(log(curwin.^2+eps).^2);
   
   % steps2: Correlation analysis
   num2 = -2.*ebp.*efp;
   den2 = efp.*efp+ebp.*ebp;
   
   % feature extraction goes here
   feat(i,:) = [ (num./den) .* (num2./den2)];
   st = st + wininc;
   en = en + wininc;
end
feat = feat(1:end-steps,:);


function Feat = KSM1(S)
% Time-domain power spectral moments (TD-PSD)
% Using Fourier relations between time domain and frequency domain to
% extract power spectral moments dircetly from time domain.
%
% Modifications
% 17/11/2013  RK: Spectral moments first created.
% 
% References
% [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees", 
%     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
% [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features", 
%     Neural Networks, vol. 55, pp. 42-58, 2014. 


%% Get the size of the input signal
[samples,channels]=size(S);

% if channels>samples
%     S  = S';
%     [samples,channels]=size(S);
% end

%% Root squared zero order moment normalized
m0     = sqrt(sum(S.^2));
m0     = m0.^.1/.1;

% Prepare derivatives for higher order moments
d1     = diff([zeros(1,channels);diff(S)],1,1);
d2     = diff([zeros(1,channels);diff(d1)],1,1);

% Root squared 2nd and 4th order moments normalized
m2     = sqrt(sum(d1.^2)./(samples-1));
m2     = m2.^.1/.1;

m4     = sqrt(sum(d2.^2)./(samples-1));
m4     = m4.^.1/.1;

%% Sparseness
sparsi = (sqrt(abs((m0-m2).*(m0-m4))).\m0);

%% Irregularity Factor
IRF    = m2./sqrt(m0.*m4);

%% Waveform length ratio
WLR    = sum(abs(d1))./sum(abs(d2));

%% All features together
Feat   = log(abs([(m0) (m0-m2) (m0-m4) sparsi IRF WLR]));

