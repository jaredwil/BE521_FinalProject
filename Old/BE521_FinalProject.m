%%Final Project Attempt BE521
%Jared Wilson
%10-17-14

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear MATLAB
clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%DEFINE CONSTANTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%Bandwidths of Interest [5-15Hz/20-25Hz/75-115Hz/125-160Hz/160- 175Hz] 
                            %index
bw1 = 5:15;                  %1-11
bw2  = 20:25;                %12-17
bw3 = 75:115;                %18-58
bw4 = 125:160;               %59-95
bw5 = 160:175;               %96-110
range = [bw1 bw2 bw3 bw4 bw5];
% extract frequency-band features
freqBands = [ 5 15;
                20 25;
                75 115;
                125 160;
                160 175]; % in Hz

%SPECTROGRAM VARIABLES
window = 100;
overlap = 50; 
fs = 1e3;
%loop variables
channels = 62;
features = 6;
N = 1:3;       %Number of time bins included in R matrix
%Load all the data
load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub1_compData.mat');
load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub1_testLabels.mat');
%train_data -- EEG data
%size: 400000 samples x 62 channels
%train_dg  -- This is the data glove information
      
%%%%%%%%%%%%%%%%%%%%%%%%FEATURE EXTRACTION%%%%%%%%%%%%%%%%%%%%%%
F = [];
for i = 1:channels
% % % % % % [S,F,T,P]=spectrogram(train_data(:,i),window,overlap,range,fs)
% % % % % % % spectrogram(train_data(:,1),window,overlap,range,fs);
% % % % % % %compute the F.T. for small windows (100 ms) and find the frequencies
% % % % % % %contained in those "windows" using the PSD (power spectral desnsity) P.
% % % % % % 
% % % % % % %These create maxices for each channel in each time "window" of the average
% % % % % % %PSD for each bandwidth (bw) of interest. [see above for ranges]
% % % % % % for k = 1:length(T)
% % % % % % average_bw1(k) = mean(P(1:11,k));
% % % % % % average_bw2(k) = mean(P(12:17,k));
% % % % % % average_bw3(k) = mean(P(18:58,k));
% % % % % % average_bw4(k) = mean(P(59:95,k));
% % % % % % average_bw5(k) = mean(P(96:110,k));
% % % % % % end
% % % % % % 
% % % % % % %creat a 3D matrix easy to acces values each page represents a different
% % % % % % %channel        [indexed BWs(bandwidth, timebin, channel)]
% % % % % % BWs(:,:,i) = [average_bw1; average_bw2; average_bw3; average_bw4; average_bw5];
% % % % % % for j = 1:(length(train_data)/50)-1;
% % % % % % ATDV(i,j) = mean(train_data(((j-1)*50)+1:((j-1)*50)+100,i));
% % % % % % end
% % %            BW(:,:,i) = freqFeats(:,:);  


  % get spectrogram and frequency bins
  %[chSpect, freqBins] = spectrogram(chTrace,winLen*fs/1000, ...
  %  (winLen-winDisp)*fs/1000,freqBands(1):freqBands(end),fs);
  [chSpect, freqBins] = spectrogram(train_data(:,i),window,overlap,1024,fs);
  
  % construct freq-domain feats
  freqFeats = zeros(size(freqBands,1),size(chSpect,2));
  for j = 1:size(freqFeats)
    bandInds = freqBins >= freqBands(j,1) & freqBins <= freqBands(j,2);
    freqFeats(j,:) = mean(abs(chSpect(bandInds,:)),1);
    %freqFeats(i,:) = log(sum(abs(chSpect(bandInds,:)),1)+1);
  end



C = conv(train_data(:,i),ones(1,window)/window,'valid');
timeavg_bin = C(1:overlap:end)';

F = [F freqFeats' timeavg_bin'];

fprintf('%d',i)
end

%%%%%%%%%%%%%%%%%%%%%%CREATE R MATRIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
samples = size(F,1);
features = size(F,2);
startSAMP = max(N)+1;
M = (samples-length(N)+1);  %number of time bins
%create 'R' matrix for linear regression algorithm
r = zeros(M, features*length(N)+1);
% temp = 1;
for i = 1:M
    temp = F(startSAMP + (i-1) - N,:);   %temp is a temporary matrix    
    r(i,:) = [1 temp(:)'];
%     temp = 1;
end

%%%%%%%%%%%%%%%%compute weight matrix%%%%%%%%%%%%
train1 = decimate(train_dg(:,1),50);
train2 = decimate(train_dg(:,2),50);
train3 = decimate(train_dg(:,3),50);
train4 = decimate(train_dg(:,4),50);
train5 = decimate(train_dg(:,5),50);
train = [train1 train2 train3 train4 train5];
train = train(length(N):length(train)-1,:);

Weights = mldivide((r'*r),(r'*train));
save('sub1_weights.mat','Weights');

%%%%%%%%%%%%%%%%%%%CHECK DIZ WIF SUM HAND DATA%%%%%%%%%%%%%%%%

%because of lost sample at end and 100 ms offset from beggining the data
%glove data must be alterted to acount for time shift
data_glove = train_dg(101:length(train_dg)-50,:);

u = r*(Weights);    %predicted dataglove data
x = length(train);  
ttt = (1:x)*(50/1000);

tt = (1:length(data_glove))*(1/1000);

for i = 1:5
u_data_glove(:,i) = spline(ttt,u(:,i),tt)';
end

% correlation(1) = corr(u(:,1),train(:,1)); 
% correlation(2) = corr(u(:,2),train(:,2));
% correlation(3) = corr(u(:,3),train(:,3));
% correlation(4) = corr(u(:,4),train(:,4));
% correlation(5) = corr(u(:,5),train(:,5))

correlation2(1) = corr(u_data_glove(:,1),data_glove(:,1));
correlation2(2) = corr(u_data_glove(:,2),data_glove(:,2));
correlation2(3) = corr(u_data_glove(:,3),data_glove(:,3));
correlation2(4) = corr(u_data_glove(:,4),data_glove(:,4));
correlation2(5) = corr(u_data_glove(:,5),data_glove(:,5))


for i = 1:5
figure(i)
plot(tt,u_data_glove(:,i))
hold all;
plot(tt,data_glove(:,i))
pause(3)
end




