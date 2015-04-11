%%Final Project Attempt BE521
%Jared Wilson
%10-17-14

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear MATLAB
clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%DEFINE CONSTANTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% % extract frequency-band features
freqBands =      [5 15;
                 20 25;
                 75 115;
                 125 160;
                 160 175]; % in Hz
       

%SPECTROGRAM VARIABLES
window = 100;
overlap = 50; 
fs = 1e3;
features = size(freqBands,1)+1;
%loop variables

N = 1:3;       %Number of time bins included in R matrix
%Load all the data
s{1} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub1_compData.mat');
t{1} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub1_testLabels.mat');
s{2} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub2_compData.mat');
t{2} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub2_testLabels.mat');
s{3} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub3_compData.mat');
t{3} = load('C:\Users\Jared\Desktop\FinalProject\data\be521_sub3_testLabels.mat');
%train_data -- EEG data
%size: 400000 samples x 62 channels
%train_dg  -- This is the data glove information

for sub = 1:3
    
train_data = s{sub}.train_data;
train_dg   = s{sub}.train_dg;
test_data  = s{sub}.test_data;
test_dg    = t{sub}.test_dg;
channels = size(train_data,2);    

    %%%%%%%%%%%%%%%%%%%%%%%%FEATURE EXTRACTION%%%%%%%%%%%%%%%%%%%%%%
    F = [];
    for i = 1:channels
                      
      % get spectrogram and frequency bins
      [chSpect, freqBins] = spectrogram(train_data(:,i),window,overlap,1024,fs);

      % construct freq-domain feats
      freqFeats = zeros(size(freqBands,1),size(chSpect,2));
      for j = 1:size(freqFeats)
        bandInds = freqBins >= freqBands(j,1) & freqBins <= freqBands(j,2);
        freqFeats(j,:) = mean(abs(chSpect(bandInds,:)),1);
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
    M = (samples-length(N));  %number of time bins
    %create 'R' matrix for linear regression algorithm
    r = zeros(M, features*length(N)+1);
    % temp = 1;
    for i = 1:M
        temp = F(startSAMP + (i-1) - N,:);   %temp is a temporary matrix    
        r(i,:) = [1 temp(:)'];
    %     temp = 1;
    end



    %%%%%%%%%%%%%%%%compute weight matrix%%%%%%%%%%%%
    train1 = decimate(train_dg(:,1),overlap);
    train2 = decimate(train_dg(:,2),overlap);
    train3 = decimate(train_dg(:,3),overlap);
    train4 = decimate(train_dg(:,4),overlap);
    train5 = decimate(train_dg(:,5),overlap);
    train = [train1 train2 train3 train4 train5];
    train = train(startSAMP:samples,:);
    
    
    X{sub}.weights = mldivide((r'*r),(r'*train));
%   save('sub1_weights.mat','Weights');
     
    %%%%%%%%%%%%%%%%%%%CHECK DIZ WIF SUM HAND DATA%%%%%%%%%%%%%%%%

    %because of lost sample at end and 100 ms offset from beggining the data
    %glove data must be alterted to acount for time shift
%   data_glove = train_dg(101:length(train_dg)-50,:);
    data_glove = train_dg;

    u = r*(X{sub}.weights);    %predicted dataglove data
    u = [zeros(max(N),5); u; zeros(1,5)];
    x = size(u,1);  
    ttt = (1:x)*(overlap/1000);

    tt = (1:length(data_glove))*(1/1000);

    for i = 1:5
    u_data_glove(:,i) = spline(ttt,u(:,i),tt)';
    u_data_glove(:,i) = smooth(u_data_glove(:,i),20);
    end
        
    correlation2(1) = corr(u_data_glove(:,1),data_glove(:,1));
    correlation2(2) = corr(u_data_glove(:,2),data_glove(:,2));
    correlation2(3) = corr(u_data_glove(:,3),data_glove(:,3));
    correlation2(4) = corr(u_data_glove(:,4),data_glove(:,4));
    correlation2(5) = corr(u_data_glove(:,5),data_glove(:,5));
    
    CORR{sub}.x = correlation2; 

%%%%%%%%%%%%%%%%%%%%%%%%SEE WHAT IS GOIN' ON%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%     for i = 1:5                       %Create a plot of all channels
%     figure(i)
% %     plot(tt,u_data_glove(:,i))
%     hold all;
%     plot(tt,data_glove(:,i))
%     pause(3)
%     end     

fprintf('%d\n',sub)    
end

in = [1 2 3 5];
Total_corr = [CORR{1}.x(in) CORR{2}.x(in) CORR{3}.x(in)]
Total_avg_corr = mean(Total_corr)


save('weights.mat','X')





