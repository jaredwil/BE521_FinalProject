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

train_data = s{1}.train_data;
train_dg   = s{1}.train_dg;
test_data  = s{1}.test_data;
test_dg    = t{1}.test_dg;
channels = size(train_data,2);    


% var = zeros(1,channels);
% av = zeros(1,channels);
% for i = 1:channels
%     var(i) = (std(train_data(:,i)))/1000;
%     av(i) = (mean(train_data(:,i)));    
% end
% 
% var_i = find(var > 2.5)
% av_i = find(abs(av) > 1.5)

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
    
    
var = std(F);


tt = (1:length(data_glove))*(1/1000);
plot(tt, train_data(:,1))





