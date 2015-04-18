%%%This file should be used to get datasets off portal
clear all;
close all;
clc;
addpath(genpath('ieeg-matlab-1.8.3'))


id{1}.train_E = 'I521_A0009_D001';
id{1}.train_dg = 'I521_A0009_D002';
id{1}.test_E = 'I521_A0009_D003';

id{2}.train_E = 'I521_A0010_D001';
id{2}.train_dg = 'I521_A0010_D002';
id{2}.test_E = 'I521_A0010_D003';

id{3}.train_E = 'I521_A0011_D001';
id{3}.train_dg = 'I521_A0011_D002';
id{3}.test_E = 'I521_A0011_D003';


session = IEEGSession(id{1}.train_E,'jaredwil','jar_ieeglogin.bin');
channels = size(session.data.channels,2);
samples = round((session.data.channels(1).get_tsdetails.getEndTime)/1000);
train_data = session.data.getvalues(1:samples,1:channels);

%Read in the rest of data in a similar fashion and save it in a .mat file
%so this does not have to be done again. 


save 'train_data.mat'