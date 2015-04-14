% function [predicted_dg] = make_predictions(test_ecog)
%
% Inputs: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.
% Outputs: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)
% Run time: The script has to run less than 1 hour. Any longer and the 
%   team is disqualified from the final rankings.
% Errors or inability to handle test_data or output a correctly sized 
%   predicted_dg result in automatic disqualification in the final
%   rankings.
% 
% Any errors that are encountered during Tuesday, May 5, when the TAs are 
% running each team's code will result in automatic disqualification of the 
% team. Make sure to check with the TAs on Monday to ensure the script 
% is properly finished and formatted. Remember the script and auxiliary 
% files for the script is due 1:00 AM May 5th, 2015. Any team that submits this 
% script as well as any other auxiliary files late is AUTOMATICALLY 
% DISQUALIFIED from the final ranking.
%
% The following is a sample script.

% Load Model
% Imagine this mat file has the following variables:
% winDisp, filtTPs, trainFeats (cell array), 
load team_awesome_model.mat 
%load weights for each subject and each finger
%w is a 3 x 5 cell array, containing the weights for each subject per row,
%and model for each finger per column

% Predict using linear predictor for each subject
%create cell array with one element for each subject
predicted_dg = cell(3,1);

%for each subject
for subj = 1:3 
    
    %get the testing ecog
    testset = test_ecog{subj}; 
    
    %initialize the predicted dataglove matrix
    yhat = zeros(size(testset,1),5);
    
    %for each finger
    for i = 1:5 
        
        %predict dg based on ECOG for each finger
        yhat(:,i) = testset(:,i)*w{subj,i} + intercept{subj,i};
        
    end
    predicted_dg{subj} = yhat;
end

