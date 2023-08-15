%% Dummy test to verify that CI is working for Matlab.
%
classdef TestDummy < matlab.unittest.TestCase
%% Setup and Teardown Methods
%
%  Add and remove path to utils functions to be tested.
%
    methods (TestClassSetup)
        function addPath(testCase)
            addpath('./') %addpath(genpath('../../lib/utils'));
        end
    end
    methods (TestClassTeardown)
        function removePath(testCase)
            rmpath('./') %rmpath(genpath('../../lib/utils'));
        end
    end
    
%% Tests
%
    methods (Test)
        function testCeilOdd1(testCase)
            actSolution = ceil(56.1);
            expSolution = 57;
            testCase.verifyEqual(actSolution, expSolution);  
        end

    end    
end