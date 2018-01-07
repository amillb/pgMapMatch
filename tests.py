# -*- coding: utf-8 -*-

"""
Various tests
Will only succeed if likelihood function parameters in config and mapMatching_coefficients.txt are unmodified
Requires sf_streets table to be imported to postgres
Run from command line as python tests.py 
"""
import os, sys
import pgMapMatch as mm
import numpy as np

try:
    execPath = os.path.dirname(os.path.realpath(__file__))
except:
    execPath = os.getcwd()
    
class test_mapmatch():
    def  __init__(self, streetsTable, verbose=False):
        self.path = execPath+'/testdata/'
        self.testFiles = [fn for fn in os.listdir(execPath+'/testdata') if fn.endswith('.gpx')]
        self.mm = mm.mapMatcher(streetsTable,qualityModelFn=execPath+'/mapMatching_coefficients.txt')
        self.verbose=verbose

    def testall(self):
        for fn in self.testFiles:
            self.test_matchGPXTrace(fn)
        
    def test_matchGPXTrace(self,gpxFn):
            
        bestRoutes = {'testtrace_36.gpx':      [107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951],
                      'testtrace_36sparse.gpx':[107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951],
                      'testtrace_36Uturns.gpx':[107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951, 39951, 39950],
                      'testtrace_65.gpx':[101614, 125200, 125199, 82980, 125554, 89545, 125199, 82980, 82981, 125744, 89546, 89545, 125200, 101615, 101616]}
        startEndPts = {'testtrace_36.gpx':     ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.7798707692),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46183,37.77674),4326),3493)'],
                      'testtrace_36sparse.gpx':['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.7798707692),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46183,37.77674),4326),3493)'],
                      'testtrace_36Uturns.gpx':['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.7798707692),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46195,37.7785296798),4326),3493)'],
                      'testtrace_65.gpx':      ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.438457692,37.7616715385),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.438007857,37.7580557143),4326),3493)']}  
        uturnFrcs  = {'testtrace_36.gpx':[-1]*21,
                       'testtrace_36sparse.gpx':[-1]*21,
                       'testtrace_36Uturns.gpx':[-1]*20+[(0.034169679384602601, 0.34231930206380401), -1, -1],
                      'testtrace_65.gpx':[-1]*15}   
        LLs = {'testtrace_36.gpx':       [-0.5559, -3.2346, -0.2803, -24.5974, -0.8357, -21.4633],
               'testtrace_36sparse.gpx': [-0.7538, -3.2346, -0.2106, -0.5969, -1.0613, -3.3082],
               'testtrace_36Uturns.gpx': [-0.5559, -3.2352, -0.7195, -306.1156, -0.8356, -21.4633],
               'testtrace_65.gpx':[-0.5615, -3.2411, -0.3882, -16.7599, -0.9046, -21.3427]    }   
        lengths = {'testtrace_36.gpx':3152,
                   'testtrace_36sparse.gpx':3152,
                   'testtrace_36Uturns.gpx':3364,
                   'testtrace_65.gpx':2039}   
        frechet = {'testtrace_36.gpx':18.7,
                   'testtrace_36sparse.gpx':118.3,
                   'testtrace_36Uturns.gpx':20.6,
                   'testtrace_65.gpx':25.5}   
        match_scores = {'testtrace_36.gpx':0.975,
                   'testtrace_36sparse.gpx':0.320,
                   'testtrace_36Uturns.gpx':0.952,
                   'testtrace_65.gpx':0.966}   

        if not(gpxFn in bestRoutes):
            print('No test data found for %s. Skipping.' % gpxFn)
            
        print('Testing %s' % gpxFn)
        self.mm.matchGPXTrace(self.path+gpxFn)
        self.mm.getMatchedLineString()

        if self.verbose: print self.mm.bestRoute
        assert self.mm.bestRoute==bestRoutes[gpxFn]

        if self.verbose: print self.mm.startEndPts
        assert self.mm.startEndPts==startEndPts[gpxFn]

        if self.verbose: print self.mm.uturnFrcs
        assert self.mm.uturnFrcs==uturnFrcs[gpxFn]

        if self.verbose: print ', '.join([str(np.round(ll,4)) for ll in self.mm.LL])
        assert all([np.round(ll1,4)==ll2 for ll1,ll2 in zip(self.mm.LL,LLs[gpxFn])])

        if self.verbose: print self.mm.db.execfetch('SELECT ST_Length(geom) FROM (%s) t1;' % self.mm.matchedLineString)[0][0]
        assert np.round(self.mm.db.execfetch('SELECT ST_Length(geom) FROM (%s) t1;' % self.mm.matchedLineString)[0][0]) == lengths[gpxFn]

        if self.verbose: print np.round(self.mm.frechet(),1)
        assert np.round(self.mm.frechet(),1)==frechet[gpxFn]
        
        if self.verbose: print np.round(self.mm.getMatchScore(),3)
        assert np.round(self.mm.getMatchScore(),3)==match_scores[gpxFn]

if __name__=='__main__':
    streetsTable = 'sf_streets' if len(sys.argv)<2 else sys.argv[1].lower()
    verbose = True if 'verbose' in sys.argv else False
    print('Running tests with streets table %s' % streetsTable)
    test_mapmatch(streetsTable,verbose=verbose).testall()
    print('All tests completed successfully')