# -*- coding: utf-8 -*-

"""
Various tests
Will only succeed if likelihood function parameters in config and
    mapMatching_coefficients.txt are unmodified
Requires sf_streets table to be imported to postgres,
    and projected in 3493 (California State Plane Zone 3, meters)
You can download the streets data from https://mapzen.com/data/metro-extracts/metro/san-francisco_california/,
    and use osm2po or a similar tool to import to PostgreSQL
For convenience, the shapefile (based on OSM) is provided in the
    testdata folder. Unzip and import to Postgres using:
shp2pgsql -g geom_way -I sf_streets.shp <your_schema_name>.sf_streets | psql -q -d <your_db_name> -h <your_host_name> -U <your_user_name>

Run from command line as python tests.py
"""
import os
import sys
import pgMapMatch as mm
import numpy as np

try:
    execPath = os.path.dirname(os.path.realpath(__file__))
except NameError:
    execPath = os.getcwd()


class test_mapmatch():
    def __init__(self, streetsTable, verbose=False):
        self.path = execPath+'/testdata/'
        self.testFiles = [fn for fn in os.listdir(execPath+'/testdata') if fn.endswith('.gpx')]
        self.mm = mm.mapMatcher(streetsTable,
                                qualityModelFn=execPath+'/mapMatching_coefficients.txt')
        self.mm.allowFinalUturn = False
        self.verbose = verbose

    def testall(self):
        for fn in self.testFiles:
            self.test_matchGPXTrace(fn)

    def test_matchGPXTrace(self, gpxFn):

        bestRoutes = {'testtrace_36.gpx':       [107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951],
                      'testtrace_36sparse.gpx': [107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951],
                      'testtrace_36Uturns.gpx': [107059, 107060, 87737, 107056, 107057, 108203, 113936, 125564, 107060, 87796, 107055, 107056, 87725, 125408, 37189, 37188, 87797, 87796, 87795, 39950, 39951, 39951, 39950],
                      'testtrace_65.gpx': [101614, 125200, 125199, 82980, 125554, 89545, 125199, 82980, 82981, 125744, 89546, 89545, 125200, 101615, 101616]}
        startEndPts = {'testtrace_36.gpx':      ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.77987),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46183,37.77674),4326),3493)'],
                      'testtrace_36sparse.gpx': ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.77987),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46183,37.77674),4326),3493)'],
                      'testtrace_36Uturns.gpx': ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.46426,37.77987),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.46195,37.77853),4326),3493)'],
                      'testtrace_65.gpx':       ['ST_Transform(ST_SetSRID(ST_MakePoint(-122.43846,37.76167),4326),3493)', 'ST_Transform(ST_SetSRID(ST_MakePoint(-122.43801,37.75806),4326),3493)']}
        uturnFrcs = {'testtrace_36.gpx': [-1]*21,
                     'testtrace_36sparse.gpx': [-1]*21,
                     'testtrace_36Uturns.gpx': [-1]*20+[(0.0341696794, 0.3423193021), -1, -1],
                     'testtrace_65.gpx': [-1]*15}
        LLs = {'testtrace_36.gpx':       [-0.5559, -3.2346, -0.2822, -24.7070, -0.8357, -21.4633],
               'testtrace_36sparse.gpx': [-0.7538, -3.2346, -0.2106, -0.5969, -1.0613, -3.3082],
               'testtrace_36Uturns.gpx': [-0.5559, -3.2352, -0.7215, -306.2741, -0.8356, -21.4633],
               'testtrace_65.gpx': [-0.5615, -3.2393, -0.3919, -16.8696, -0.9046, -21.3427]}
        lengths = {'testtrace_36.gpx': 3152,
                   'testtrace_36sparse.gpx': 3152,
                   'testtrace_36Uturns.gpx': 3364,
                   'testtrace_65.gpx': 2039}
        frechet = {'testtrace_36.gpx': 19,
                   'testtrace_36sparse.gpx': 118,
                   'testtrace_36Uturns.gpx': 21,
                   'testtrace_65.gpx': 25}
        match_scores = {'testtrace_36.gpx': 0.975,
                   'testtrace_36sparse.gpx': 0.320,
                   'testtrace_36Uturns.gpx': 0.952,
                   'testtrace_65.gpx': 0.966}

        if not(gpxFn in bestRoutes):
            print('No test data found for %s. Skipping.' % gpxFn)

        print('Testing %s' % gpxFn)
        self.mm.matchGPXTrace(self.path+gpxFn)
        self.mm.getMatchedLineString()

        if self.verbose: print(self.mm.bestRoute)
        assert self.mm.bestRoute == bestRoutes[gpxFn]

        if self.verbose: print(self.mm.startEndPts)
        assert self.mm.startEndPts == startEndPts[gpxFn]

        if self.verbose: print(self.mm.uturnFrcs)
        self.mm.uturnFrcs = [ut if isinstance(ut, int) else (np.round(ut[0], 10), np.round(ut[1], 10)) for ut in self.mm.uturnFrcs]
        for ii, jj in zip(self.mm.uturnFrcs, uturnFrcs[gpxFn]):
            if ii == -1:  # no U-turn
                assert jj == -1
            else:
                assert all([np.round(iii, 5) == np.round(jjj, 5) for iii, jjj in zip(ii, jj)])

        if self.verbose: print(', '.join([str(np.round(ll, 4)) for ll in self.mm.LL]))
        for ll1, ll2 in zip(self.mm.LL, LLs[gpxFn]):
            assert (np.round(ll1, 4) == np.round(ll2, 4) or
                    np.round(ll1, 3) == np.round(ll2, 3) or
                    np.round(ll1, 2) == np.round(ll2, 2))

        if self.verbose: print(self.mm.db.execfetch('SELECT ST_Length(geom) FROM (%s) t1;' % self.mm.matchedLineString)[0][0])
        assert np.round(self.mm.db.execfetch('SELECT ST_Length(geom) FROM (%s) t1;' % self.mm.matchedLineString)[0][0]) == lengths[gpxFn]

        if self.verbose: print(np.round(self.mm.frechet(), 1))
        assert np.round(self.mm.frechet(), 0) == frechet[gpxFn]

        if self.verbose: print(np.round(self.mm.getMatchScore(), 3))
        # rounding can be tricky depending on precision
        assert (np.round(self.mm.getMatchScore(), 3) == match_scores[gpxFn] or 
                np.round(self.mm.getMatchScore(), 2) == np.round(match_scores[gpxFn],2))

if __name__ == '__main__':
    streetsTable = 'sf_streets' if len(sys.argv) < 2 else sys.argv[1].lower()
    verbose = True if 'verbose' in sys.argv else False
    print('Running tests with streets table %s' % streetsTable)
    test_mapmatch(streetsTable, verbose=verbose).testall()
    print('All tests completed successfully')
