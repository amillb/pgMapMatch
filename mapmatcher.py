# -*- coding: utf-8 -*-

"""
Map-matching of GPS traces using PostgreSQL and pgrouting
See Millard-Ball, Adam; Hampshire, Robert and Weinberger, Rachel,
    "Map-matching poor-quality GPS data in urban environments:
     The pgMapMatch package."
"""

import time
import datetime
import random
import math
import os
import pandas as pd
import numpy as np
from scipy import stats, sparse

# pgMapMatch tools
from . import tools as mmt

try:
    from .config import *
except ModuleNotFoundError:
    raise Warning('config.py not found. Using config_template.py instead')
    from .config_template import *


class traceCleaner():
    def __init__(self, traceTable, idName, geomNameOld, geomNameNew=None,
                 maxSpeed=maxSpeed, logFn=None):
        """Remove errant points from a GPS trace using a speed threshold

        traceTable: name of the postgres table that stores the GPS traces
        idName: name of a unique ID column
        geomNameOld: name of the geometry column to be cleaned
        geomNameNew: name of the geometry column to populate.
                     Anything in this column will be overwritten.
                     If None, is geomNameOld
        logFn: optional filename to create a log file
        """

        self.db = mmt.dbConnection(pgLogin=pgInfo)
        cmd = 'SELECT COUNT(%s)=COUNT(DISTINCT %s) FROM %s' % (idName, idName, traceTable)
        if self.db.execfetch(cmd)[0][0] is False:
            raise Exception('idName column %s must be unique' % idName)

        self.traceTable = traceTable
        self.idName = idName
        self.geomNameOld = geomNameOld
        self.geomNameNew = geomNameOld if geomNameNew is None else geomNameNew
        cmd = 'SELECT ST_Srid(%s) FROM %s LIMIT 1;' % (self.geomNameOld, self.traceTable)
        self.srid = self.db.execfetch(cmd)[0][0]
        self.maxSpeed = maxSpeed
        self.ptsToDrop = None
        self.logFn = logFn

        if logFn is not None:
            with open(self.logFn, 'a') as f:
                print('Logging to %s' % self.logFn)
                currentTime = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
                f.write('\n____________NEW TRACE CLEANER table %s AT %s____________\n' % (traceTable,  currentTime))

    def getTracesDf(self, traceId=None, numLags=5):
        """Gets a pandas dataframe of traces.
        Optionally limits to the ids specified in the traceId list"""

        selectClause1 = '''SELECT traceid, ptid,\n'''
        lagClause1    = ',\n'.join(['ST_Distance(geom, l%(lag)s)/NULLIF(ST_M(l%(lag)s)-ST_M(geom),0)/1000*60*60 AS speed%(lag)s' % {'lag': lag} for lag in range(1, numLags+1)])
        selectClause2 = '''\nFROM (SELECT traceid, (dp).path[1] AS ptid, (dp).geom AS geom,\n'''
        lagClause2    = ',\n'.join(['lag((dp).geom,%(lag)s) OVER (PARTITION BY traceid ORDER BY (dp).path[1] DESC) AS l%(lag)s' % {'lag': lag} for lag in range(1, numLags+1)])
        fromClause = '''\nFROM (SELECT %(idName)s AS traceid, ST_DumpPoints(%(geomNameOld)s) AS dp FROM %(traceTable)s
                      ''' % {'idName': self.idName, 'geomNameOld': self.geomNameOld, 'traceTable': self.traceTable}
        whereClause = '' if traceId is None else ' WHERE %s = %s' % (self.idName, traceId)
        cmd = selectClause1+lagClause1+selectClause2+lagClause2+fromClause+whereClause+') AS t1) AS t2;'
        return self.db.execfetchDf(cmd)

    def fetchPtsToDrop(self, traceId=None, numLags=5):
        """Returns a list of tuples of traceid, ptnum
        Max speed must be in the units of your projection
            (normally meters), per second"""

        df = self.getTracesDf(traceId=traceId, numLags=numLags)

        df['dropPt'] = False  # will we drop that point?
        ptsToDrop = []  # list of (traceid, ptid) tuples

        for gSize in range(1, numLags+1):  # loop over groups of 1...5 points
            df['minPt'] = df[df.dropPt == False].groupby('traceid').ptid.min()
            df['maxPt'] = df[df.dropPt == False].groupby('traceid').ptid.max()
            df['lagSpeed'] = df.speed1.shift(gSize*-1)
            df.loc[(df.shift(gSize*-1).traceid != df.traceid), 'lagSpeed'] = np.nan  # where traceid doesn't match
            dropMask = ((df.lagSpeed > self.maxSpeed) & (df.speed1 > self.maxSpeed)) | ((df.ptid == df.minPt+gSize-1) & (df.speed1 > self.maxSpeed)) | ((df.lagSpeed > self.maxSpeed) & (df.ptid == df.maxPt))

            for lag in range(0, gSize):  # make sure we get 'intermediate' points in that group too
                df.loc[dropMask.shift(lag) & (df.shift(lag).traceid == df.traceid), 'dropPt'] = True
                df.loc[dropMask.shift(lag) & (df.shift(lag).traceid == df.traceid), ['speed'+str(s) for s in range(1, 6)]] = np.nan

            # Don't drop last point in a trace
            df.loc[(df.dropPt) & (df.ptid == df.maxPt), 'dropPt'] = False

            # Move the speeds forward one lag
            for lag in range(0, 5):
                oldCols = ['speed'+str(s) for s in range(lag+1, 6-gSize)]
                newCols = ['speed'+str(s) for s in range(lag+1+gSize, 6)]
                mask = dropMask.shift(gSize+lag) & (df.shift(gSize+lag).traceid == df.traceid)
                df.loc[mask, oldCols] = df.loc[mask, newCols].values
                df.loc[mask, ['speed'+str(s) for s in range(max(lag+1, 6-gSize), 6)]] = np.nan

            ptsToDrop += list(df[df.dropPt].set_index('traceid').ptid.iteritems())
            df = df[df.dropPt == False]

        # Finally, select any 'singletons' where there is one errant point
        # that's not picked up by the lagged values
        ptsToDrop += list(df[df.speed1 > self.maxSpeed].set_index('traceid').ptid.iteritems())

        self.ptsToDrop = ptsToDrop

    def dropPoints(self):
        """Update the database"""
        traceIds = list(set([pt[0] for pt in self.ptsToDrop]))
        if self.geomNameNew not in self.db.list_columns_in_table(self.traceTable):
            self.db.execute("SELECT AddGeometryColumn('%s','%s',%s,'LineStringM',3);" % (self.traceTable, self.geomNameNew, self.srid))

        for traceId in traceIds:
            ptIds = [pt[1] for pt in self.ptsToDrop if pt[0] == traceId]
            dropIds = str(tuple(ptIds)) if len(ptIds) > 1 else '('+str(ptIds[0])+')'
            cmdDict = {'traceTable': self.traceTable, 'geomNameOld': self.geomNameOld, 'geomNameNew': self.geomNameNew,
                       'idName': self.idName, 'id': traceId, 'dropIds': dropIds}
            cmd = '''UPDATE %(traceTable)s SET %(geomNameNew)s = newline FROM
                        (SELECT ST_MakeLine((pts).geom ORDER BY (pts).path[1]) AS newline FROM
                            (SELECT ST_DumpPoints(%(geomNameOld)s) pts FROM %(traceTable)s
                            WHERE %(idName)s=%(id)s) t1
                        WHERE (pts).path[1] NOT IN %(dropIds)s) t2
                     WHERE %(idName)s=%(id)s;''' % cmdDict
            self.db.execute(cmd)

    def reportDroppedPoints(self):
        nIds = len(set([pt[0] for pt in self.ptsToDrop]))
        print('%s pts from %s traces in the drop list' % (len(self.ptsToDrop), nIds))
        if self.logFn is not None:
            with open(self.logFn, 'a') as f:
                f.write('%s pts to be dropped from the following %s trace ids:\n' % (len(self.ptsToDrop), nIds))
                f.write(', '.join([str(pp) for pp in sorted(list(set([pt[0] for pt in self.ptsToDrop])))]))

    def fetchAndDrop(self):
        self.fetchPtsToDrop()
        self.reportDroppedPoints()
        self.dropPoints()


class mapMatcher():
    def __init__(self, streetsTable, traceTable=None, idName=None, geomName=None, newGeomName='matched_line', cleanedGeomName=None, qualityModelFn='mapmatching_coefficients.txt', db=None, verbose=True):
        """Creates a mapMatcher object that loads the streets (edges) table
        and can then be called to match a trace

        streetsTable:    PostgreSQL table of the street edges. Column names are specified in the config file
        traceTable:      PostgreSQL table of the GPS traces. None if a GPX file is being used
        idName:          Column name in traceTable of the unique ID. None if a GPX file is being used
        geomName:        Column name in traceTable of the GPS geometry for traceTable. Use a projected coordinate system. None if a GPX file is being used
        newGeomName:     Column name in traceTable of the geometry column to write the matched line.
        cleanedGeomName: Column name in traceTable of the geometry column to write the cleaned geometry (i.e., geomName minus dropped points)
        qualityModelFn:  File name with the logit model coefficients (needed to write the match score)
        db:              postgres database connection object (optional - will be created using the config information if left as None
        verbose:         Echo postgres queries to stdout
        """

        if db is None:
            db = mmt.dbConnection(pgLogin=pgInfo, verbose=verbose)
        self.db = db
        self.pgr_version = self.db.execfetch('SELECT pgr_version();')[0][0].strip('()').split(',')[0]
        if float(self.pgr_version.split('.')[0]) < 2 or float(self.pgr_version.split('.')[0]) == 2 and float(self.pgr_version.split('.')[1]) < 3:
            raise Exception('pgRouting v2.3.0 or greater required to use pgr_dijkstraCostMatrix')
        self.postgis_version = self.db.execfetch('SELECT PostGIS_lib_version();')[0][0]
        self.streetsTable = streetsTable
        self.traceTable = traceTable
        self.verbose = verbose
        self.idName = idName  #
        self.geomName = geomName  # for trace
        self.newGeomName = newGeomName
        self.cleanedGeomName = cleanedGeomName  # optional - will rewrite geomName without any dropped points
        self.clearCurrentRoutes()

        self.streets_srid = str(self.db.execfetch('SELECT ST_SRID(%s) FROM %s LIMIT 1;' % (streetGeomCol, streetsTable))[0][0])
        if self.streets_srid == '4326':
            raise Exception('Streets SRID is 4326. You should use a projected coordinate system. Project your streets geometry column and try again.')
        if traceTable is not None:
            self.trace_srid = str(self.db.execfetch('SELECT ST_Srid(%s) FROM %s LIMIT 1' % (self.geomName, self.traceTable))[0][0])
            if self.trace_srid != self.streets_srid:
                raise Exception('Trace SRID %s and streets SRID %s do not match' % (self.trace_srid, self.streets_srid))

        self.qualityModelFn = qualityModelFn
        self.qualityModelCoeffs = None

        # dictionary of common substitutions into SQL commands
        self.cmdDict = {'streetsTable': streetsTable, 'streetIdCol': streetIdCol, 'streetGeomCol': streetGeomCol,
              'source': startNodeCol, 'target': endNodeCol,
              'cost': travelCostCol, 'reverse_cost': travelCostReverseCol,
              'km': streetLengthCol,
              'kmh': speedLimitCol, 'fwayCols': '' if fwayCols.strip() == '' else ','+fwayCols,
              'traceTable': traceTable, 'idName': idName,
              'geomName': geomName, 'newGeomName': newGeomName, 'cleanedGeomName': cleanedGeomName,
              'gpsError': gpsError, 'gpsError_fway': gpsError_fway}
        fwayColList = [cc.strip() for cc in fwayCols.split(',')]

        # Check geometry types of streets
        result = self.db.execfetch('''SELECT DISTINCT ST_GeometryType(%(streetGeomCol)s) FROM %(streetsTable)s;''' % self.cmdDict)
        if any([rr[0] == 'ST_MultiLineString' for rr in result]):
            raise Exception('Streets geometry column %s must be ST_LineString, not ST_MultiLineString. Use ST_LineMerge() to convert your table.')
        if not all([rr[0] == 'ST_LineString' for rr in result]):
            raise Exception('Streets geometry column %s must be ST_LineString, not %s.' % (streetGeomCol, result[0][0]))

        # Check length of table
        nsts = self.db.execfetch('''SELECT COUNT(*) FROM %(streetsTable)s;''' % self.cmdDict)
        if nsts[0][0] > 1000000:
            print('Streets table has %d rows, which may impair pgrouting performance. Consider clipping the table to your area of interest.' % nsts[0][0])

        # Check geometry types of trace table
        if traceTable is not None:
            result = self.db.execfetch('''SELECT ST_M(geom) is Not Null, test1, test2 FROM
                                        (SELECT ST_GeometryType(%(geomName)s)='ST_LineString' test1,
                                                Not ST_IsCollection(%(geomName)s) test2, (ST_DumpPoints(%(geomName)s)).geom
                                                FROM %(traceTable)s WHERE %(geomName)s is Not Null LIMIT 1) t1;''' % self.cmdDict)
            if not all(result[0]):
                raise Exception('GPS trace geometry column %s must be LineString with M coordinate' % geomName)

        cmd = '''SELECT %(streetIdCol)s, %(source)s::integer, %(target)s::integer, %(cost)s::real, %(reverse_cost)s::real, %(km)s::real,
               %(kmh)s::real %(fwayCols)s FROM %(streetsTable)s''' % self.cmdDict
        edgesDf = pd.DataFrame(self.db.execfetch(cmd), columns=['edge', 'source', 'target', 'cost', 'reverse_cost', 'km', 'kmh']+fwayColList).set_index('edge')

        # ratio of projected units to meters (will be calculated if needed for Frechet distance )
        self.projectionRatio = None

        # overwrite cost if speed limit is low (so likely to be exceeded)
        mask = (edgesDf.cost < 1000000) & (edgesDf.kmh < 15)
        edgesDf.loc[mask, 'cost'] = edgesDf.loc[mask, 'cost']*(edgesDf.loc[mask, 'kmh']/15.)
        mask = (edgesDf.reverse_cost < 1000000) & (edgesDf.kmh < 15)
        edgesDf.loc[mask, 'reverse_cost'] = edgesDf.loc[mask, 'reverse_cost']*(edgesDf.loc[mask, 'kmh']/15.)
        self.edgesDf = edgesDf

        if uturnCost is None: # not defined in config file
            self.uturncost = (self.edgesDf.cost.median()+self.edgesDf.reverse_cost.median())/2.
        else:
            self.uturncost = float(uturnCost)
        self.skip_penalty = abs(temporalLL(skip_penalty))
        self.allowFinalUturn = allowFinalUturn   # allow U-turn on last edge? see issue #12. Not fully tested
        assert isinstance(self.allowFinalUturn, bool)

        # DOK matrix (like a dictionary) of shortest paths from node1 to node2
        maxN = max(edgesDf.source.max(), edgesDf.target.max()) + 1
        if maxN <= 1:
            raise Exception('Streets table %s cannot be read. %s' % streetsTable)
        self.costMatrix = sparse.dok_matrix((maxN, maxN))
        self.distMatrix = sparse.dok_matrix((maxN, maxN))

        try:
            self.costMatrix.update({})
        except NotImplementedError:  # see https://github.com/scipy/scipy/issues/8338
            self.costMatrix.update = self.costMatrix._update
            self.distMatrix.update = self.distMatrix._update

        self.timing = {'updateCostMatrix': 0, 'fillRouteGaps': 0,
                       'getPoints': 0, 'iterator': 0, 'decode': 0,
                       'median_times': []}

    def matchPostgresTrace(self, traceId):
        """Top level function to match a trace from a Postgres database
        Populates the self.bestRoute and other attributes"""

        self.clearCurrentRoutes()
        self.traceId = traceId
        self.startEndPts = ['ST_StartPoint(%s)' % self.geomName, 'ST_EndPoint(%s)' % self.geomName]
        self.matchTrace()

    def matchGPXTrace(self, fn):
        """Equivalent to matchPostgresTrace for a GPX file
        If there are multiple segments, only matches the first one
        (for now...this is easy to modify)"""

        import gpxpy
        with open(fn) as f:
            tracks = gpxpy.parse(f).tracks
        if len(tracks) == 0:
            raise Exception('GPX file %s contains no tracks' % fn)
        elif len(tracks) > 1:
            raise Warning('GPX file %s contains multiple tracks. Only parsing the first one.' % fn)
        elif len(tracks[0].segments) > 1:
            raise Warning('GPX file %s contains multiple segments on track 1. Only parsing the first one.' % fn)

        traceSegment = tracks[0].segments[0]
        self.matchGPXTraceSegment(traceSegment)

    def matchGPXTraceSegment(self, traceSegment):
        """Equivalent to matchPostgresTrace for a GPX trace segment object
        (avoid writing out GPX files)"""

        self.clearCurrentRoutes()
        if not (traceSegment.has_times()) and any([pt.time is None for pt in traceSegment.points]):  # second is because of bug in gpxpy that doesn't correctly return has_times for short traces
            raise Exception('GPX track has no readable timestamps')

        self.traceLineStr = 'ST_Transform(ST_SetSRID(ST_MakeLine(ARRAY[' + ', '.join([
            'ST_MakePointM(' + str(pt.longitude) + ',' + str(pt.latitude) + ',' +
            str(int((pt.time.replace(tzinfo=None) - datetime.datetime(1970, 1, 1)).total_seconds())) + ')'
            for pt in traceSegment.points]) + ']),4326),%s)' % self.streets_srid
        self.startEndPts = ['ST_Transform(ST_SetSRID(ST_MakePoint(' + str(np.round(pt.longitude, 5)) + ','
                            + str(np.round(pt.latitude, 5)) + '),4326),' + self.streets_srid + ')' for pt in
                            [traceSegment.points[0], traceSegment.points[-1]]]

        self.matchTrace()

    def matchTrace(self):
        """New version using Viterbi"""
        # full_starttime = time.time() # for testing

        self.getPtsDf()

        if len(self.nids) < 3:
            self.matchStatus = 1
            print('Skipping match. Need at least 3 points in trace')
            return

        self.updateCostMatrix()

        starttime = time.time()
        self.N = len(self.ptsDf)*2
        rowKeys, colKeys, scores = [], [], []   # scores are held here, before conversion to a sparse matrix

        for nid1, rr1 in self.ptsDf.loc[:self.nids[-2]].iterrows():
            rr1 = rr1.to_dict()
            # iterrows loses the dtype of the ints (converts them to floats), and only ints can index a sparse matrix
            for kk in ['edge','source','target']:
                rr1[kk] = int(rr1[kk])
            for dir1 in [0, -1]:
                idx1 = int(rr1['rownum']*2-dir1)
                # fill diagonal
                rowKeys.append(idx1)
                colKeys.append(idx1)
                scores.append((1e-10, 1, -1))
                seglength, lastnid = 0., -1
                for nid2, rr2 in self.ptsDf.loc[nid1+1:nid1+1+max_skip].drop_duplicates('edge').iterrows():  # max_skip is the maximum number of rows to skip. We drop duplicates because if this edge was done at a previous nid, we can skip
                    rr2 = rr2.to_dict()
                    for kk in ['edge','source','target']:
                        rr2[kk] = int(rr2[kk])
                    if nid2 != lastnid:  # update seglength if a new nid is being entered, and pass it to the scores functions
                        seglength += rr2['seglength']
                        lastnid = nid2
                    for dir2 in [0, -1]:
                        rowKeys.append(idx1)
                        colKeys.append(int(rr2['rownum']*2-dir2))
                        if rr1['edge'] == rr2['edge'] and dir1 == dir2:
                            scores.append(self.transProbSameEdge(rr1, rr2, dir1, seglength))
                        elif rr1['edge'] == rr2['edge'] or rr1[['target', 'source'][dir1]] != rr2[['target', 'source'][dir2]]:
                            scores.append(self.transProb(rr1, rr2, dir1, dir2, seglength))
                        else:
                            scores.append((1e10, 1e10, -1))

        # coo matrix from lists is fastest way to build a sparse matrix,
        # rather than assigning to lil or dok matrix directly
        # then convert to csr, which is more efficient for matrix algebra
        self.temporalScores = sparse.coo_matrix(([ii[0] for ii in scores], (rowKeys, colKeys)), shape=(self.N, self.N), dtype=np.float32).tocsr()
        self.topologicalScores = sparse.coo_matrix(([ii[1] for ii in scores], (rowKeys, colKeys)), shape=(self.N, self.N), dtype=np.float32).tocsr()
        self.uturns = sparse.coo_matrix(([ii[2] for ii in scores], (rowKeys, colKeys)), shape=(self.N, self.N), dtype=np.int8).tocsr()
        self.timing['iterator'] += (time.time()-starttime)

        starttime = time.time()
        d = self.viterbi()
        self.timing['decode'] += (time.time()-starttime)

        route = [(self.ptsDf.loc[self.ptsDf.rownum == int(dd/2), 'edge'].values[0], -1*dd % 2) for dd in d]  # tuple of (edge, direction)
        uturns = [self.uturns[n1, n2] for n1, n2 in zip(d[:-1], d[1:])]
        self.fillRouteGaps(route, uturns)  # fill in gaps in chosen route, populates self.bestRoute

        if self.bestRoute is None or -1 in self.bestRoute:
            self.matchStatus = 2
            return

        self.cleanupRoute(d)
        self.pointsToDrop = [ii+1 for ii, dd in enumerate(zip(d[1:], d[:-1])) if dd[0] == dd[1]]

        distLLs = self.ptsDf[self.ptsDf.rownum.isin([int(dd/2) for dd in d])].distprob.describe()[['mean', 'min']].tolist()
        distLLs[0] = distLLs[0]/weight_1stlast
        distLLs[-1] = distLLs[-1]/weight_1stlast
        temporalLLarray = temporalLL([self.temporalScores[n1, n2] for n1, n2 in zip(d[:-1], d[1:])])
        topologicalLLarray = topologicalLL([self.topologicalScores[n1, n2] for n1, n2 in zip(d[:-1], d[1:])])

        self.LL = distLLs+[temporalLLarray.mean(), temporalLLarray.min(), topologicalLLarray.mean(), topologicalLLarray.min()]
        self.matchStatus = 0
        # self.timing['median_times'].append(time.time()-full_starttime)  # for testing only

    def getPtsDf(self):
        """ Get all points on GPS traces, and edges within the search radius of each"""
        starttime = time.time()

        fwayQueryTxt = '' if fwayQuery.strip() == '' else '('+fwayQuery+') AND '

        assert self.traceId is not None or self.traceLineStr is not None
        if self.traceId is not None:  # get trace from postgres
            dpStr = 'SELECT ST_DumpPoints(%(geomName)s) AS dp FROM %(traceTable)s WHERE %(idName)s=%(traceId)s' % dict(self.cmdDict, **{'traceId': self.traceId})
        else:   # trace was a GPX file
            dpStr = 'SELECT ST_DumpPoints(%s) AS dp' % self.traceLineStr

        cmd = '''SELECT (dp).path[1]-1 AS path, %(streetIdCol)s, ST_Distance((dp).geom, %(streetGeomCol)s),
                        ST_LineLocatePoint(%(streetGeomCol)s, (dp).geom), ST_M((dp).geom)
                 FROM %(streetsTable)s, (%(dpStr)s) AS pts
                WHERE ST_DWithin((dp).geom, %(streetGeomCol)s, %(gpsError)s)
                    OR (%(fwayQuery)s ST_DWithin((dp).geom, %(streetGeomCol)s, %(gpsError_fway)s))
                ORDER BY path, st_distance''' % dict(self.cmdDict, **{'dpStr': dpStr, 'fwayQuery': fwayQueryTxt})
        pts = self.db.execfetch(cmd)
        if pts is None or pts == []:
            self.ptsDf, self.nids = None, []
            print('No streets found within the tolerance of the trace.')
            print('You might want to check the projection of the streets table and trace, or the gpsError configuration parameter.')
            return
        self.ptsDf = pd.DataFrame(pts, columns=['nid', 'edge', 'dist', 'frcalong', 'secs']).set_index('edge')

        nidStr = str(tuple(self.ptsDf.nid.unique())) if len(self.ptsDf.nid.unique()) > 1 else '('+str(self.ptsDf.nid.iloc[0])+')'
        cmd = '''SELECT (dp).path[1]-1 AS path, ST_Distance((dp).geom,lag((dp).geom) OVER (order by(dp).path[1]))/1000 AS seglength_km
                    FROM (%(dpStr)s) AS pts
                    WHERE (dp).path[1]-1 IN %(nidStr)s;''' % dict(self.cmdDict, **{'dpStr': dpStr, 'nidStr': nidStr})
        traceSegs = self.db.execfetch(cmd)

        self.ptsDf = self.ptsDf.join(self.edgesDf).reset_index().set_index('nid')
        self.ptsDf = self.ptsDf.join(pd.DataFrame(traceSegs, columns=['nid', 'seglength']).set_index('nid')).sort_index()
        if not len(np.unique(self.ptsDf.index)) == self.ptsDf.index.max()+1:  # renumber nids to consecutive range
            lookup = dict(zip(np.unique(self.ptsDf.index), np.arange(self.ptsDf.index.max()+1)))
            self.ptsDf.reset_index(inplace=True)
            self.ptsDf['nid'] = self.ptsDf.nid.map(lookup)
            self.ptsDf.set_index('nid', inplace=True)

        self.ptsDf['distprob'] = self.ptsDf.dist.apply(lambda x: distanceLL(x))
        self.nids = self.ptsDf.index.unique()
        self.ptsDf.loc[self.nids.max(), 'distprob'] *= weight_1stlast  # first point is dealt with in viterbi
        self.ptsDf['rownum'] = list(range(len(self.ptsDf)))

        self.timing['getPoints'] += (time.time()-starttime)

    def writeMatchToPostgres(self, edgeIdCol='edge_ids', writeEdgeIds=True, writeGeom=True, writeMatchScore=True, writeLLs=False):
        """Writes the edges ids back to the traces table
        writeMatchScore estimates the probability of a good match
        writeLLs writes the log likelihood information for the match, which can be used in estimating a new model"""
        if writeMatchScore and not writeGeom:  # because we compute Frechet distance and length ratios based on the new line
            raise Exception('Cannot write match score without also writing geometry.')

        cols = [(edgeIdCol, 'int[]')]
        if writeMatchScore: cols += [('match_score', 'real')]
        if writeLLs: cols += [('ll_dist_mean', 'real'), ('ll_dist_min', 'real'), ('ll_topol_mean', 'real'), ('ll_topol_min', 'real'), ('ll_distratio_mean', 'real'), ('ll_distratio_min', 'real')]
        self.db.addColumns(cols, self.traceTable, skipIfExists=True)
        if writeGeom: self.writeGeomToPostgres()  # needed for writeMatchScore, so do this first

        cDict = dict(self.cmdDict, **{'edgeIdCol': edgeIdCol,
                                      'edges': '['+','.join([str(rr) for rr in self.bestRoute])+']', 'traceId': self.traceId,
                                      'll_dist_mean': self.LL[0], 'll_dist_min': self.LL[1],
                                      'll_topol_mean': self.LL[2], 'll_topol_min': self.LL[3],
                                      'll_distratio_mean': self.LL[4], 'll_distratio_min': self.LL[5]})
        if writeMatchScore: match_score = self.getMatchScore()

        if writeEdgeIds or writeMatchScore or writeLLs:
            cmd = '''UPDATE %(traceTable)s SET ''' % cDict
            if writeEdgeIds:    cmd += '''%(edgeIdCol)s = ARRAY%(edges)s,''' % cDict
            if writeMatchScore:
                if np.isnan(match_score): 
                    cmd += 'match_score=Null,'
                else:
                    cmd += 'match_score=%s,' % np.float32(match_score)   # float32 needed to avoid underflow with very low scores
            if writeLLs: cmd += '''ll_dist_mean=%(ll_dist_mean)s, ll_dist_min=%(ll_dist_min)s,
                                   ll_topol_mean=%(ll_topol_mean)s, ll_topol_min=%(ll_topol_min)s,
                                   ll_distratio_mean=%(ll_distratio_mean)s, ll_distratio_min=%(ll_distratio_min)s,''' % cDict
            cmd = cmd[:-1]  # remove final comma
            cmd += '\n        WHERE %(idName)s=%(traceId)s;' % cDict

            self.db.execute(cmd)

    def writeGeomToPostgres(self):
        """Writes the map-matched geometry back to the traces table"""
        if self.newGeomName not in self.db.list_columns_in_table(self.traceTable):
            self.db.execute("SELECT AddGeometryColumn('%s','%s',%s,'LineString',2);" % (self.traceTable, self.newGeomName, self.trace_srid))
        if self.cleanedGeomName is not None and self.cleanedGeomName not in self.db.list_columns_in_table(self.traceTable):
            self.db.execute("SELECT AddGeometryColumn('%s','%s',%s,'LineStringM',3);" % (self.traceTable, self.cleanedGeomName, self.trace_srid))

        self.getMatchedLineString()

        if self.matchedLineString is not None:
            cmd = '''UPDATE %(traceTable)s SET %(newGeomName)s = q.geom FROM (%(matchedLineStr)s) q
                     WHERE %(idName)s = %(traceId)s;''' % dict(self.cmdDict, **{'matchedLineStr': self.matchedLineString, 'traceId': self.traceId})
            self.db.execute(cmd)

        if self.cleanedGeomName is not None:
            self.writeCleanedGeom()

    def getMatchedLineString(self):
        """Populates self.getMatchedLineString with the matched line string"""

        if self.bestRoute is None:
            self.matchedLineString = None
            return

        if self.traceId is not None:  # get startpoint and endpoint from postgres
            fromExtra = ', %s t' % self.traceTable
            whereExtra = ' AND t.%s=%s' % (self.idName, self.traceId)
        else:  # from GPX
            fromExtra, whereExtra = '', ''

        route = self.bestRoute
        if len(route) == 1:  # single edge to match
            cDict = dict(self.cmdDict, **{'edge': str(route[0]), 'traceId': self.traceId, 'startGeom': self.startEndPts[0], 'endGeom': self.startEndPts[1],
                                          'fromExtra': fromExtra, 'whereExtra': whereExtra})
            linestr = '''SELECT CASE WHEN stfrac = endfrac THEN Null
                          WHEN stfrac<endfrac THEN ST_LineSubstring(s.%(streetGeomCol)s, stfrac, endfrac)
                          ELSE ST_Reverse(ST_LineSubstring(s.%(streetGeomCol)s, endfrac, stfrac)) END AS geom
                    FROM (SELECT ST_LineLocatePoint(s1.%(streetGeomCol)s, %(startGeom)s) AS stfrac,
                                         ST_LineLocatePoint(s1.%(streetGeomCol)s, %(endGeom)s) AS endfrac
                                   FROM %(streetsTable)s s1 %(fromExtra)s WHERE s1.%(streetIdCol)s=%(edge)s %(whereExtra)s) AS pts,
                        %(streetsTable)s s WHERE s.%(streetIdCol)s=%(edge)s''' % cDict
            self.matchedLineString = linestr
            return

        prevNode = self.edgesDf.source[route[0]] if self.edgesDf.target[route[0]] in self.edgesDf.loc[route[1]][['source', 'target']].tolist() else self.edgesDf.target[route[0]]
        # This doesn't work - ST_LineMerge only handles a certain number of lines, it seems
        # linestr = 'UPDATE %s SET matched_line%s = merged_line FROM\n(SELECT ST_LineMerge(ST_Collect(st_geom)) AS merged_line FROM (\n'
        # Instead, we build up the string through iterating over the edges in the route
        routeFrcs = None  # frc of edge to use if we have a u turn
        linestr = '''SELECT ST_RemoveRepeatedPoints(merged_line) AS geom FROM
                        (SELECT ST_LineFromMultiPoint(ST_Collect((nodes).geom)) AS merged_line FROM
                            (SELECT lorder, st_dumppoints(st_geom) AS nodes FROM (\n''' % self.cmdDict

        def addEdge(linestr, ii, edge, reverse, routeFrcs, lineindex, geomField, extra=None):
            if extra == None: extra = ' WHERE'   # allows for trip_id to be added as a table
            if reverse:  # linestring needs to be reversed
                geomStr = 'ST_Reverse(%s)' % geomField if routeFrcs in [None, (0, 1)] else 'ST_Reverse(ST_LineSubstring(%s,%s,%s))' % (geomField, routeFrcs[0], routeFrcs[1])
                prevNode = self.edgesDf.source[edge]
            else:
                geomStr = '%s' % geomField if routeFrcs in [None, (0, 1)] else 'ST_LineSubstring(%s,%s,%s)' % (geomField, routeFrcs[0], routeFrcs[1])
                prevNode = self.edgesDf.target[edge]
            linestr += '\t\t\t\tSELECT %s AS lorder, %s AS st_geom FROM %s s%s s.%s = %s UNION ALL\n' % (str(lineindex), geomStr, self.streetsTable, extra, self.cmdDict['streetIdCol'], str(edge))
            return linestr, prevNode

        for ii, edge in enumerate(route):
            if prevNode not in self.edgesDf.loc[edge][['source', 'target']].tolist():
                # need to repeat the last edge - an out and back situation
                assert prevNode in self.edgesDf.loc[route[ii-1]][['source', 'target']].tolist()
                self.matchedLineString = None
                return
                linestr, prevNode = addEdge(linestr, ii-1, route[ii-1], not(reverse), 0, ii+0.5, streetGeomCol)
                stophere
            reverse = True if prevNode == self.edgesDf.target[edge] else False
            if ii == 0:  # first point - don't need whole edge
                if reverse:
                    geomField = 'ST_LineSubString(s.%(streetGeomCol)s, 0, ST_LineLocatePoint(s.%(streetGeomCol)s, %(startGeom)s))' % dict(self.cmdDict, **{'startGeom': self.startEndPts[0]})
                else:
                    geomField = 'ST_LineSubString(s.%(streetGeomCol)s, ST_LineLocatePoint(s.%(streetGeomCol)s, %(startGeom)s), 1)' % dict(self.cmdDict, **{'startGeom': self.startEndPts[0]})
                routeFrcs = None
                extra = ', %s t WHERE t.%s=%s AND' % (self.traceTable, self.idName, self.traceId) if self.traceId is not None else ' WHERE '
            elif ii == len(route)-1:  # last point
                if reverse:
                    geomField = 'ST_LineSubString(s.%(streetGeomCol)s, ST_LineLocatePoint(s.%(streetGeomCol)s, %(endGeom)s), 1)' % dict(self.cmdDict, **{'endGeom': self.startEndPts[1]})
                else:
                    geomField = 'ST_LineSubString(s.%(streetGeomCol)s, 0, ST_LineLocatePoint(s.%(streetGeomCol)s, %(endGeom)s))' % dict(self.cmdDict, **{'endGeom': self.startEndPts[1]})
                routeFrcs = None
                extra = ', %s t WHERE t.%s=%s AND' % (self.traceTable, self.idName, self.traceId) if self.traceId is not None else ' WHERE '
            else:
                geomField, extra = streetGeomCol, None
                if edge == route[ii+1] and edge != route[ii-1] and (ii+2 == len(route) or edge != route[ii+2]):  # first edge in uTurn, but not if we have a triple
                    # uturnFrcs==-1 means an unexpected Uturn
                    routeFrcs = None if self.uturnFrcs[ii] == -1 else (self.uturnFrcs[ii][0], 1) if reverse else (0, self.uturnFrcs[ii][1])
                    if routeFrcs is None: print('Warning. Missing Uturn in trace %s' % self.traceId)
                elif edge == route[ii-1] and (ii == 1 or edge != route[ii-2]):  # second edge in uTurn
                    routeFrcs = None if self.uturnFrcs[ii-1] == -1 else (0, self.uturnFrcs[ii-1][1]) if reverse else (self.uturnFrcs[ii-1][0], 1)
                    if routeFrcs is None: print('Warning. Missing Uturn in trace %s' % self.traceId)
                else:
                    routeFrcs = None
            linestr, prevNode = addEdge(linestr, ii, edge, reverse, routeFrcs, ii+1, geomField, extra)

        linestr = linestr[:-11] + ') AS e ORDER BY lorder) AS m) AS p'  # remove last UNION ALL

        self.matchedLineString = linestr

    def getMatchAsWKT(self):
        """Returns self.matchedLineString as WKT"""
        if self.bestRoute is None:
            raise Exception('No matched route found. Run matchPostgresTrace() or matchGPXTrace() first')
        if self.matchStatus != 0:
            return None

        if self.matchedLineString is None:
            self.getMatchedLineString()
        return self.db.execfetch('SELECT ST_AsText(geom) FROM (%s) t1;' % self.matchedLineString)[0][0]

    def writeCleanedGeom(self):
        """Writes new geometry of original traces minus any dropped points"""
        cmd = 'UPDATE %s SET %s = OLDGEOM WHERE %s = %s;' % (self.traceTable, self.cleanedGeomName, self.idName, self.traceId)
        for ptNum in self.pointsToDrop:
            cmd = cmd.replace('OLDGEOM', 'ST_RemovePoint(OLDGEOM, %d)' % ptNum)
        cmd = cmd.replace('OLDGEOM', self.geomName)
        self.db.execute(cmd)

    def getMatchScore(self, verbose=False):
        """Calculates the match score for the current trace"""

        if self.bestRoute is None: return None
        if self.qualityModelCoeffs is None: self.loadQualityModel()
        if verbose:
            print('Coefficients: {}'.format(self.qualityModelCoeffs))
        xb = self.qualityModelCoeffs['intercept']
        if 'frechet_dist' in self.qualityModelCoeffs :
            if self.projectionRatio is None:        # get ratio of projected units to meters (to be able to use Frechet distance)
                cmd = '''SELECT AVG(ST_Length(%(streetGeomCol)s) / st_length(ST_Transform(%(streetGeomCol)s, 4326)::geography)) 
                            FROM %(streetsTable)s WHERE ST_Length(%(streetGeomCol)s)>0 LIMIT 1000;''' % self.cmdDict
                self.projectionRatio = self.db.execfetch(cmd)[0][0]
                if self.verbose: print('Using projection ratio (map units to meters) of {:.3f}'.format(self.projectionRatio ))
            xb += self.qualityModelCoeffs['frechet_dist']*self.frechet()*self.projectionRatio
            if verbose: print('Frechet: {}'.format(self.frechet()))
        for ii, llName in enumerate(['ll_dist_mean', 'll_dist_min', 'll_topol_mean', 'll_topol_min', 'll_distratio_mean', 'll_distratio_min']):
            if llName in self.qualityModelCoeffs:
                xb += self.qualityModelCoeffs[llName]*self.LL[ii]
            if verbose: print(llName+': '+str(self.LL[ii]))
        if 'gpsMatchRatio' in self.qualityModelCoeffs or 'matchGpsRatio' in self.qualityModelCoeffs:
            geomToUse = self.geomName if self.cleanedGeomName is None else self.cleanedGeomName
            gpslength, matchlength = self.db.execfetch('''SELECT ST_Length(%(geomToUse)s), ST_Length(%(newGeomName)s
                                                             FROM %(traceTable)s WHERE trip_id=%(id)s;''' % dict(self.cmdDict, **{'geomToUse': geomToUse, 'id': self.traceId}))
            if 'gpsMatchRatio' in self.qualityModelCoeffs:
                xb += self.qualityModelCoeffs['gpsMatchRatio']*1.*matchlength/gpslength
                if verbose: print('gpsMatchRatio: {}'.format(1.*matchlength/gpslength))
            if 'matchGpsRatio' in self.qualityModelCoeffs:
                xb += self.qualityModelCoeffs['matchGpsRatio']*1.*gpslength/matchlength
                if verbose: print('matchGpsRatio: {}'.format(1.*gpslength/matchlength))

        return 1./(1+math.exp(-1*max(xb, -500)))  # max is to avoid overflow error

    def loadQualityModel(self):
        """Loads the coefficients from the rpy2 model estimated previously"""
        try:
            execPath = os.path.dirname(os.path.realpath(__file__))+'/'
        except:
            execPath = os.getcwd()+'/'
        if os.path.exists(self.qualityModelFn):
            fn = self.qualityModelFn
        elif os.path.exists(execPath+self.qualityModelFn):
            fn = execPath+self.qualityModelFn
        else:
            raise Exception('Quality model coefficients file %s not found' % self.qualityModelFn)
        with open(fn) as f:
            coeffLines = f.read().split('\n')

        for ii, cl in enumerate(coeffLines):
            if 'intercept' in cl.lower():
                coeffLines[ii] = 'intercept\t'+cl.split()[1]
                break
            raise Exception('Quality model %s must include an intercept' % self.qualityModelFn)
        self.qualityModelCoeffs = dict([(cl.split()[0], float(cl.split()[1])) for cl in coeffLines if len(cl.split()) == 2])
        knownCols = ['intercept', 'frechet_dist', 'll_dist_mean', 'll_dist_min', 'll_topol_mean', 'll_topol_min', 'll_distratio_mean', 'll_distratio_min', 'gpsMatchRatio', 'matchGpsRatio']
        if any([cc not in knownCols for cc in self.qualityModelCoeffs]):
            raise Exception('Error in quality model %s. Coefficient name not understood' % self.qualityModelFn)

    def updateCostMatrix(self):
        """Get cost and route to travel from each edge to each edge within 1km, and store it as a dok matrix
        For each edge in edgesToDo, we need to get the cost to travel to all other edges in edgesToDo
        edgesToDo is a dictionary, with a list of edges to do for each (edge) key"""

        starttime = time.time()
        # Old version - pre-pgr v2.3
        if 0:
            nodeList = np.unique(self.edgesDf.loc[self.ptsDf.edge.unique(), ['source', 'target']].values.flatten())
            allNodesToDo = [n1 for n1 in nodeList if not all([(n1, n2) in self.costMatrix for n2 in nodeList])]
            for n1 in allNodesToDo:
                n2sToDo = [n2 for n2 in allNodesToDo if (n1, n2) not in self.costMatrix]  # eliminate ones we've already done
                if n2sToDo == []: continue
                # Get routes from node to all other nodes within 1km, or that are in edgeTuple
                cmd = '''SELECT id1, sum(pgr.cost), sum(km)
                      FROM pgr_kdijkstraPath(
                          'SELECT %(streetIdCol)s, %(source)s, %(target)s, %(cost)s, %(reverse_cost)s FROM %(streetsTable)s',
                          %(n1)s, ARRAY%(n2s)s, True, True) pgr, %(streetsTable)s s
                          WHERE id3=s.id GROUP BY id1;''' % dict(self.cmdDict, **{'n1': str(n1), 'n2s': str(list(n2sToDo))})
                result = self.db.execfetch(cmd)
                result = [(rr[0], rr[1], rr[2]) if rr[1] >= 0 else (rr[0], 10000000, 10000000) for rr in result]
                self.costMatrix.update(dict([((n1, ff[0]), ff[1]) for ff in result]))
                self.distMatrix.update(dict([((n1, ff[0]), ff[2]) for ff in result]))
                self.costMatrix.update({(n1, n1): 0})
                self.distMatrix.update({(n1, n1): 0})

        if 1:
            nodeList = np.unique(self.edgesDf.loc[self.ptsDf.edge.unique(), ['source', 'target']].values.flatten())
            allNodesToDo = [n1 for n1 in nodeList if not all([(n1, n2) in self.costMatrix for n2 in nodeList])]

            # to avoid postgres memory problems, split up into sublists
            nodesToDoList = [allNodesToDo[x:x+maxNodes] for x in range(0, len(allNodesToDo), maxNodes)]

            for nodesToDo_src in nodesToDoList:
                for nodesToDo_tgt in nodesToDoList:
                    if nodesToDo_src and nodesToDo_tgt:
                        cmd = '''SELECT start_vid, end_vid, sum(pgr.cost) AS pgrcost, sum(s.%(km)s::real) AS length_km
                              FROM %(streetsTable)s s,
                                   pgr_dijkstra('SELECT %(streetIdCol)s, %(source)s, %(target)s, %(cost)s, %(reverse_cost)s FROM %(streetsTable)s',
                                                 ARRAY%(srcnodes)s, ARRAY%(tgtnodes)s, True) AS pgr
                              WHERE s.%(streetIdCol)s=pgr.edge
                              GROUP BY start_vid,end_vid;''' % dict(self.cmdDict, **{'srcnodes': str(list(nodesToDo_src)), 'tgtnodes': str(list(nodesToDo_tgt))})
                        result = self.db.execfetch(cmd)
                        self.costMatrix.update({((ff[0], ff[1]), ff[2]) if ff[2] >= 0 else ((ff[0], ff[1]), 10000000) for ff in result})
                        self.distMatrix.update({((ff[0], ff[1]), ff[3]) if ff[3] >= 0 else ((ff[0], ff[1]), 10000000) for ff in result})

                        # add route to/from same node
                        self.costMatrix.update({((nn, nn), 0.0) for nn in nodesToDo_src})
                        self.distMatrix.update({((nn, nn), 0.0) for nn in nodesToDo_src})

                        # add route where pgr_dijkstra does not return a result, usually because of islands
                        problemNodes = {((n1, n2), 10000000) for n1 in nodesToDo_src for n2 in nodesToDo_tgt if (n1, n2) not in self.costMatrix}
                        self.costMatrix.update(problemNodes)
                        self.distMatrix.update(problemNodes)

        self.timing['updateCostMatrix'] += (time.time()-starttime)

    def fillRouteGaps(self, route, uturns):
        """Takes the top route in the list (must be sorted!) and fills in gaps"""
        # remove duplicates
        keepPts = [0]+[ii+1 for ii, rr in enumerate(route[1:]) if route[ii] != rr]
        edgeList = [rr[0] for ii, rr in enumerate(route) if ii in keepPts]
        nodeList = [int(self.edgesDf.loc[rr[0], 'target']) if rr[1] == 0 else
                    int(self.edgesDf.loc[rr[0], 'source'])
                    for ii, rr in enumerate(route) if ii in keepPts]
        uturnList = [rr for ii, rr in enumerate(uturns) if ii+1 in keepPts]

        if len(edgeList) == 1:
            self.bestRoute = edgeList
            self.uturnFrcs = []
            return

        starttime = time.time()

        firstEdge = True
        fullroute = [edgeList[0]]  # first edge
        uturnFrcs = [-1]
        keepPts.append(len(self.nids))  # this is for indexing (below)
        oNode = nodeList[0]  # origin for next route
        for ii, (edge, node, uturn, ptId) in enumerate(zip(edgeList[1:], nodeList[1:], uturnList, keepPts[1:-1])):
            # node is far end of the edge we are going to. We want the other end, called dNode
            if uturn in [3, 4]:  # uturn on first edge
                fullroute.append(fullroute[-1])
                frcs = self.ptsDf.loc[keepPts[ii]:keepPts[ii+1]-1].loc[self.ptsDf.loc[keepPts[ii]:keepPts[ii+1]-1].edge == route[ptId-1][0], 'frcalong']
                uturnFrcs = uturnFrcs[:-1]+[(frcs.min(), frcs.max()), -1]
                oNode = self.edgesDf.target[fullroute[-1]] if self.edgesDf.source[fullroute[-1]] == oNode else self.edgesDf.source[fullroute[-1]]
            if uturn in [2, 4]:  # uturn on last edge
                dNode = node
            else:
                dNode = self.edgesDf.target[edge] if self.edgesDf.source[edge] == node else self.edgesDf.source[edge]

            if oNode != dNode:  # missing edges
                cmd = '''SELECT array_agg(edge) AS edges FROM pgr_dijkstra(
                            'SELECT %(streetIdCol)s, %(source)s, %(target)s, %(cost)s, %(reverse_cost)s FROM %(streetsTable)s', %(oNode)s, %(dNode)s, True);
                      ''' % dict(self.cmdDict, **{'oNode': str(oNode), 'dNode': str(dNode)})
                result = self.db.execfetch(cmd)
                if result[0][0] is None:  # because of a disconnected network/islands
                    print('Could not match trace. Network may be disconnected or have islands.')
                    self.bestRoute = None
                    self.uturnFrcs = None
                    return
                fullroute += result[0][0][:-1]
                uturnFrcs += [-1]*(len(result[0][0])-1)
            if uturn in [2, 4]:
                fullroute.append(edge)
                frcs = self.ptsDf.loc[keepPts[ii+1]:keepPts[ii+2]].loc[self.ptsDf.loc[keepPts[ii+1]:keepPts[ii+2]].edge == route[ptId][0], 'frcalong']
                uturnFrcs.append((frcs.min(), frcs.max()))
            fullroute.append(edge)
            if uturn == 1:
                frcs = self.ptsDf.loc[keepPts[ii]:keepPts[ii+2]-1].loc[self.ptsDf.loc[keepPts[ii]:keepPts[ii+2]-1].edge == route[ptId][0], 'frcalong']
                if fullroute[-1] == fullroute[-2] and uturnFrcs[-1] == -1:  # uturn already made on previous edge, but not counted
                    uturnFrcs = uturnFrcs[:-1]+[(frcs.min(), frcs.max()), -1]
                else:
                    uturnFrcs.append((frcs.min(), frcs.max()))
            else:
                uturnFrcs.append(-1)

            oNode = node   # starting point for next edge

        # Check whether there is a U-turn on the final edge -  issue #12
        if self.allowFinalUturn:
            for nid in reversed(self.nids[1:]):  # find out where the final edge starts
                if route[nid]!=route[nid-1]:
                    break
            frcsAlong = self.ptsDf[self.ptsDf.edge==route[-1][0]].loc[nid:,'frcalong']
            threshold = 0.1 # how many km the furthest GPS ping has to be along, in order to add a uturn
            if ((route[-1][1] == 0 and frcsAlong.max()*self.edgesDf.loc[route[-1][0],'km'] > threshold) or
                (route[-1][1] == 1 and (1-frcsAlong.min())*self.edgesDf.loc[route[-1][0],'km'] > threshold)):  
                fullroute.append(fullroute[-1])
                uturnFrcs[-1] = (frcsAlong.min(), frcsAlong.max())
                uturnFrcs.append(-1)


        self.timing['fillRouteGaps'] += (time.time()-starttime)
        assert len(uturnFrcs) == len(fullroute)
        self.bestRoute = fullroute
        self.uturnFrcs = uturnFrcs

    def cleanupRoute(self, d):
        # if the first or last edge has hardly been started (less than 5m), drop it
        lastEdgeLength = self.edgesDf.loc[self.bestRoute[-1], 'km']
        frc = self.ptsDf.frcalong[self.ptsDf.rownum == int(d[-1]/2)].values[0]
        if d[-1] % 2 == 1: frc = 1-frc  # reverse
        while (len(self.bestRoute) > 1 and 
              (lastEdgeLength*frc < 0.005 or (self.allowFinalUturn is False and self.bestRoute[-1] == self.bestRoute[-2]))):
            self.bestRoute = self.bestRoute[:-1]
            self.uturnFrcs = self.uturnFrcs[:-1]
            lastEdgeLength, frc = 1, 1

        firstEdgeLength = self.edgesDf.loc[self.bestRoute[0], 'km']
        frc = self.ptsDf.frcalong[self.ptsDf.rownum == int(d[0]/2)].values[0]
        if d[0] % 2 == 0: frc = 1-frc  # reverse
        # also get rid of duplicate first edges. This happens because the 1st edge must include at least 20% of its length, but this is bypassed if there is a U-turn
        while len(self.bestRoute) > 1 and (firstEdgeLength*frc < 0.005 or self.bestRoute[0] == self.bestRoute[1]):
            self.bestRoute = self.bestRoute[1:]
            self.uturnFrcs = self.uturnFrcs[1:]
            firstEdgeLength, frc = 1, 1

        if len(self.bestRoute) > 1 and self.bestRoute[0] == self.bestRoute[1]:
            self.bestRoute = self.bestRoute[1:]
            self.uturnFrcs = self.uturnFrcs[1:]

        # get rid of triple edge sequences (often because of a GPS 'splodge'
        while len(self.bestRoute) > 2:
            for ii, id in enumerate(self.bestRoute[2:]):
                if [id]*2 == self.bestRoute[ii:ii+2] and len(self.bestRoute) > 2:
                    self.bestRoute = self.bestRoute[:ii]+self.bestRoute[ii+2:]
                    self.uturnFrcs = self.uturnFrcs[:ii]+self.uturnFrcs[ii+2:]
                    break  # do another full loop
            break  # break when no more triple duplicates

    def clearCurrentRoutes(self):
        """Clears previous information about the route and best match"""
        self.minTime = -1
        self.bestRoute = None
        self.uturnFrcs = None
        self.pointsToDrop = []
        self.routes = []
        self.matchStatus = None
        self.LL = None
        self.traceId = None
        self.ptsDf = None
        self.traceId = None
        self.traceLineStr = None
        self.startEndPts = None
        self.matchedLineString = None
        self.nids = []

    def addQualityColumns(self, columns=None, forceUpdate=False):
        """Add columns to the table that allow the quality of the match to be predicted
        For development use in estimating the logistic model
        Columns can be specified, or None does all of them"""
        if self.newGeomName not in self.db.list_columns_in_table(self.traceTable):
            raise Exception('geometry columns %s does not exist in table %s' % (self.newGeomName, self.traceTable))

        colsToAdd = [('pingtime_max', 'real'), ('pingtime_mean', 'real'),
                     ('gpsdist', 'real'), ('matchdist', 'real'),
                     ('frechet_dist', 'real')]
        if columns is not None:
            if not isinstance(columns, list): columns = [columns]
            colsToAdd = [cc for cc in colsToAdd if cc[0] in columns]
            assert len(colsToAdd) > 0

        if colsToAdd[0][0] in self.db.list_columns_in_table(self.traceTable):
            if forceUpdate:
                dropTxt = ', '.join(['DROP COLUMN IF EXISTS '+cc[0] for cc in colsToAdd])
                self.db.execute('ALTER TABLE %s %s;' % (self.traceTable, dropTxt))
            else:
                print('Quality columns already added. Skipping')
                return

        self.db.addColumns(colsToAdd, self.traceTable)

        if columns is None or 'pingtime_max' in columns or 'pingtime_mean' in columns:
            cmd = 'UPDATE %s t SET ' % self.traceTable
            cmd += ', '.join([cc+'=t3.'+cc for cc, _ in colsToAdd if cc in ['pingtime_max', 'pingtime_mean']])
            cmd += ''' FROM (
                      SELECT %(idName)s, MAX(timedelta) AS pingtime_max, AVG(timedelta) AS pingtime_mean
                      FROM (
                         SELECT %(idName)s, (ST_M((dp).geom) - lag(ST_M((dp).geom), 1)
                           OVER (PARTITION BY %(idName)s ORDER BY (dp).path[1])) AS timedelta
                         FROM (
                             SELECT %(idName)s, ST_DumpPoints(%(geomName)s) AS dp
                                 FROM %(traceTable)s) AS t1) t2
                      GROUP BY %(idName)s) t3
                    WHERE t.%(idName)s=t3.%(idName)s;''' % self.cmdDict
            self.db.execute(cmd)

        if columns is None or 'gpsdist' in columns:
            self.db.execute('UPDATE %(traceTable)s SET gpsdist = ST_Length(%(geomName)s);' % self.cmdDict)
        if columns is None or 'matchdist' in columns:
            self.db.execute('UPDATE %(traceTable)s SET matchdist = ST_Length(%(newGeomName)s);' % self.cmdDict)

        if columns is None or 'frechet_dist' in columns:
            ids = sorted([ii[0] for ii in self.db.execfetch('SELECT %(idName)s FROM %(traceTable)s WHERE %(newGeomName)s IS NOT NULL;' % self.cmdDict)])
            for id in ids:
                self.traceId = id
                fd = self.frechet()
                self.db.execute('UPDATE %s SET frechet_dist = %s WHERE trip_id=%s;' % (self.traceTable, fd, id))

    def transProbSameEdge(self, rr1, rr2, dir, sl):
        """Returns input to likelihood functions for movement along an edge
           (rr1.edge==rr1.edge)"""
        frc = abs(rr2['frcalong']-rr1['frcalong']) if dir == 0 else abs(rr1['frcalong']-rr2['frcalong'])
        distratio = 1 if sl == 0 else rr1['km']*frc*1./sl
        return max(rr1[['cost', 'reverse_cost'][dir]]*frc*60*60./max(1, rr2['secs']-rr1['secs']), 1e-10), max(distratio, 1), -1

    def transProb(self, rr1, rr2, dir1, dir2, sl):
        """Returns input to likelihood functions for transition between edges
           (rr1.edge!=rr1.edge)"""
        if dir1 == 0:
            frc1 = 1-rr1['frcalong']   # frc of edge remaining
            n0 = rr1['source']
            n1 = rr1['target']
            e1cost = rr1['cost']
        else:
            frc1 = rr1['frcalong']
            n0 = rr1['target']
            n1 = rr1['source']
            e1cost = rr1['reverse_cost']

        if dir2 == 0:
            frc2 = rr2['frcalong']  # frc of edge that will be traveled
            n2 = rr2['source']
            n3 = rr2['target']
            e2cost = rr2['cost']
        else:
            frc2 = 1-rr2['frcalong']
            n2 = rr2['target']
            n3 = rr2['source']
            e2cost = rr2['reverse_cost']

        # routing cost is cost of 1st edge + routing cost + uturn cost + cost of last edge. e1!=e2 needed in case there are self-loops
        rc = self.costMatrix[n1, n2]  # routing cost
        uturn = -1
        if rr1['edge'] == rr2['edge'] and dir1 != dir2:
            uturn = 1
            if frc1 > 0.95 and frc2 > 0.95:
                frc1, frc2 = 50, 50  # haven't started the edge and already want to do a U-turn? Disallow
            elif frc1 > frc2:  # continuing along original path, then U turn
                frc1, frc2 = frc1-frc2, 0
            else:  # U turn, then backtrack
                frc1, frc2 = 0, frc2-frc1
        elif rr1['edge'] != rr2['edge']:
            e1Uturn = np.round(rc, 8) == np.round(self.costMatrix[n1, n0]+self.costMatrix[n0, n2], 8)
            e2Uturn = np.round(rc, 8) == np.round(self.costMatrix[n1, n3]+e2cost, 8)
            if e2Uturn:  # U-turn on e2, disallow Uturns on both (causes problems in a grid setting)
                uturn = 2
                e2cost = rr2[['reverse_cost', 'cost'][dir2]]
                frc2 = 1-frc2
                if frc2 < 0.05: frc2 = 100  # haven't started the edge and already want to do a U-turn? Disallow
            if e1Uturn:  # u-turn on e1
                uturn = 3
                e1cost = rr1[['reverse_cost', 'cost'][dir1]]
                frc1 = 1-frc1
                if frc1 < 0.05: frc1 = 100  # haven't started the edge and already want to do a U-turn? Disallow
            if e1Uturn and e2Uturn:
                uturn = 4  # note this attracts a greater uturncost
        else:
            somemistake

        if sl == 0 or uturn == 1:  # Uturn on e1, but same edge as e2
            dratio = 1
        else:  # note that frc1 and frc2 have already been changed in the event of a Uturn
            if uturn == -1:  # no Uturn
                networkDist = self.distMatrix[n1, n2]+rr1['km']*frc1+rr2['km']*frc2
            elif uturn == 2:  # Uturn on e2
                networkDist = self.distMatrix[n1, n3]+rr1['km']*frc1+rr2['km']*frc2
            elif uturn == 3:   # Uturn on e1
                networkDist = self.distMatrix[n0, n2]+rr1['km']*frc1+rr2['km']*frc2
            elif uturn == 4:   # Uturn on both
                networkDist = self.distMatrix[n0, n3]+rr1['km']*frc1+rr2['km']*frc2
            else:
                somemistakehere
            dratio = networkDist*1./sl

        cratio = (e1cost*frc1+rc+self.uturncost*(uturn > 0)+self.uturncost*2*(uturn == 4)+e2cost*frc2)*60*60./max(1, rr2['secs']-rr1['secs'])

        # The values will be in a sparse matrix, so zeros aren't distinguishable from empty arrays
        return max(cratio, 1e-10), max(dratio, 1), uturn

    def viterbi(self):
        """Derived from https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
           Key changes are to work in log space, make use of self.matrices, and use sparse representations to save memory"""
        # obs=None simply steps through the K state probabilities in order, i.e. K = self.obsProb.shape[1]
        # if obs is None: obs=range(self.obsProb.shape[1])

        # to distinguish true zeros from empty cells, we will add 1 to all values here
        backpt = sparse.lil_matrix((self.N, self.nids.max()+1), dtype=np.int32)

        # initialization
        trellis_0 = (np.repeat(self.ptsDf.loc[[0], 'distprob'], 2) * weight_1stlast).values
        lastidx1, lastidx2 = 0, trellis_0.shape[0]
        for nid in self.nids[1:]:

            nToSkip = 0 if nid == len(self.nids)-1 else max_skip  # Don't allow last point to be dropped, so set other probabilities very small

            idx1 = int(self.ptsDf.loc[max(0, nid-nToSkip), 'rownum'].min()*2)
            idx2 = int(self.ptsDf.loc[nid, 'rownum'].max()*2+2)
            iterN = idx2-idx1  # npoints on this iteration

            # calculate the probabilities from the scores matrices
            transScoreArr = self.temporalScores[lastidx1:lastidx2, idx1:idx2].toarray()
            transScoreArr[transScoreArr == 0] = 1e10  # these are not true zeros, but empty cells
            distratioArr = self.topologicalScores[lastidx1:lastidx2, idx1:idx2].toarray()
            distratioArr[distratioArr == 0] = 1e10  # these are not true zeros, but empty cells

            LL = temporalLL(transScoreArr) + topologicalLL(distratioArr)

            # 1st term is the probability from the previous iteration. 2nd term is observation probability. 3rd term is the transition probability
            trellis_1 = np.nanmax(np.broadcast_to(trellis_0.reshape(-1, 1), (trellis_0.shape[0], iterN)) +
                                  (np.repeat(self.ptsDf.loc[nid-nToSkip:nid, 'distprob'], 2)-self.skip_penalty*np.repeat(np.array(nid-self.ptsDf.loc[nid-nToSkip:nid].index)**2, 2)).values.reshape(-1, 1).T +
                                  LL, 0)

            backpt[idx1:idx2, nid] = np.nanargmax(np.broadcast_to(trellis_0.reshape(-1, 1), (trellis_0.shape[0], iterN)) + LL, 0).reshape(-1, 1) + lastidx1 + 1

            trellis_0 = trellis_1
            lastidx1, lastidx2 = idx1, idx2

        # termination
        tokens = [np.nanargmax(trellis_1)+idx1]
        for i in reversed(self.nids[1:]):
            tokens.append(backpt[tokens[-1], i]-1)
        return tokens[::-1]

    def frechet(self, resolution=20):
        """Returns Frechet distance between match and the GPS trace
        Note that the GPX variant does not consider the cleaned line (yet)
        """

        assert self.traceId is not None or self.traceLineStr is not None

        # PostGIS now has a Frechet function. But this seems much slower than the implementation here, so don't use it

        if 1 or float(self.postgis_version.split('.')[0]) < 3 and float(self.postgis_version.split('.')[1]) < 4:
                # This is placeholder pending ST_FrechetDistance() in PostGIS 2.4.0
            if self.traceId is not None:
                frechetGeomName = self.cleanedGeomName if self.cleanedGeomName else self.geomName
                traceLength = self.db.execfetch('SELECT ST_Length(%(frechetGeomName)s) FROM %(traceTable)s WHERE %(idName)s=%(traceId)s;' % dict(self.cmdDict, **{'traceId': self.traceId, 'frechetGeomName': frechetGeomName}))[0][0]
                cmd = '''WITH l AS (SELECT ST_Force2D(%(newGeomName)s) as l1,
                                        ST_Force2D(%(frechetGeomName)s) as l2
                                    FROM %(traceTable)s where %(idName)s=%(traceId)s)''' % dict(self.cmdDict, **{'traceId': self.traceId, 'frechetGeomName': frechetGeomName})
            else:
                if self.matchedLineString is None: self.getMatchedLineString()
                traceLength = self.db.execfetch('SELECT ST_Length(%s)' % self.traceLineStr)[0][0]
                cmd = '''WITH l AS (SELECT ST_Force2D(geom) as l1, ST_Force2D(%s) AS l2 FROM (%s) mls)''' % (self.traceLineStr, self.matchedLineString)

            if traceLength == 0: return np.nan
            frc = max(0.0025, resolution*1./traceLength)  # max is because of limit of target lists can have at most 1664 entries

            cmd += '\nSELECT '+','.join([' ST_AsText(ST_LineInterpolatePoint(l1,'+str(f)+')), ST_AsText(ST_LineInterpolatePoint(l2,'+str(f)+'))' for f in np.arange(0, 1, frc)])
            cmd += ' FROM l'
            try:
                pts = [rr.strip('POINT()').split() for rr in self.db.execfetch(cmd)[0]]
            except Exception as e:  # 'AttributeError if matched_line is None
                print(e)
                print(cmd)
                return np.nan
            l1 = np.array([[float(ll[0]), float(ll[1])] for ii, ll in enumerate(pts) if ii % 2 == 0])
            l2 = np.array([[float(ll[0]), float(ll[1])] for ii, ll in enumerate(pts) if ii % 2 == 1])
            frechet_dist = mmt.frechetDist(l1, l2)

        else: # use new PostGIS function
            if self.traceId:
                frechetGeomName = self.cleanedGeomName if self.cleanedGeomName else self.geomName
                frechet_dist = self.db.execfetch('''SELECT ST_FrechetDistance(%(newGeomName)s, %(frechetGeomName)s, 0.05) 
                                                        FROM %(traceTable)s WHERE %(idName)s=%(traceId)s;''' % dict(self.cmdDict, **{'traceId': self.traceId, 'frechetGeomName': frechetGeomName}))[0][0]
            else:
                if self.matchedLineString is None: self.getMatchedLineString()
                cmd = '''WITH l AS (SELECT ST_Force2D(geom) as l1, ST_Force2D(%s) AS l2 FROM (%s) mls)
                            SELECT ST_FrechetDistance(l1, l2, 0.01) FROM l;''' % (self.traceLineStr, self.matchedLineString)
                frechet_dist = self.db.execfetch(cmd)[0][0]

        return frechet_dist

def distanceLL(distance):
    """Geometric log likelihood function for how to penalize edges that are further from the point
    Similar to Newson and Krummer 2009
    This can take a scalar or a numpy array"""
    return stats.t(df=20, scale=sigma_z).logpdf(distance)


def temporalLL(travelcostratio):
    """Log likelihood function for the transition between different edges
    Input is ratio of implied speed to speed limit"""
    if isinstance(travelcostratio, list):
        travelcostratio = np.array(travelcostratio)
    if isinstance(travelcostratio, np.ndarray):
        retvals = stats.expon(scale=temporal_scale).logpdf(travelcostratio)
        retvals[travelcostratio > 1] = (stats.norm(1, scale=sigma_t).logpdf(travelcostratio[travelcostratio > 1])+temporalLL_ratio)
        return retvals*temporal_weight
    else:  # scalar
        if travelcostratio <= 1:
            return stats.expon(scale=temporal_scale).logpdf(travelcostratio)*temporal_weight
        else:
            return (stats.norm(1, scale=sigma_t).logpdf(travelcostratio)+temporalLL_ratio)*temporal_weight


def topologicalLL(distratio):
    """this is the topological log likelihood function, based on the distance ratio between GPS trace and matched line"""
    dr = np.maximum(0, np.array(distratio)-1)    # distratio can be less than 1 if there is a U-turn, so enforce a minimum
    return stats.t(df=20, scale=sigma_topol).logpdf(dr)*topol_weight


# ensures that the two distributions match at 1
temporalLL_ratio = (stats.expon(scale=temporal_scale).logpdf(1)-stats.norm(scale=sigma_t).logpdf(0))


class qualityPredictor():
    """Estimates a new logistic model to predict match score
    This is only required if you want to restimate the coefficients in mapMatching_coefficients.txt"""
    def __init__(self, traceTable, idName, geomName, path='', db=None):
        self.cmdDict = dict({'traceTable': traceTable, 'idName': idName, 'geomName': geomName})
        if path != '' and not(path.endswith('/')): path += '/'
        self.traceTable = traceTable
        self.path = path
        self.outLogitFn = path+'mapmatching_model.pickle'
        self.outLogitCoeffsFn = path+'mapmatching_coefficients.txt'
        self.outLogFn = path+'logitModelFitting.txt'
        self.writeLog('\n____________qualityPredictor LOG table %s____________\n' % (traceTable))

        if db is None:
            self.db = mmt.dbConnection(pgLogin=pgInfo)
        else:
            self.db = db

    def getPostgresData(self, table=None, estSample=True):
        if table is None: table = self.traceTable
        cols = 'pingtime_max, pingtime_mean, frechet_dist, gpsdist, matchdist, ll_dist_mean, ll_dist_min, ll_topol_mean, ll_topol_min, ll_distratio_mean, ll_distratio_min'
        if estSample:
            cmd = """SELECT %(idName)s, match_good, ST_NPoints(%(geomName)s), %(cols)s
                        FROM %(table)s
                        WHERE matched_line is Not Null AND match_good!='drop' AND match_good is not Null""" % dict(self.cmdDict, **{'table': table, 'cols': cols})
            df = pd.DataFrame(self.db.execfetch(cmd), columns=['trip_id', 'match_good', 'num_pings']+cols.split(', ')).set_index('trip_id')

            assert all(df.match_good.isin(['y', 'n']))
            df.match_good = (df.match_good == 'y').astype(float)

        else:  # we don't have match_good columns
            cmd = """SELECT %(idName)s, ST_NPoints(%(geomName)s), %(cols)s
                        FROM %(table)s
                        WHERE matched_line is Not Null""" % dict(self.cmdDict, **{'table': table, 'cols': cols})
            df = pd.DataFrame(self.db.execfetch(cmd), columns=['trip_id', 'num_pings']+cols.split(', ')).set_index('trip_id')

        df['gpsMatchRatio'] = df.matchdist/df.gpsdist
        df['gpsMatchRatio_sq'] = df.gpsMatchRatio**2
        df['matchGpsRatio'] = 1./df.gpsMatchRatio   # to catch trips where we cut too many corners as well
        for col in ['gpsMatchRatio', 'pingtime_max']:
            df[col+'Sq'] = df[col]**2
        df['gpsMatchRatio_pingtimemean'] = df.gpsMatchRatio*df.pingtime_mean
        df['highfreq'] = df.pingtime_mean < 5
        df['gpsMatchRatio_highfreq'] = df.gpsMatchRatio*df.highfreq
        df = df[df.pingtime_mean <= 30]  # drop low-frequency traces

        return df

    def estimateQualityModel(self, holdback=0.3, threshold_score=0.7):
        """
        Given a table of the "good" trip ids (i.e., the trace has sufficient information and the fit is good)
        comes up with a prediction of the quality of that trip id

        holdback is the fraction that is held back for out of sample validation
        """
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri
        import cPickle
        rpy2.robjects.numpy2ri.activate()
        df = self.getPostgresData()

        self.scatterplots(df)
        self.boxplots(df)

        # split sample into estimation and reserve
        random.seed(1)
        ids_reserve = random.sample(df.index.tolist(), int(len(df)*holdback))

        dfEst = df.loc[np.logical_not(df.index.isin(ids_reserve))]
        dfRes = df.loc[ids_reserve]

        models = {}

        models['model1'] = ['pingtime_mean', 'pingtime_max', 'frechet_dist', 'gpsMatchRatio', 'matchGpsRatio',
                            'll_dist_mean', 'll_dist_min', 'll_topol_mean', 'll_topol_min', 'll_distratio_mean',
                            'll_distratio_min']  # all variables
        models['model2'] = ['frechet_dist']  # minimalist
        models['model3'] = ['frechet_dist', 'll_topol_min']
        modelToUse = 'model3'  # this is the one that we choose, and we will save the fit
        for model in ['model1', 'model2', 'model3']:
            varsToUse = models[model]

            formula = 'match_good ~ ' + '+'.join(varsToUse)
            if model == modelToUse: print('\nThis is the preferred model.')

            mask = np.all([pd.notnull(dfEst[var]) for var in varsToUse+['match_good']], axis=0)
            maskRes = np.all([pd.notnull(dfRes[var]) for var in varsToUse+['match_good']], axis=0)
            rdata = dfEst[mask].to_records()

            logitmodel = ro.r.glm(formula, data=rdata, family=ro.r('binomial'))
            fit = ro.r.summary(logitmodel)

            self.writeLog(formula)
            self.writeLog(str(fit.rx2('coefficients'))+'\n')
            self.writeLog('N: %d' % (len(rdata)))
            self.writeLog('AIC: %.2f' % (fit.rx2('aic')[0]))

            dfEst.loc[mask, 'pr_good'] = ro.r.predict(logitmodel, type='response')
            dfEst.loc[:, 'pred_good'] = dfEst.pr_good >= threshold_score
            dfRes.loc[maskRes, 'pr_good'] = ro.r.predict(logitmodel, newdata=dfRes[maskRes].to_records(), type='response')
            dfRes.loc[:, 'pred_good'] = dfRes.pr_good >= threshold_score

            xtabEst = pd.crosstab(dfEst[mask].match_good.astype(bool), dfEst[mask].pred_good, rownames=['Observed'], colnames=['Predicted (regression)'])
            self.writeLog(xtabEst.to_string()+'\n')
            self.writeLog('Hit rate: %.2f' % ((xtabEst[0][0]+xtabEst[1][1]) * 1. / xtabEst.sum().sum()))

            xtabRes = pd.crosstab(dfRes[maskRes].match_good.astype(bool), dfRes[maskRes].pred_good, rownames=['Observed'], colnames=['Predicted (regression)'])
            self.writeLog(xtabRes.to_string()+'\n')
            self.writeLog('Hit rate: %.2f' % ((xtabRes[0][0]+xtabRes[1][1]) * 1. / xtabRes.sum().sum()))

            if model == modelToUse:
                self.model_boxplots(dfEst, dfRes)

            if model == modelToUse:
                self.writeLog('Saving %s to %s' % (model, self.outLogitFn+'\n'))
                with open(self.outLogitFn, "wb") as output_file:
                    cPickle.dump(logitmodel, output_file)
                # save coefficients as text
                with open(self.outLogitCoeffsFn, 'w') as output_file:
                    for ii in range(fit.rx2('coefficients').nrow):
                        output_file.write(fit.rx2('coefficients').rownames[ii]+'\t'+str(fit.rx2('coefficients')[ii])+'\n')

            self.writeLog('\nFalse negatives:')
            self.writeLog(dfEst.loc[(dfEst.match_good == True) & (dfEst.pred_good == 0)].pr_good)
            self.writeLog(dfRes.loc[(dfRes.match_good == True) & (dfRes.pred_good == 0)].pr_good)
            self.writeLog('\nFalse positives:')
            self.writeLog(dfEst.loc[(dfEst.match_good == False) & (dfEst.pred_good == 1)].pr_good)
            self.writeLog(dfRes.loc[(dfRes.match_good == False) & (dfRes.pred_good == 1)].pr_good)

    def predictQuality(self, table=None):
        """Uses the previously estimated model to populate pr_good scores in postgres"""
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri
        import cPickle
        rpy2.robjects.numpy2ri.activate()

        if table is None: table = self.traceTable

        print('Loading original fit')
        with open(self.outLogitFn, "rb") as input_file:
            logitmodel = cPickle.load(input_file)
        fit = ro.r.summary(logitmodel)
        varsToUse = list(fit.rx2('coefficients').rownames)[1:]

        df = self.getPostgresData(table=table, estSample=False)

        mask = np.all([pd.notnull(df[var]) for var in varsToUse], axis=0)

        df.loc[mask, 'pr_good'] = ro.r.predict(logitmodel, newdata=df[mask][varsToUse].to_records(), type='response')  # Convert to record array to use in R

        # Write back to the database
        print('Writing fits to temp table')
        engine = mmt.getPgEngine(pgLogin=pgInfo)

        df.pr_good.to_sql('tmpimport', engine, if_exists='replace')
        self.db.fix_permissions_of_new_table('tmpimport') 
        print('Merging and cleaning up')
        self.db.execute('ALTER TABLE %s DROP COLUMN IF EXISTS pr_good;' % table)
        self.db.execute('ALTER TABLE %s ADD COLUMN pr_good real;' % table)
        self.db.execute('UPDATE %(table)s t1 SET pr_good = t2.pr_good FROM tmpimport AS t2 WHERE t1.%(idName)s=t2.%(idName)s;' % dict(self.cmdDict, **{'table': table}))
        self.db.execute('DROP TABLE tmpimport')

        return

    def scatterplots(self, df):
        """scatterplots of relationships"""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(5, 2, figsize=(6.5, 9))
        df['log_pingtime_max'] = df.pingtime_max.apply(np.log)
        df['log_pingtime_mean'] = df.pingtime_mean.apply(np.log)

        yvars = ['log_pingtime_max', 'frechet_dist', 'gpsMatchRatio',
                 'matchGpsRatio', 'll_dist_mean', 'll_dist_min',
                 'll_topol_mean', 'll_topol_min', 'll_distratio_mean',
                 'll_distratio_min']
        for ii in range(0, 10):
            row = int(math.floor(ii/2.))
            col = ii % 2
            if ii >= len(yvars): break

            df[df.match_good == 1].plot.scatter('log_pingtime_mean', yvars[ii], ax=axes[row, col], color='g', s=2)
            df[df.match_good == 0].plot.scatter('log_pingtime_mean', yvars[ii], ax=axes[row, col], color='r', s=2)
        plt.tight_layout()
        fig.savefig(self.path+'pr_good_scatters.pdf')

    def boxplots(self, df):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(6.5, 9))
        df['log_pingtime_max'] = df.pingtime_max.apply(np.log)
        df['log_pingtime_mean'] = df.pingtime_mean.apply(np.log)

        yvars = ['log_pingtime_mean', 'log_pingtime_max', 'frechet_dist',
                 'gpsMatchRatio', 'matchGpsRatio', 'gpsMatchRatio_highfreq',
                 'll_dist_mean', 'll_dist_min', 'll_topol_mean', 'll_topol_min',
                 'll_distratio_mean', 'll_distratio_min']
        for ii in range(0, 12):
            row = int(math.floor(ii/3.))
            col = ii % 3
            if ii >= len(yvars): break

            df.boxplot(column=yvars[ii], by='match_good', ax=axes[row, col])
        plt.tight_layout()
        fig.suptitle('')
        fig.savefig(self.path+'boxplots.pdf')

    def model_boxplots(self, df1, df2):
        """Boxplots of predicted probability vs actual"""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(4, 2.5))
        df1.boxplot('pr_good', by=['match_good'], ax=axes[0])
        df2.boxplot('pr_good', by=['match_good'], ax=axes[1])
        axes[0].set_title('Estimation sample', fontsize=10)
        axes[1].set_title('Validation sample', fontsize=10)
        axes[0].set_ylabel('Probability [match is good]', fontsize=10)
        axes[1].set_yticklabels([])
        for ax in axes:
            ax.set_xlabel('')
            ax.set_xticklabels(['Fail', 'Success'])
            ax.set_ylim(-0.01, 1.01)
        fig.text(0.4, 0.02, 'Observed quality of match', fontsize=10)
        fig.suptitle('')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(self.path+'modelfit_boxplots.pdf')

    def writeLog(self, txt):
        print(txt)
        currentTime = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
        with open(self.outLogFn, 'a') as outLogF:
            outLogF.writelines(currentTime+':\t: '+str(txt)+'\n')
