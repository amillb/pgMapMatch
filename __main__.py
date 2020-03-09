# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
#########################################################
### pgMapMatch                                        ###
### Map-matching of GPS traces, built using pgrouting ###
#########################################################

See Millard-Ball, Adam; Hampshire, Robert; and Weinberger, Rachel, "Map-matching poor-quality GPS data in urban environments: The pgMapMatch package."
MIT License applies: https://opensource.org/licenses/MIT

Usage:
  pgMapMatch gpx <gpxtrace_file>... --streets=<postgres_streets_table> [--output=<output_filename>]
  pgMapMatch pg <pgtable> <idname> <geomname> --streets=<postgres_streets_table> [options]
  pgMapMatch test

Arguments:
  <gpxtrace_file>                    Name(s) of GPX file(s) to match, when gpx mode is invoked
  <pgtable>                          Name of PostgreSQL table with GPS traces, when pg mode is invoked
  <idname>                           Name of PostgreSQL column with unique id for GPS traces, when pg mode is invoked
  <geomname>                         Name of PostgreSQL column with geometry of GPS traces, when pg mode is invoked
  <postgres_streets_table>           Name of PostgreSQL table of street edges

Options:
  -h --help                          Show this screen
  --ids=<id...>                      Only match specified trace ids in the table (comma separated)
  --write_edges                      Write array of edges to postgres (column will be called edge_ids)
  --write_geom                       Write matched line geometry to postgres  (column will be called matched_line)
  --write_score                      Write match score (i.e. probability that match is good) to postgres (column will be called match_score)
  --clean_geom=<column_name>         Write cleaned geometry (i.e. without ignored points) to postgres
  --output=<output_filename>         Write array of edges, matched line WKT and match score to output_filename. Overwrite if the file already exists
"""

import docopt
import os
import sys
from pgMapMatch import mapMatcher


def write_output(id, mm, fn):
    with open(fn, 'a') as f:
        if mm.matchStatus == 0:
            f.write('\t'.join([str(id), str(mm.bestRoute),
                    str(mm.getMatchAsWKT()),
                    str(mm.getMatchScore())])+'\n')
        else:
            f.write(str(id)+'\n')


def reportMatch(id, mm):
    if mm.matchStatus == 0:
        print('Matched edges for %s: %s' % (id, mm.bestRoute))
    else:
        print('Failed to match %s' % id)


args = docopt.docopt(__doc__)

if args['test']:
    import tests
    tests.test_mapmatch('sf_streets', verbose=False).testall()
    sys.exit(0)

if args['--output']:
    with open(args['--output'], 'w') as f:
        f.write('id\tedge_ids\tmatched_line\tmatch_score\n')

if args['gpx']:
    if os.path.isdir(args['<gpxtrace_file>'][0]):
        if len(args['<gpxtrace_file>']) > 1:
            raise Exception('More than one directory of GPX files specified. Can only match one at a time.')
        print('Matching directory %s' % args['<gpxtrace_file>'][0])
        gpxFiles = [args['<gpxtrace_file>'][0]+'/'+gpx_file
                    for gpx_file in os.listdir(args['<gpxtrace_file>'][0])
                    if gpx_file.endswith('.gpx') and not(gpx_file.endswith('gpx.res.gpx'))]
    else:
        gpxFiles = args['<gpxtrace_file>']

    mm = mapMatcher(args['--streets'])

    for gpx_file in gpxFiles:
        mm.matchGPXTrace(gpx_file)
        reportMatch(gpx_file, mm)
        if args['--output']:
            write_output(gpx_file, mm, args['--output'])

if args['pg']:
    if args['--write_score'] and not args['--write_geom']:
        raise Exception('Sorry. Cannot write match score without also writing match line geometry.')

    cleanedGeomName = None
    mm = mapMatcher(args['--streets'], args['<pgtable>'], args['<idname>'], args['<geomname>'], cleanedGeomName=args['--clean_geom'])
    if args['--ids']:
        ids = args['--ids'].split(',')
    else:
        ids = mm.db.execfetch('SELECT %s FROM %s;' % (args['<idname>'], args['<pgtable>']))
        ids = sorted([ii[0] for ii in ids])

    for id in ids:
        mm.matchPostgresTrace(id)
        reportMatch(id, mm)
        if mm.matchStatus == 0 and (args['--write_edges'] or args['--write_geom'] or args['--write_score']):
            mm.writeMatchToPostgres(writeEdgeIds=args['--write_edges'], writeGeom=args['--write_geom'], writeMatchScore=args['--write_score'])
        if args['--output']:
            write_output(id, mm, args['--output'])
