# -*- coding: utf-8 -*-


"""Configuration file for pgMapMatch
Make changes here to:
1. Set up your PostgreSQL database, schema, host and user names
2. Set the parameters of the map-matching likelihood functions
3. Provide details on the column names of the streets table
   (if you are not using osm2po)

Save this file as `config.py`.
`config_template.py` will be used only if `config.py` is not found.
"""

# postgres connection information
pgInfo = {'db': 'your_database_name',
          'schema': 'your_schema_name',    # schema with GPS traces and streets table
          'user': 'your_postgres_username',
          'host': 'localhost_or_IP_address',
          'requirePassword': True  # Prompt for password? Normally, False for localhost
          }

# how many nodes pgrouting can route at once.
# If you have lots of memory, increase. If you get a memory error, reduce
maxNodes = 250

# All these parameters assume meters, but must match the units of your projection
# State Plane meters will normally work best
weight_1stlast = 6      # for the first and last node, how much distance weight is increased by. This helps to ensure the match is closer to and from the start and endpoint
gpsError = 50           # max distance that streets can be from GPS point to be considered
gpsError_fway = 70      # same, but for freeway (kmh>100) - because these roads are much wider
maxSpeed = 120           # speed threshold for deleting pings (kmh) in traceCleaner

sigma_z = 10.0          # std dev parameter for geometric likelihood. Relates to GPS noise. Newson and Kummel 2009 say 4.0
sigma_t = 0.3           # std dev parameter for temporal likelihood
sigma_topol = 0.6       # std dev parameter for topological likelihood
temporal_scale = 0.55   # scale parameter for temporal likelihood
temporal_weight = 1.7   # how more more the temporal likelihood is weighted relative to the distance likelihood score
topol_weight = 1.7      # how more more the topological likelihood is weighted relative to the distance likelihood score
skip_penalty = 3        # penalty for skipping a point is temporalLL(skip_penalty)
max_skip = 4            # maximum number of points to skip. Reducing this will improve performance
uturnCost = None        # if None, use the default (the average of the median cost and reverse cost in the edges)
allowFinalUturn = True  # if True, allow a U-turn on the final edge

# column identifiers for the PostGIS table of streets
# the default values here are compatible with osm2po
streetIdCol = 'id'          # unique id for street edge (i.e. segment or block)
streetGeomCol = 'geom_way'  # geometry column (LineString) for street edge
startNodeCol = 'source'     # id of node at which the street edge starts
endNodeCol = 'target'       # id of node at which the street edge ends
travelCostCol = 'cost'      # generalized cost to go from startNode to endNode
travelCostReverseCol = 'reverse_cost'  # generalized cost to go from endNode to startNode. Can be same as travelCostCol if you have no one-way streets
streetLengthCol = 'km'      # length of street, in km
speedLimitCol = 'kmh'       # speed limit on street, in km per hour

# SQL-compliant query that identifies freeways (with the higher gps error tolerance)
fwayQuery = 'clazz<15 OR kmh>=100'
# comma-separated list of columns that are needed in fwayQuery, but are not listed above
fwayCols = 'clazz'
