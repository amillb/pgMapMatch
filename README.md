# pgMapMatch
map-matching of GPS traces, built using pgrouting

For details of the algorithm, see [Millard-Ball, Adam; Hampshire, Robert and Weinberger, Rachel (2019), "Map-matching poor-quality GPS data in urban environments: The pgMapMatch package." *Transportation Planning and Technology.*](https://doi.org/10.1080/03081060.2019.1622249) (A non-paywalled pre-print is available [here](https://people.ucsc.edu/~adammb/publications.html).)

For questions and feedback, please contact [Adam Millard-Ball](https://people.ucsc.edu/~adammb/).

## Getting started ##
1. Download this repository using `git clone https://github.com/amillb/pgMapMatch.git`

2. You will need to load a table of street edges into a PostgreSQL database. The easiest way to do this is:
  * Download an extract from [OpenStreetMap](http://www.openstreetmap.org/), such as those produced for metropolitan areas by [Mapzen](https://mapzen.com/data/metro-extracts/).
  * Use [osm2po](http://osm2po.de) to load the OpenStreetMap data into PostgreSQL. You can use another OSM import tool, but you will need to make sure your street column names match those that are in `config.py` (see Step 3 below).
  * Transform the street geometries to a suitable projection, for example: 

     `ALTER TABLE streets_table ALTER COLUMN geom_way TYPE Geometry(LineString, your_srid) USING ST_Transform(geom_way, your_srid);` 

3. Copy `config_template.py` to `config.py`, and adjust the configuration settings. You will almost certainly need to adjust the postgres login information in the `pgInfo` dictionary. If you don't use osm2po, you may also need to adjust the column names for the streets table specified in `config.py`. The default parameters in `config_template.py` assume your units are in meters, so it will be easiest to use a projection that is also in meters. If you use feet, you will need to change the parameters, and/or risk getting unexpected results.

## Usage ##
You can match traces from GPX files (a `time` field must be included), or a PostgreSQL table. For the postgres option, the traces must be LineStrings with an M coordinate providing the timestamp of each point, and the projection must be the same as the table of streets.

You can call pgMapMatch from the command line. You can match either a GPX file, or a table of GPS traces loaded into PostgreSQL. `python pgMapMatch --help` gives you a list of options. 

For example, to match a GPX file:
```
python pgMapMatch gpx pgMapMatch/testdata/testtrace_36.gpx --streets=sf_streets
```

For example, to match a table of GPS traces (called `gps_trace_table`), and write the matched geometries and match score:
```
python pgMapMatch pg gps_trace_table trace_id trace_geom --streets=sf_streets --write_geom --write_score
```

For a large number of traces, the second option will be much more efficient.

You can also import pgMapMatch into Python, and use the class `mapMatcher()`. After you call `matchGPXTrace()` or `matchPostgresTrace()`, you can access the sequence of edges, matched geometry and match score, and write them to Postgres. For example:
```
import pgMapMatch
your_streets_table = 'sf_streets'   # the name of your Postgres table with streets
mm = pgMapMatch.mapMatcher(your_streets_table)
your_gpx_filename = 'pgMapMatch/testdata/testtrace_36.gpx'
mm.matchGPXTrace(your_gpx_filename)
print(mm.bestRoute)        # returns the sequence of edge ids (based on the id column in the streets table)
print(mm.getMatchAsWKT())  # returns the matched geometry as Well-Known Text
print(mm.getMatchScore())  # returns the match score (probability that the match is good)
```

## Dependencies ##

[PostgreSQL](https://www.postgresql.org/download/), with [PostGIS](https://postgis.net/install/) 2.3+ and [pgrouting](http://pgrouting.org/download.html) 2.4.1+ installed. A local installation is not required; PostgreSQL can run on a remote server. Make sure to update `pgInfo` in `config.py` with the database connection information.

The following Python packages:  
* numpy 1.11.3+  
* scipy 0.19.0+
* pandas 0.19.2+  
* gpxpy 1.1.2+  
* psycopg2 2.5.2+  
* sqlalchemy 1.1.6+  
* docopt 0.6.1+  

Other versions may work, but have not been tested.

You can install all the Python packages with:
`pip install numpy scipy pandas gpxpy psycopg2 sqlalchemy docopt`
