# -*- coding: utf-8 -*-

"""
Tools for database connection, etc. for use with map-matching analysis
Adapted from postgres_tools.py by Chris Barrington-Leigh and Adam Millard-Ball
"""
import os
import math
import six
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd


def getPgLogin(user=None, db=None, host=None, requirePassword=False, forceUpdate=False):
    """
    Returns dictionary of credentials to login to postgres
    To change access permissions (e.g. where someone can login from),
       edit /home/projects/var_lib_pgsql_9.3/data/pg_hba.conf
    """
    # see if we already have the information - avoid asking for password again
    if 'pgLogin' in globals() and not forceUpdate:
        return globals()['pgLogin']

    user = six.moves.input('Username for database:') if user is None else user
    db = six.moves.input('Database name:') if db is None else db
    host = 'localhost' if host is None else host

    pgLogin = {'user': user, 'db': db, 'host': host, 'pw': ''}

    if requirePassword:
        import getpass
        pw = getpass.getpass('Enter postgres password for %s: ' % user)
        pgLogin.update({'pw': pw})

    return pgLogin


def getPgEngine(pgLogin=None):
    """
    returns an engine object that connects to postgres
    From the docs: You only need to create the engine once
                   per database you are connecting to
    """
    from sqlalchemy import create_engine
    if pgLogin is None:
        pgLogin = getPgLogin()
    thehost = '' if 'host' not in pgLogin else pgLogin['host']+':5432'
    engine = create_engine('postgresql://%s:%s@%s/%s' % (pgLogin['user'], pgLogin['pw'], thehost, pgLogin['db']))

    return engine


class dbConnection():
    def __init__(self, user=None, db=None, host=None, requirePassword=True,
                 pgLogin=None, schema=None, role=None, curType='default',
                 verbose=True, logger=None):
        """
        This returns a connection to the database.
        If role is not None, new tables will be owned by this role
            (rather than user)
        """
        if pgLogin is not None:
            user = pgLogin['user']
            db = pgLogin['db']
            host = pgLogin['host']
            schema = pgLogin['schema']
            requirePassword = pgLogin['requirePassword']
        self.pgLogin = getPgLogin(user=user, db=db, host=host, requirePassword=requirePassword)
        assert curType in ['DictCursor', 'default']
        self.curType = curType
        if schema is None:
            schema = 'public'
        self.default_schema = schema
        # Connect to database
        coninfo = ' '.join([{'db': 'dbname', 'pw': 'password'}.get(key, key)+' = '+val
                           for key, val in self.pgLogin.items()])
        con = psycopg2.connect(coninfo)
        con.set_isolation_level(0)   # autocommit - see http://stackoverflow.com/questions/1219326/how-do-i-do-database-transactions-with-psycopg2-python-db-api
        self.cursor = None
        self.connection = con
        search_path = [self.default_schema]+['public']
        self.execute('SET search_path = '+','.join(search_path))
        print('SET search_path = '+','.join(search_path))
        if role is not None:
            # ensures that new tables are owned by the group
            self.execute('''SET role %s;''' % role)
        assert not self.cursor.closed
        self.verbose = verbose
        self.logger = logger

    def cur(self):
        """ Create a new cursor. This can be done frequently.
        The cursor is a lightweight object, and can be deleted after each use.
        That might help with postgres memory use.

        The latest cursor is always available as self.cursor but the only
          intended use is outside calls of the form:
            thisobject.cur().execute('pg command')
        """
        if self.curType == 'DictCursor':
            # So everything coming from the database will be in Python dict format.
            cur = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        elif self.curType == 'default':  # will return data as tuples - easier to convert to pandas
            cur = self.connection.cursor()
        del self.cursor
        self.cursor = cur
        return cur

    def execute(self, cmd):
        if self.cursor is None:
            self.cur()
        return self.cursor.execute(cmd)

    def fetchall(self):
        return self.cursor.fetchall()

    def execfetch(self, cmd):
        """Execute an SQL command and fetch (fetchall) the result"""
        self.execute(cmd)
        return self.fetchall()

    def execfetchDf(self, cmd):
        """Execute an SQL command and return the result as a pandas dataframe"""
        import pandas as pd
        try:
            return pd.read_sql_query(cmd, self.connection)
        except AttributeError:  # probably an older version of pandas
            assert not getattr(pd, 'read_sql_table', None)
            tmpcur = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            tmpcur.execute(cmd)
            colNames = [desc[0] for desc in tmpcur.description]
            results = tmpcur.fetchall()
            return pd.DataFrame([[row[col] for col in colNames] for row in results], columns=colNames)

    def report(self, ss):
        """
        A wrapper for "print".
        In verbose mode, it prints comments and SQL commands.
        If a logger has been provided, it calls logger.write() instead of print().
        """
        if not self.verbose: return
        if self.logger:
            self.logger.write(ss)
        else:
            print(ss)

    def df2db(self, df, tablename, if_exists='replace', index=False, chunksize=100000):
        """
        Create a table using tabular data in a Pandas dataframe.
        """
        engine = getPgEngine(pgLogin=self.pgLogin)
        df.to_sql(tablename, engine, schema=self.default_schema, if_exists=if_exists, index=index, chunksize=chunksize)
        self.fix_permissions_of_new_table(tablename)

    def fix_permissions_of_new_table(self, tablename):
        """
        Newly created tables have the owner of the person running the code, whereas we want them to be owned by parkingusers.
        This cannot be done from the cursor without changing the role, so it's easier to do it from a separate psql call.
        """
        # We could do this with self.cursor, too;
        # but we'd have to change the role back to a person from osmusers.
        os.system('psql -c "ALTER TABLE '+tablename+' OWNER to parkingusers" %s ' % (self.psql_command_line_flags()))

    def psql_command_line_flags(self):
        """ Return a string to be used in a psql command line to identify host, user, and database.
        Not yet tested with passwords
        """
        outs = '' if 'host' not in self.pgLogin else '-h %s ' % self.pgLogin['host']
        outs += ' -d %s -U %s ' % (self.pgLogin['db'], self.pgLogin['user'])
        return outs

    def list_columns_in_table(self, table):
        cmd = """SELECT column_name FROM information_schema.columns
                 WHERE table_schema = '%s' AND table_name='%s';""" % (self.default_schema, table)
        colNames = self.execfetch(cmd)
        return [cn[0] for cn in colNames]

    def addColumns(self, columns, table, skipIfExists=False, dropOld=False):
        """Columns should be a tuple of (name, type) or a list of tuples"""
        if isinstance(columns, tuple):
            columns = [columns]
        if skipIfExists:
            columns = [cc for cc in columns if not(cc[0] in self.list_columns_in_table(table))]
            if not columns: return  # all exist
        assert isinstance(columns, list) and all([isinstance(cc, tuple) for cc in columns])
        if dropOld:
            self.execute('ALTER TABLE '+table + ','.join([' DROP COLUMN IF EXISTS '+cc[0] for cc in columns]))
        self.execute('ALTER TABLE '+table + ','.join([' ADD COLUMN '+cc[0]+' '+cc[1] for cc in columns]))

    def merge_table_into_table(self, from_table, targettablename, commoncolumns,
                               drop_old=True, drop_incoming_duplicates=False,
                               recreate_indices=True, forceUpdate=True):
        """
        To merge two tables (with different but overlapping columns), the efficient method is to delete the original tables and create a new one.
        The from_table should contain no columns in common with targettable, except for the matching column(s), "commoncolumns".   If it does, an error will be thrown (to do) unless forceUpdate is True, in which case the common columns will simply be dropped! (as opposed to a nicer merge.

        Often the commoncolumns is the primary key or should be unique.
        In this case, drop_incoming_duplicates will use only one copy of each row with a distinct key value in from_table.

        Note that when one renames tables, one does not rename their indices automatically.

        This is not yet maybe tested properly using multiple (comma-separated values in)  "commoncolumns".
        """
        if isinstance(commoncolumns, list):
            commoncolumns = ','.join(commoncolumns)

        # Check we don't have a conflict:
        ftc = self.list_columns_in_table(from_table)
        ttc = self.list_columns_in_table(targettablename)
        if [cc for cc in list(set(ftc) & set(ttc)) if cc not in commoncolumns.split(',')]:
                self.report('  Conflict in columns. We could implement something fancier to stitch together the changed and unchanged rows... or just stop here')
                self.report('  Maybe you need to force-update your run so that the target table is dropped/recreated?  Or maybe something else is wrong. It looks like columns you are insterting, other than the index column, already exist in the target.')
                self.report('         '+from_table+':'+str(ftc)+'\n            '+targettablename+':'+str(ttc))
                thats_a_problem
        if drop_incoming_duplicates:
                from_table = '(SELECT DISTINCT ON ('+commoncolumns+') * from '+from_table+') as distinct_from'
        if recreate_indices:
            indices = self.execfetch("SELECT indexdef FROM pg_indexes WHERE tablename='%s'" % (targettablename))

        cmd = """CREATE TABLE new_"""+targettablename+""" AS (
                    SELECT * FROM """+targettablename+""" LEFT JOIN """+from_table+""" USING ("""+commoncolumns+"""));"""
        self.execute(cmd)
        self.execute(' DROP TABLE IF EXISTS old_'+targettablename+';')
        self.execute(' ALTER TABLE '+targettablename+' RENAME TO old_'+targettablename+';')
        if drop_old:
            self.execute(' DROP TABLE old_'+targettablename+';')
        self.execute(' ALTER TABLE new_'+targettablename+' RENAME TO  '+targettablename+';')

        if recreate_indices:
            for idxCmd in indices: self.execute(idxCmd[0])

    def update_table_from_array(self, data, targettablename, joinOnIndices=False,
                                temptablename=None, if_exists='replace',
                                preserve_uppercase=False, drop_incoming_duplicates=False):
        """
        Take a tab-separated file, or a Pandas dataframe,
        and use its first column to join the remaining columns
        (which should not exist in the targettablename)
        to update the matching target table rows.

        data can be a df or a filename
        If joinOnIndices is True, will join on a (Multi)Index. Otherwise, join on the first column.

        Comments:
        Uses sqlalchemy to handle buffering/inserts to SQL.

        By default, pandas' to_sql uses quotes around any column names if they (any) contain capital letters. That can mess up a lot of stuff. So by default, this function will convert column names to lowercase.
        To do: Make sure that the data do not contain duplicates of the intended primary key.
        """
        engine = getPgEngine(pgLogin=self.pgLogin)
        ntmp = temptablename if temptablename else 'tmp_for_insertion_'+targettablename
        if isinstance(data, str):
                self.report(' Loading '+data)
                df = pd.read_table(data)
        else:
                df = data
        df.columns = df.columns.map(lambda ss: ss.lower())
        self.report(' Inserting (from df) to '+ntmp)
        if joinOnIndices:
            df.index.names = [ss.lower() for ss in df.index.names]
            indexcolumn = df.index.names
            df.to_sql(ntmp, engine, schema=self.default_schema, if_exists=if_exists, index=True, chunksize=100000)
        else:
            indexcolumn = df.columns[0]
            df.to_sql(ntmp, engine, schema=self.default_schema, if_exists=if_exists, index=False, chunksize=100000)
        self.fix_permissions_of_new_table(ntmp)
        self.report(' Merging '+ntmp+' into '+targettablename)
        self.merge_table_into_table(ntmp, targettablename, indexcolumn, drop_incoming_duplicates=drop_incoming_duplicates)

    def create_indices(self, table, key=None, non_unique_columns=None,
                       geom=False, skip_vacuum=False, forceUpdate=True):
        """Basic use is to specify the index column, which is a unique index, as key.
        One can also or instead specify other non-unique columns.
        Special options for geom create spatial indices"""

        # You can create multicolumn indices, e.g. id0,id1,id2,
        # but it's subtle as to when one method will be fastest.
        strlu = dict(table=table, pkey=key, geom='geom' if geom is True else geom,
                     IFNOTEXISTS=' IF NOT EXISTS '*(not forceUpdate))
        if key:
            self.execute('CREATE UNIQUE    INDEX  %(IFNOTEXISTS)s %(table)s_idx_%(pkey)s ON %(table)s (%(pkey)s);' % strlu)
        if non_unique_columns:
            for col in non_unique_columns:
                self.execute('CREATE  INDEX %(IFNOTEXISTS)s ' % strlu+table+'_idx_'+col+' ON '+table+' ('+col+') ;')
        if geom:
            self.execute('CREATE INDEX  %(IFNOTEXISTS)s %(table)s_%(geom)s_spat_idx  ON %(table)s  USING gist  (%(geom)s);' % strlu)
        if not skip_vacuum:
            self.execute('VACUUM  ANALYZE '+table)

    def drop_indices(self, table, key=None, non_unique_columns=None, geom=False):
        """ Drop tables probably created by the create_indices method.
        Dropping indices can be important before doing a big update to a table.

        See create_indices()
        """
        self.report('Dropping indices if exist...')
        strlu = dict(table=table, pkey=key)
        if key:
            self.execute('DROP INDEX IF EXISTS  %(table)s_idx_%(pkey)s ;' % strlu)
        if geom:
            self.execute('DROP INDEX IF EXISTS  %(table)s_spat_idx ;' % strlu)

    def get_indices(self, table=None, contains=None):
        """Get all indices, optionally restricted to those on a specified table or contains a certain string
        Returns a list of tuples of (schema, table, index, indexdef)"""
        tableTxt = '' if table is None else " AND t.relname='%s' " % table
        containsTxt = '' if contains is None else " AND c.relname LIKE '%%%s%%' " % contains
        cmd = '''SELECT n.nspname  as "schema", t.relname  as "table", c.relname  as "index", pg_get_indexdef(indexrelid) as "def"
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n ON n.oid    = c.relnamespace
                    JOIN pg_catalog.pg_index i ON i.indexrelid = c.oid
                    JOIN pg_catalog.pg_class t ON i.indrelid   = t.oid
                    WHERE c.relkind = 'i'
                        and n.nspname not in ('pg_catalog', 'pg_toast')
                        and pg_catalog.pg_table_is_visible(c.oid) %s %s;''' % (tableTxt, containsTxt)
        return self.execfetch(cmd)

    def list_tables(self):
        """ List all tables in the schema. Views are not tables, so don't list them. """
        cmd = "SELECT table_name FROM information_schema.tables WHERE table_type != 'VIEW' and table_schema = '"+self.default_schema+"';"
        return [tt[0] for tt in self.execfetch(cmd)]


"""
Functions to calculate Frechet Distance
https://www.snip2code.com/Snippet/76076/Fr-chet-Distance-in-Python
"""


def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i-1, j, P, Q), _c(ca, i-1, j-1, P, Q),
                           _c(ca, i, j-1, P, Q)),
                       euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
This is a placeholder pending addition of ST_FrechetDistance() to PostGIS (scheduled for 2.4.0)
"""


def frechetDist(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P)-1, len(Q)-1, P, Q)
