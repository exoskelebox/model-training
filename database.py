import psycopg2
from configparser import ConfigParser
import os

#### SHOULD ONLY EXIST UNTIL DATASET IS PUBLISHED #####


class Database():
    def __init__(self, config_file='config.ini', section='postgresql'):
        try:
            parser = ConfigParser()
            parser.read(os.path.join(os.path.dirname(__file__), config_file))
            config = {option: parser.get(section, option) for option in parser.options(
                section) if parser.has_section(section)}
            self.conn = psycopg2.connect(**config)
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL:", error)
        else:
            print(self.conn)
            print("PostgreSQL connection opened")

    def cursor(self):
        return self.conn.cursor()

    def query(self, query):
        try:
            cur = self.cursor()
            cur.execute(query)
            return cur
        except (Exception, psycopg2.Error) as error:
            print("Error while executing PostgreSQL query:", error)

    def table_query(self, table, query='') -> psycopg2.extensions.cursor:
        psql = f'SELECT * FROM {table} {query};'
        return self.query(psql)

    def close(self):
        try:
            self.conn.close()
        except (Exception, psycopg2.Error) as error:
            print("Error while closing PostgreSQL connection:", error)
        else:
            print("PostgreSQL connection closed")
