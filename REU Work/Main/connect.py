import pandas as pd
import psycopg2

def gen_df(rows, cols, tablename, conn):
    if isinstance(cols, list):
        return pd.DataFrame(data=rows, columns = cols)
    else:
        temp = []
        x = cols.split(",")
        for i in x:
            temp.append(i.split()[-1])
        return pd.DataFrame(data=rows, columns = temp)


def get(table_name, cols='*', where=None, modifier=None, wantrows = False):
    q = df = None
    
    try:
        connection = psycopg2.connect(
        host='localhost',  # host on which the database is running
        database='REU',  # name of the database to connect to
        user='postgres',  # username to connect with
        password='8538'  # insert your password here
    )
    except: 
        print('Connection failed...')
        pass

    else:
        cursor = connection.cursor()
        if where == None and modifier == None:
            q = f'SELECT {cols} FROM {table_name};'
        elif where != None and modifier == None:
            q = f'SELECT {cols} FROM {table_name} where {where};'
        elif where == None and modifier != None:
            q = f'SELECT {cols} FROM {table_name} {modifier};'
        else:
            q = f'SELECT {cols} FROM {table_name} WHERE {where} {modifier};'

        print(f'Firing ...{q}')
        cursor.execute(q)
        rows = cursor.fetchall()
        df = gen_df(rows, cols, table_name, connection)
        connection.close()
        if wantrows:
            return df, len(rows) # can take out length to get all data in tuples
        else:
            return df

        

