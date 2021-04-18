import pandas as pd

# Mysql tools
from mysql.connector import connect
from sqlalchemy import create_engine

# utilities
from pathlib import Path
import os
import joblib
import kaggle as kg

# directories
data_folder = Path.cwd() / 'datasets'
model_folder = Path.cwd() / 'models'

# credits
db_credits = dict(user=os.environ.get('AWS_DB_USER'),
                  host=os.environ.get('AWS_DB_HOST'),
                  password=os.environ.get('AWS_DB_PASSW'))


def load_sql_data(database: str, table: str, login: dict) -> pd.DataFrame:
    """
    :param database: an existing database name [type str]
    :param table: an existing table name [type str]
    :param login: a dict with user, host, password key to access the database [type dict]
    :return: a pd.Dataframe object obtain by a 'SELECT *' in the table
    """
    query = f'SELECT * FROM {table};'
    login = login | dict(database=database)

    cnx = connect(**login,
                  use_pure=True,
                  use_unicode=True,
                  charset='utf8')

    try:
        data = pd.read_sql(query, cnx)
        cnx.close()
        return data
    except Exception as e:
        cnx.close()
        print(e)


def load_from_kaggle(dataset: str) -> pd.DataFrame:
    """
    :param dataset: an url of a kaggle dataset  [type str]
    :return: a pd.DataFrame if only one file has been downloaded and if its a csv
    """
    if not data_folder.is_dir():
        Path.mkdir(data_folder, exist_ok=True)

    kg.api.authenticate()
    dataset = dataset.lstrip('https://www.kaggle.com/')
    old_files = {file.name for file in data_folder.iterdir()}
    kg.api.dataset_download_files(dataset=dataset, unzip=True, path=data_folder)
    new_files = {file.name for file in data_folder.iterdir()}.difference(old_files)

    if len(new_files) != 1:
        print('Please choose one of this file with `load_csv` helping function:', *list(new_files), sep='\n')
    else:
        return pd.read_csv(data_folder / list(new_files)[0])


def load_csv(filename: str = None, folder=data_folder):
    """
    :param filename: name of the dataset with extension ['type str']
    :param folder: use defaut for conveniency to respect a WD/datasets/datafiles structure
    :return: if filename is None print all available, else return a pd.DataFrame
    """
    if not folder.is_dir():
        Path.mkdir(data_folder, exist_ok=True)
        raise FileNotFoundError('No data folder found. A template has just been created.')

    all_files = [file.name for file in data_folder.iterdir()]
    if not (filename or all_files):
        raise FileNotFoundError('No data file found. Please import one.')

    elif filename in all_files:
        return pd.read_csv(filename)
    else:
        print('PICK ONE OF THESE:\n', *all_files, sep='\n')


def save_data_to_sql(data: pd.DataFrame, database: str, table: str, login: dict, if_exists: str = 'fail') -> None:
    """
    :param data: your dataset [type pd.DataFrame]
    :param database: SQL backup database // MUST BE SET BEFOREHANDE [type:str]
    :param table: the table to use [type:str]
    :param login: a dict with user, host, password key to access the database [type dict]
    :param if_exists: {'fail', 'replace', 'append'} default 'fail'
    :return: None
    """
    login = login | dict(database=database)
    engine = create_engine('mysql+pymysql://{user}:{password}@{host}/{database}'.format(**login))

    with engine.begin() as connection:
        data.to_sql(name=table, con=connection, if_exists=if_exists, index=False)


# WIP
def dump_model(model, name: str) -> None:
    """
    function still not fully tested
    :param model: a model fully set (example SVM(**best_parameters))
    :param name: the filename without extension
    :return: None
    """
    Path.mkdir(model_folder, exist_ok=True)
    joblib.dump(model, model_folder / f'{name}.pkl')
    return None


# WIP
def load_model(model=None):
    """
    :param model: the model you want to retrieve
    :return: print the list of model it not arg passed, else the model.
    """
    Path.mkdir(model_folder, exist_ok=True)
    all_models = [file.name for file in model_folder.iterdir()]

    if model:
        try:
            loaded = joblib.load(model_folder / f'{model}.pkl')
        except FileNotFoundError as e:
            print('This model seems absent, here the list of all models:\n',
                  *all_models, e, sep='\n')
            return None
        return loaded
    else:
        print('Here the list of all models:\n',
              *all_models, sep='\n')
        return None
