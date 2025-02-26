import os 
import tarfile
import urllib.request

# Definindo o caminho do dataset
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# download do dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path) # Faz o download do arquivo
    housing_tgz = tarfile.open(tgz_path) # Abre o arquivo
    housing_tgz.extractall(path=housing_path) # Extrai o arquivo
    housing_tgz.close()# Fecha o arquivo

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#preparando os dados

#ver dados faltantes
print(housing.isnull().sum())

# longitude               0
# latitude                0
# housing_median_age      0
# total_rooms             0
# total_bedrooms        207
# population              0
# households              0
# median_income           0
# median_house_value      0
# ocean_proximity         0
# dtype: int64


#substuindo os valores faltantes pela mediana
median = housing["total_bedrooms"].median()

housing["total_bedrooms"].fillna(median, inplace=True)


#verificando se ainda existem valores faltantes
print(housing.isnull().sum())


