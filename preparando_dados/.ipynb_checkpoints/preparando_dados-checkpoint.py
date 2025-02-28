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

housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)



#verificando se ainda existem valores faltantes
print(housing.isnull().sum())

#
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") # criando um objeto imputer

# ocean_proximity é um atributo categórico, vamos transformar em numérico
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

#verificando a mediana de cada atributo
print(imputer.statistics_)
# [ -118.51      34.26      29.      2119.5      433.     1164.       408.
#     3.5409  179700.    ]

#verificando se a mediana está correta
print(housing_num.median().values)
# [-118.51     34.26     29.     2119.5     433.     1164.      408.
#     3.5409 179700.   ]


x = imputer.transform(housing_num) # transformando os valores faltantes pela mediana

housing_tr = pd.DataFrame(x, columns=housing_num.columns, index=housing_num.index) # transformando em dataframe

print(housing_tr.isnull().sum())

#manipulando texto e atributos categóricos

housing_cat = housing[["ocean_proximity"]] # pegando apenas o atributo ocean_proximity

print(housing_cat.head(10)) # verificando os valores de ocean_proximity

from sklearn.preprocessing import OrdinalEncoder 

ordinal_encoder = OrdinalEncoder() # criando um objeto ordinal_encoder
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) # transformando os valores categóricos em numéricos

print(housing_cat_encoded[:10]) # verificando os valores de ocean_proximity

print(ordinal_encoder.categories_) # verificando as categorias

#trasformando os valores categóricos em vetores binários
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder() # criando um objeto cat_encoder
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # transformando os valores categóricos em vetores binários

print(housing_cat_1hot.toarray()) # verificando os valores de ocean_proximity
