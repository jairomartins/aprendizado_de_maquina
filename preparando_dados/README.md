# Preparando os dados para o Algoritmo de Aprendizado de Maquina

A preparação de dados é o processo de preparar os dados brutos para adequá-los as etapas seguintes de processamento e analise. 

Segundo a AWS, a preparação de dados pode consumir até 80% do tempo investido em um projeto de ML(Machine Learing).

Para entender melhor sobre **preparação de dados**, consulte este artigo da AWS:  
[O que é preparação de dados?](https://aws.amazon.com/pt/what-is/data-preparation/)


A preparação de dados segue uma série de etapas que começa com a coleta de dados, seguida pela limpeza, rotulagem, validação e visualização.


### Limpando os dados
A limpeza de dados corrige erros e preenche dados ausentes como uma etapa de garantia da qualidade dos dados. Após a limpeza dos dados, será necessário transformá-los em um formato consistente e passível de leitura. Esse processo pode incluir a alteração de formatos de campo como datas e moeda, a modificação de convenções de nomenclatura e a correção de valores e unidades de medida para promover consistência.


A maioria dos algoritmos de ML, não funcionam com caracteristicas faltantes. Como verificar isso no nosso projeto de estudo ? veja a seguir! 

```bash 
#verificando dados nulos
>>> print(housing.isnull().sum())

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


```

É possivel notar que faltam alguns valores para o atributo *'total_bedrooms'*
conseguimos contornar a situação de 3 formas:

* Ignorar as regioes correspondentes
* Ignorar o atributo 
* Definir valores para algum valor ( zero, media, mediana)

