﻿# Aprendizado de Marquina - Estudo Básico

Este repositório é utilizado para armazenar codigos usados durante estudo de aprendizagem de maquina. Aqui você encontrará um resumo da minha  trajetoria como estudante de Machine Learning, utilizando ferramentas como Python, Pandas, Matplotlib, scikit-Learn, Keras, TensorFLow entre outras bibliotecas populares.

## Objetivo

O objetivo deste repositório é guardar os códigos utilizados durante o estudo. 


## Estrutura do Projeto

# 📂 **Aprendizagem_de_Maquina**  
├── 📂 **datasets**  
│   ├── 📄 housing.csv  
├── 📂 **preparando_dados**  
│   ├── 📄 preparando_dados.py  
├── 📄 modelo.py  
├── 📄 README.md  


## Descrição dos Arquivos

- **📂 datasets/**: Contém os arquivos de dados usados para a análise.
  - **📄 housing.csv**: Os dados contêm informações do censo da Califórnia de 1990.

- **📂 preparando_dados/**: Contém os arquivos responsáveis pela preparação dos dados do projeto (limpeza, rotulagem, validação etc.).
  - **📄 preparando_dados.py**: Código usado para preparação dos dados.
  - **📄 modelo.py**: Script principal para o estudo.

- **📄 README.md**: Documentação do projeto.

## Sobre o Conjunto de Dados

#### Contexto
Este é o conjunto de dados utilizado no segundo capítulo do livro recente de Aurélien Géron, Hands-On Machine Learning with Scikit-Learn and TensorFlow. Ele serve como uma excelente introdução à implementação de algoritmos de aprendizado de máquina, pois exige apenas uma limpeza básica dos dados, possui uma lista de variáveis de fácil compreensão e tem um tamanho ideal – nem muito pequeno a ponto de ser simplista, nem tão grande a ponto de ser excessivamente complicado.

Os dados contêm informações do censo da Califórnia de 1990. Embora este conjunto de dados não ajude a prever os preços atuais das casas, como o Zillow Zestimate dataset, ele fornece um material acessível para ensinar os conceitos básicos de aprendizado de máquina.

#### Conteúdo
Os dados referem-se às casas encontradas em um determinado distrito da Califórnia, incluindo algumas estatísticas resumidas com base nos dados do censo de 1990. É importante notar que os dados não estão limpos, portanto, algumas etapas de pré-processamento são necessárias!

As colunas do conjunto de dados são as seguintes (seus nomes são autoexplicativos):

* longitude (longitude)
* latitude (latitude)
* housing_median_age (idade mediana das habitações)
* total_rooms (total de quartos)
* total_bedrooms (total de dormitórios)
* population (população)
* households (número de domicílios)
* median_income (renda mediana)
* median_house_value (valor mediano das casas)
* ocean_proximity (proximidade do oceano)

## Vamos começar ! 

[Preparando Dados](./preparando_dados/README.md)
