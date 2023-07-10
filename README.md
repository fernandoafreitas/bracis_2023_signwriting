# bracis_2023_signwriting

Passo 1 - Instalar as bibliotecas mencionadas no arquivo requirements.txt

Passo 2 - Gerar os modelos de descrições com base nos arquivos *.spml extraídos do site https://www.signbank.org/signpuddle/ que se encontram na pasta Dicts. O código encontra-se no arquivo Create_dicts.ipynb

Passo 2 - Gerar o tokenizador customizado no arquivo Custom_tokenizer.ipynb

Passo 3.1 - Para treinar a rede DCNN do experimento, utilize o arquivo DCNN_sample.py. Modifique o dicionário "parameters" conforme seu interesse.

Passo 3.2 - Para treinar o modelo Distilber do experimento, utilize o arquivo Distilbert_sample.py

Os resultados estarão na pasta Resultados e os artefatos gerados no passo 1 e 2 estarão na pasta Outputs.