# Use a imagem base Lambda com suporte ao tempo de execução Python
FROM public.ecr.aws/lambda/python:3.9

# Copie o código da função para a imagem
COPY . /var/task

# Instale as dependências do Lambda Layer
RUN pip install -r requirements.txt -t .

# Indique o nome da função handler (arquivo .py) e o nome da função (definido na criação da função Lambda)
CMD ["app.lambda_handler"]
