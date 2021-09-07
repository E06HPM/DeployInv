# FROM python:3.6.7
# LABEL maintainer danny
# ENV PYTHONUNBUFFERED 1

# RUN mkdir /docker_api
# WORKDIR /docker_api
# COPY . /docker_api/



# RUN apt-get update && apt-get -y install apt-transport-https curl

# RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# # install msodbcsql17

# RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list

# RUN exit

# RUN apt-get update

# RUN ACCEPT_EULA=Y apt-get -y install msodbcsql17

# #before install pyodbc , we need unixodbc-dev 

# RUN apt-get install unixodbc-dev
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     unixodbc-dev \
# #     unixodbc \
# #     libpq-dev 

# RUN pip install -r requirements.txt






FROM python:3.6.7
LABEL maintainer danny
ENV PYTHONUNBUFFERED 1
RUN mkdir /docker_api
WORKDIR /docker_api
COPY ./requirements.txt /docker_api/
RUN apt-get update && apt-get install -y curl apt-transport-https
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
RUN apt-get install -y unixodbc-dev
RUN pip install -r requirements.txt
ENTRYPOINT python manage.py runserver 0.0.0.0:7777
