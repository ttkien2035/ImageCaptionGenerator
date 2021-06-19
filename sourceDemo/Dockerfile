FROM python:3.9.5
RUN mkdir /mainapp
WORKDIR /mainapp
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "wsgi.py"]