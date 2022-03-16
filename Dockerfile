FROM tensorflow/tensorflow:nightly-gpu

COPY req.txt /req.txt

RUN pip install -r req.txt


