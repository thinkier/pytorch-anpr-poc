FROM ultralytics/ultralytics

WORKDIR /app/parseq
RUN apt-get install --yes build-essential
RUN pip install dill pydantic lmdb
RUN git clone https://github.com/baudm/parseq .
RUN pip install -r requirements/core.txt

WORKDIR /app
COPY install.py install.py
RUN python3 install.py

COPY . .
RUN mv parseq-*.pt /root/.cache/torch/hub/checkpoints/

CMD python3 main.py
