FROM freesurfer/synthstrip

RUN pip install nibabel antspyx numpy

COPY preprocess.py /app/preprocess.py

WORKDIR /app

ENTRYPOINT ["python", "/app/preprocess.py"]
