FROM freesurfer/freesurfer:7.4.1

# Install uv and use it to manage a clean Python env separate from FreeSurfer's
# stripped Python 3.8 (which is missing _sqlite3 and other stdlib modules).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv python install 3.12 && \
    uv venv --python 3.12 /opt/venv && \
    uv pip install --python /opt/venv nibabel antspyx numpy templateflow && \
    TEMPLATEFLOW_HOME=/tmp/templateflow python -c "from templateflow import api as tflow; tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='.nii.gz'); tflow.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', extension='.nii.gz')" && \
    mkdir -p /opt/templateflow/tpl-MNI152NLin2009cAsym && \
    cp /tmp/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz /opt/templateflow/tpl-MNI152NLin2009cAsym/ && \
    cp /tmp/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz /opt/templateflow/tpl-MNI152NLin2009cAsym/ && \
    rm -rf /tmp/templateflow

COPY preprocess.py /app/preprocess.py
COPY README.md /app/README.md

WORKDIR /app

ENTRYPOINT ["python", "/app/preprocess.py"]
