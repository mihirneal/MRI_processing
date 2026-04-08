FROM freesurfer/synthstrip

RUN pip install nibabel antspyx numpy templateflow && \
    TEMPLATEFLOW_HOME=/tmp/templateflow python -c "from templateflow import api as tflow; tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='.nii.gz'); tflow.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', extension='.nii.gz')" && \
    mkdir -p /opt/templateflow/tpl-MNI152NLin2009cAsym && \
    cp /tmp/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz /opt/templateflow/tpl-MNI152NLin2009cAsym/ && \
    cp /tmp/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz /opt/templateflow/tpl-MNI152NLin2009cAsym/ && \
    rm -rf /tmp/templateflow

COPY preprocess.py /app/preprocess.py
COPY README.md /app/README.md

WORKDIR /app

ENTRYPOINT ["python", "/app/preprocess.py"]
