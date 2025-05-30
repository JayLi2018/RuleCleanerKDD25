FROM continuumio/miniconda3

WORKDIR /app

# Install git (for cloning WRENCH)
RUN apt-get update && apt-get install -y git

# Clone WRENCH repo
RUN git clone https://github.com/JieyuZ2/wrench.git

# Create conda env but skip pip dependencies
RUN conda config --set pip_interop_enabled False && \
    conda env create --file wrench/environment.yml --name wrench || echo "conda env create failed but continuing"
RUN conda run -n wrench conda clean --all --yes || echo "conda clean failed but continuing"

# Now install pip packages manually
RUN grep -A 1000 'pip:' wrench/environment.yml | sed '1d' | sed 's/^- //' > wrench/requirements-pip.txt

# Patch broken pip packages
RUN conda run -n wrench pip uninstall -y huggingface-hub setuptools importlib-metadata || true && \
    conda run -n wrench pip install --no-cache-dir \
        huggingface-hub==0.0.8 \
        setuptools==59.6.0 \
        importlib-metadata==4.8.3 && \
    conda run -n wrench pip install -r wrench/requirements-pip.txt

# Add your packages
RUN conda run -n wrench pip install psycopg2-binary && \
    conda run -n wrench conda install -y -c conda-forge jupyterlab=2.2.9

# Set default shell to use the wrench environment
SHELL ["conda", "run", "-n", "wrench", "/bin/bash", "-c"]

# Set working directory to Rulecleaner code
WORKDIR /app/Rulecleaner

# Expose Jupyter port
EXPOSE 8888

# Default command: run Jupyter Lab
CMD ["conda", "run", "-n", "wrench", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]