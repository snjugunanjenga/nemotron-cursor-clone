FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
ARG INSTALL_VLLM=false
RUN pip install --no-cache-dir -r requirements.txt
# Optionally install vllm (heavy) at build time: --build-arg INSTALL_VLLM=true
RUN if [ "${INSTALL_VLLM}" = "true" ] ; then pip install --no-cache-dir -r requirements-optional.txt || true ; fi
COPY . /app
EXPOSE 8000 7860
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
