# Kin's nightly reflection as a container job (Nebius Serverless AI Job, or anywhere else).
#
# It only thinks: payload in through KIN_NIGHTLY_INPUT, result out through the log. It has no database,
# no robot and no voice, so none of those dependencies are installed.
#
#   docker build -f deploy/nightly/Dockerfile -t kin-nightly .
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "openai>=1.40" "python-dotenv>=1.0" "pillow>=10.0" "numpy>=1.26" "httpx>=0.27"
COPY kin/ kin/

ENV PYTHONUNBUFFERED=1
# Set at run time: NEBIUS_API_KEY, KIN_NIGHTLY_INPUT, and optionally KIN_NIGHTLY_MODEL / KIN_PLANNER_MODEL.
ENTRYPOINT ["python", "-m", "kin.nightly", "--job"]
