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
# Set at run time: NEBIUS_API_KEY, NEBIUS_BASE_URL, KIN_PLANNER_MODEL, KIN_NIGHTLY_MODEL, and KIN_NIGHTLY_TOKEN
# (the bearer token the hub must send); for --job also KIN_NIGHTLY_INPUT.
EXPOSE 8080
# As a service (a Nebius serverless container); use "--job" instead for a one-shot job.
ENTRYPOINT ["python", "-m", "kin.nightly", "--serve"]
