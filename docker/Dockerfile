FROM python
ARG  LLMEVHA_SHA
WORKDIR /workspace
ENV HOME=/workspace
ENV LLMEVHA_SHA=${LLMEVHA_SHA}
RUN chown -R 42420:42420 /workspace
USER 42420
RUN mkdir -p /workspace/lm-evaluation-harness
COPY --chown=42420 . /workspace/lm-evaluation-harness
RUN pip install -e /workspace/lm-evaluation-harness