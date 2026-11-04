# Shared steps for the model jobs. Sourced by each *.sbatch after MODEL_KEY, HF_ID, PORT, TAG, SERVE_ARGS are set.
# (no set -u: conda activate scripts reference unset variables)
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/abstraction-grader
LOG=/tmp/$USER/vllm_${MODEL_KEY}.log
echo "== $(date) setup on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-?}"
bash ~/node09_setup.sh                                   # databases (Postgres/Mongo) and scratch dirs
while pgrep -u $USER -f "hf download" > /dev/null; do echo "waiting for another download..."; sleep 60; done

conda activate ~/envs/vllm
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/tmp/$USER/hf VLLM_SYSTEM_START_DATE=2026-11-03   # date pinned: harmony prompts include it
echo "== $(date) weights: $HF_ID"
hf download "$HF_ID" ${HF_EXCLUDE:-} > /tmp/$USER/download_${MODEL_KEY}.log 2>&1 || { echo "download failed"; tail -20 /tmp/$USER/download_${MODEL_KEY}.log; exit 1; }
find /tmp/$USER -exec touch -h {} + 2>/dev/null
SNAP=$(ls -d $HF_HOME/hub/models--${HF_ID//\//--}/snapshots/*/ | head -1)
echo "generation_config.json:"; cat "$SNAP/generation_config.json" 2>/dev/null || echo "(none)"
TPL=$(ls "$SNAP"/chat_template.jinja 2>/dev/null || echo "$SNAP/tokenizer_config.json")
echo "chat template ($(basename $TPL)) mentions:"; grep -o "reasoning_strength\|reasoning_effort\|reasoning_content\|enable_thinking" "$TPL" 2>/dev/null | sort | uniq -c

echo "== $(date) starting vLLM on port $PORT"
vllm serve "$HF_ID" $SERVE_ARGS --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --enable-prompt-tokens-details --host 127.0.0.1 --port $PORT > $LOG 2>&1 &
VLLM_PID=$!
for i in $(seq 1 240); do curl -sf 127.0.0.1:$PORT/v1/models > /dev/null && break; kill -0 $VLLM_PID 2>/dev/null || break; sleep 10; done
curl -sf 127.0.0.1:$PORT/v1/models > /dev/null || { echo "vLLM did not start"; tail -60 $LOG; kill $VLLM_PID 2>/dev/null; exit 1; }
echo "vLLM up: $(curl -s 127.0.0.1:$PORT/version)"

conda activate ~/envs/grader
export VLLM_BASE_URL=http://127.0.0.1:$PORT/v1
if [ -n "${RUN_CMD:-}" ]; then                            # one-off command against the server instead of a batch
  echo "== $(date) $RUN_CMD"; eval "$RUN_CMD"; echo "== $(date) done"; kill $VLLM_PID; exit 0
fi
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  echo "== $(date) smoke test: yelp q1, every setting, seed 0"
  python -m grader.batch --model $MODEL_KEY --tag smoke-$TAG --datasets yelp --queries yelp:1 --seeds 0
  if ! python -m grader.smoke_check --tag smoke-$TAG --model $MODEL_KEY; then
    echo "== smoke test failed; stopping before the real runs. Last vLLM log lines:"; tail -40 $LOG
    kill $VLLM_PID; exit 1
  fi
fi

echo "== $(date) main batch: ${QUERIES:-all 13 queries} x all settings x seeds ${SEEDS}, prompts ${PROMPTS:-base}"
python -m grader.batch --model $MODEL_KEY --tag $TAG --seeds $SEEDS ${QUERIES:+--queries $QUERIES} ${PROMPTS:+--prompts $PROMPTS}
python -m grader.batch --model $MODEL_KEY --tag $TAG --summary --json > runs/_batch/${TAG}_summary.txt 2>&1
cat runs/_batch/${TAG}_summary.txt
echo "== $(date) done"
kill $VLLM_PID
