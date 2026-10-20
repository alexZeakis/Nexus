#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)
methods=(Golden StandardRAG PlainLLM HyperRAG Nexus-fused Nexus-structured HyperGraphRAG Nexus-semantic PRoH)

# A-E costs one API call per question, so it is computed only for the two
# methods the paper reports it for. --resume reuses labels already written,
# so a rerun after a failure does not pay for them twice.

for data in "${datasets[@]}"; do
  for n in "${sizes[@]}"; do
    for method in "${methods[@]}"; do
      case "$method" in
        Nexus-fused|PRoH) EXTRA="--resume --judge" ;;
        *)                EXTRA="--resume" ;;
      esac
      echo "--- ${data} / ${n} / ${method} ---"
      python score_answers.py \
        --questions "../datasets/${data}/questions_${n}.json" \
        --pred      "../output/generate_${n}/${method}/${data}/" \
        --out-dir   ../output/answer_scores \
        ${EXTRA}
    done
  done
done

python gather_answers_scores.py --root ../output/answer_scores
