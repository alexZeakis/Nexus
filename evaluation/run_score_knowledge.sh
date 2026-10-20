#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)
methods=(Golden HyperRAG Nexus-fused Nexus-structured HyperGraphRAG Nexus-semantic PRoH)

for data in "${datasets[@]}"; do
  for n in "${sizes[@]}"; do
    for method in "${methods[@]}"; do
      echo "--- ${data} / ${n} / ${method} ---"
      python score_knowledge.py \
        --questions "../datasets/${data}/questions_${n}.json" \
        --pred      "../output/knowledge_${n}/${method}/${data}/" \
        --out-dir   ../output/knowledge_scores \
        --rs2
    done
  done
done

python gather_knowledge_scores.py --root ../output/knowledge_scores/
