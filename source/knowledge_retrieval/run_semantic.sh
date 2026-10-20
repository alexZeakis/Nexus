#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)
methods=(Nexus-semantic)

######## BASELINES ######
for data in "${datasets[@]}"; do
  echo "Processing dataset: $data"
  out_dir="../../output/"
  for n in "${sizes[@]}"; do
    for method in "${methods[@]}"; do

      python hyperedge_simsearch.py \
          --graph "${out_dir}/graphs/Nexus/${data}/graph.graphml" \
          --questions ../../datasets/${data}/questions_${n}.json \
          --chunks "${out_dir}/graphs/Nexus/${data}/chunks.json" \
          --out ${out_dir}/knowledge_${n}/${method}/${data}/knowledge.json

      export NEXUS_LLM_BASE_URL="http://localhost:8121/${method}_${data}/generate_${n}/v1"
      python ../text_generation/generate_answers.py \
          --retrieval ${out_dir}/knowledge_${n}/${method}/${data}/knowledge.json \
          --out ${out_dir}/generate_${n}/${method}/${data}/answers.json \
          --questions ../../datasets/${data}/questions_${n}.json \
          --knowledge all \
          --answer-length short \
          --chunk-top-k 0.5 \
          --workers 16
    done
  done
done
