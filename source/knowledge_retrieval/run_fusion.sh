#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)

for data in "${datasets[@]}"; do
  echo "Processing dataset: $data"
  out_dir="../../output"

  for n in "${sizes[@]}"; do
    echo "--- ${data} / ${n} questions ---"

    python fuse_knowledge.py \
          --semantic   "${out_dir}/knowledge_${n}/Nexus-semantic/${data}/knowledge.json" \
          --structural "${out_dir}/knowledge_${n}/Nexus-structured/${data}/knowledge.json" \
          --out        "${out_dir}/knowledge_${n}/Nexus-fused/${data}/knowledge.json" \
          --name       Nexus-fused

      # 4. Text Generation. The only LLM call in retrieval.
    export NEXUS_LLM_BASE_URL="http://localhost:8121/Nexus-fused_${data}/generate_${n}/v1"
    python ../text_generation/generate_answers.py \
          --retrieval "${out_dir}/knowledge_${n}/Nexus-fused/${data}/knowledge.json" \
          --out "${out_dir}/generate_${n}/Nexus-fused/${data}/answers.json" \
          --questions "../../datasets/${data}/questions_${n}.json" \
          --knowledge all \
          --answer-length short \
          --chunk-top-k 0.5 \
          --workers 8             
  done
done
