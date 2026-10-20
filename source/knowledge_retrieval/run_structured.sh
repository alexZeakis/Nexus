#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)
method="Nexus-structured"

for data in "${datasets[@]}"; do
  echo "Processing dataset: $data"
  out_dir="../../output"
  graph_dir="${out_dir}/graphs/Nexus/${data}"

  for n in "${sizes[@]}"; do
    questions="../../datasets/${data}/questions_${n}.json"
    know_dir="${out_dir}/knowledge_${n}/Nexus/${data}"

    echo "--- ${data} / ${n} questions ---"

    # 1. Query Entity Extraction. No LLM: the question is used as it is.
    python run_query_entity_extraction.py \
        --questions "$questions" \
        --out-dir "$know_dir" \
        --schema "${graph_dir}/schema.json" \
        --device cuda:0 \
        --threshold 0.05

    # 2. Hyperedge Retrieval. No LLM, one join.
    python run_hyperedge_retrieval.py \
        --out-dir "$know_dir" \
        --hyperedges "${graph_dir}/hyperedge_linking.json" \
        --postings "${graph_dir}/postings.json" \
        --query-entities "${know_dir}/query_entities.json" \
        --k "10"

      
    # 3. Hyperedge Binding. No LLM, one join per round.
    python run_hyperedge_binding.py \
        --retrieved       "${know_dir}/retrieved.json" \
        --query-entities  "${know_dir}/query_entities.json" \
        --hyperedges      "${graph_dir}/hyperedge_linking.json" \
        --postings        "${graph_dir}/postings.json" \
        --chunks          "${graph_dir}/chunks.json" \
        --out             "${out_dir}/knowledge_${n}/${method}/${data}/knowledge.json" \
        --expand-tokens "union" \
        --seeds "10" --k "10" --rounds "1" \
        --run-name "$method"

    # 4. Text Generation. The only LLM call in retrieval.
    export NEXUS_LLM_BASE_URL="http://localhost:8121/${method}_${data}/generate_${n}/v1"
    python ../text_generation/generate_answers.py \
        --retrieval "${out_dir}/knowledge_${n}/${method}/${data}/knowledge.json" \
        --out "${out_dir}/generate_${n}/${method}/${data}/answers.json" \
        --questions "$questions" \
        --knowledge all \
        --answer-length short \
        --chunk-top-k 0.5 \
        --workers 8             
  done
done
