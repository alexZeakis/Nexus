#!/bin/bash
# Requires NEXUS_LLM_BASE_URL and NEXUS_LLM_API_KEY in the environment.
# The API key is never written into these scripts.

datasets=("Agriculture" "CS" "Hypertension" "Legal" "Mix")

for data in "${datasets[@]}"; do
  echo "Processing dataset: $data"
  out_dir="../../output/graphs/Nexus/${data}/"

  export NEXUS_LLM_BASE_URL="http://localhost:8121/Nexus_${data}/construct/v1"

  # 1. Chunking
  python run_chunking.py \
      --data-dir "../../datasets/${data}" \
      --out-dir "${out_dir}" \
      --overlap-tokens 100 \
      --max-tokens 1200 \
      --tiktoken-model "gpt-4o-mini"

  # 2. Coreference and Fact Extraction
  python run_fact_extraction.py \
      --out-dir "${out_dir}" \
      --model "gpt-4o-mini" \
      --coref-device cuda:0

  # 3. Label Extraction
  python run_label_extraction.py \
      --out-dir "${out_dir}" \
      --model "gpt-4o-mini" \
      --n-medoids 300 \
      --max-calls 100

  # 4. Entity Extraction
  python run_entity_extraction.py \
      --out-dir "${out_dir}" \
      --device cuda:0 \
      --threshold 0.05 \
      --report-every 50

  # 5. Entity Linking
  python run_entity_linking.py \
      --out-dir "${out_dir}" \
      --device cuda:0 \
      --match-model "google/flan-t5-large"

  # 6. Hyperedge Linking
  python run_hyperedge_linking.py \
      --out-dir "${out_dir}" \
      --device cuda:0 \
      --join-threshold 0.5

  # 7. Graph Assembly and Indexing
  python run_graph_assembly.py \
      --out-dir "${out_dir}" \
      --device cuda:0 \
      --indexes

done
