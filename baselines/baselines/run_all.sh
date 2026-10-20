#!/bin/bash

datasets=(Mix Agriculture Legal CS Hypertension)
sizes=(200 512)
methods=(Golden StandardRAG PlainLLM)

######## BASELINES ######
for data in "${datasets[@]}"; do
  echo "Processing dataset: $data"
  out_dir="../../output/"

  for n in "${sizes[@]}"; do
    questions="../../datasets/${data}/questions_${n}.json"

    for method in "${methods[@]}"; do
      know="${out_dir}/knowledge_${n}/${method}/${data}/knowledge.json"
      ans="${out_dir}/generate_${n}/${method}/${data}/answers.json"

      export NEXUS_LLM_BASE_URL="http://localhost:8121/${method}_${data}/generate_${n}/v1"
      case "$method" in
        Golden)
          python golden_context.py \
              --questions "$questions" \
              --chunks "${out_dir}/graphs/Nexus/${data}/chunks.json" \
              --out "$know"
              
         python ../../source/text_generation/generate_answers.py \
            --retrieval "$know" \
            --out "$ans" \
            --questions "$questions" \
            --knowledge all \
            --answer-length short \
            --chunk-top-k 0.5 \
            --workers 16              
          ;;
        StandardRAG)
          python chunk_simsearch.py \
              --chunks "${out_dir}/graphs/Nexus/${data}/chunks.json" \
              --questions "$questions" \
              --out "$know"
              
          python ../../source/text_generation/generate_answers.py \
              --retrieval "$know" \
              --out "$ans" \
              --questions "$questions" \
              --knowledge all \
              --workers 8              
          ;;
        PlainLLM)
          python no_knowledge.py \
              --questions "$questions" \
              --out "$know"
              
          python ../../source/text_generation/generate_answers_no_knowledge.py \
              --retrieval "$know" \
              --out "$ans" \
              --questions "$questions" \
              --workers 8
          ;;
        *)
          echo "unknown method ${method}" >&2
          continue
          ;;
      esac

      

    done
  done
done
