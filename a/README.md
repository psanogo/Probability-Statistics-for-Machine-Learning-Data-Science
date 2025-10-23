--- a/README.md
+++ b/README.md
@@ -15,11 +15,11 @@
       url={https://arxiv.org/abs/2407.02273}, 
 }
 How to run
-We various steps in the pipeline, to reproduce the results we need to run all N steps, using multi_tp.main.
+There are various steps in the pipeline. To reproduce the results, you need to run all steps using `multi_tp.main`.
 
 dataset_preparation: prepare the vignette and translate them in the target language
 query_model: query the LLM in the target language
 backtranslate: translate the LLM respose from the target language to English
 parse_choice: parse the response (left/right)
-The analysis of the results can be made via analysis/anaylsis_rq.ipynb. Unzip the data folder to get our experimental results.
+The analysis of the results can be made via `analysis/analysis_rq.ipynb`. Unzip the `data.zip` file (if provided, or link to download) to get our experimental results.
 
 Details
 For inference we use the pathfinder library. The pathfinder library is a prompting library, that wraps around the most common LLM inference backends (OpenAI, Azure OpenAI, Anthropic, Mistral, OpenRouter, transformers library and vllm) and allows for easy inference with LLMs, it is available here. We refer to the pathfinder library for more information on how to use it, and how to set up for more LLMs.
@@ -31,11 +31,17 @@
 conda env create -f TrolleyCleanAPI.yml
 conda env create -f TrolleyCleanVLLM.yml
 How to run the experiments
-We chained the experiments via SLURM, sometimes they may fail. assuming the setup the various env as above.
+We chained the experiments via SLURM. Assuming you have set up the environments as described above, you can use the following functions. Note that jobs may sometimes fail, so it's good to monitor the SLURM queue.
 
 Language experiments
     submit_jobs() {
+    set -e # Exit immediately if a command exits with a non-zero status.
     local MODEL=$1
     local SLURM_SCRIPT=$2
 
+    if [ -z "${ALL_LANGUAGES-}" ] || [ ${#ALL_LANGUAGES[@]} -eq 0 ]; then
+        echo "Error: ALL_LANGUAGES array is not defined or is empty." >&2
+        return 1
+    fi
+
     local jobid=$(sbatch $SLURM_SCRIPT "TrolleyCleanVLLM" -m multi_tp.main_opt_lang steps='[query_model]' model_version=$MODEL analysis_backend_model_version=$LLAMA_3_1_8B | awk '{print $4}')
     backtranslate_jobids=()
     # Submit backtranslate jobs for all languages in parallel
@@ -49,8 +55,13 @@
 }
 
 submit_jobs_API() {
+    set -e # Exit immediately if a command exits with a non-zero status.
     local MODEL=$1
     local SLURM_SCRIPT=$2
+
+    if [ -z "${ALL_LANGUAGES-}" ] || [ ${#ALL_LANGUAGES[@]} -eq 0 ]; then
+        echo "Error: ALL_LANGUAGES array is not defined or is empty." >&2
+        return 1
+    fi
 
     backtranslate_jobids=()
     # Submit backtranslate jobs for all languages in parallel
@@ -60,7 +71,5 @@
     done
     dependency_list=$(IFS=:; echo "afterok:${backtranslate_jobids[*]}")
     sbatch --dependency=$dependency_list ./scripts/run_2x24G_fast.sh "TrolleyCleanVLLM"  -m multi_tp.main_opt_lang steps='[parse_choice]' model_version=$MODEL analysis_backend_model_version=$LLAMA_3_1_8B
-
-    sbatch ./scripts/run_2x24G_fast.sh "TrolleyCleanVLLM"  -m multi_tp.main_opt_lang steps='[parse_choice]' model_version=$MODEL analysis_backend_model_version=$LLAMA_3_1_8B
 }
