# MLSP_LS_LLM_Baseline

A baseline based on zero-shot prompting a large language model. We employ the chat-finetuned [Llama 2 70B model](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) in 4-bit quantisation. We use a the following zero-shot prompt template and temperature 0.3 to generate a maximum of 256 new tokens.

```
Context: {context}
Question: Given the above context, list ten alternative {lang_space}words for "{word}" that are easier to understand. *List only the words without translations, transcriptions or explanations.*
Answer:
```

To construct the prompt, the placeholders in curly braces are replaced by the context, the language of the instance, and the target word to be simplified. For English, the placeholder `{language}` and the subsequent space is omitted. The prompt is identical to a zero-shot prompt employed for lexical simplification using a ChatGPT model by  [Aumiller and Gertz (2022)](https://github.com/dennlinger/TSAR-2022-Shared-Task), except for the the emphasised sentence (`List only`…), which we have added to reduce unnecessary translations to English, transcriptions to Latin alphabet, or explanations. Such extra input was generated frequently when we applied the original prompt to trial data. The addition of the sentence results in both faster inference and higher accuracy.

Our postprocessing also builds on the work by Aumiller and Gertz (2022). Based on an examination of outputs using the trial data, we made minor changes reflecting a broader array of languages and scripts as well as a different model. For instance, we allow words to be separated by ideographic commas (、) commonly used in Japanese, or lists enumerated using letters (e.g. `a)`, `b)`, …), which occurred in Llama 2 output.


## Reproducing the baseline
 
Note that the [output](output) of the baseline is already included in the repository. You can reproduce it by following the steps below.

1. Install the Git submodule for [MLSP_Data](https://github.com/MLSP2024/MLSP_Data):

    ```git submodule init && git submodule update```
    
2. Install the [requirements](requirements.txt):
	
	```python -m pip install -r requirements.txt```
    
3. Run the baseline:

    ```bash experiments.sh```
