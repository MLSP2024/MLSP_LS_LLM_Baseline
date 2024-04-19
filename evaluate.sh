#!/bin/bash

if ! ( \
	[ -d MLSP_Organisers ] || \
	cp -R ../MLSP_Organisers MLSP_Organisers \
	)
then
	echo "Error: MLSP_Organisers data missing. Please copy the repo to the MLSP_Organisers directory." >&2
	exit 1
fi

# Gold label files are named by full language name (e.g. Filipino), released test data and therefore inputs by abbreviation (e.g. fil):

eval_prompt() {
	prompt="$1"
	if [ -z "$prompt" ]
	then
		dir="output"
		out="quick_results_only.tsv"
		prompt='only'
	else
		dir="output_${prompt}"
		out="quick_results_${prompt}.tsv"
	fi
	echo
	echo "PROMPT: ${prompt}"
	# TODO Make the pt file match our gold data, where one instance has been removed:
	sed -e '/tentativa da tca para/d' "${dir}/multilex_test_pt_ls.tsv" > "${dir}/multilex_test_pt_ls_568.tsv"
	cp "${dir}/multilex_test_pt_ls_diagnostics.json" "${dir}/multilex_test_pt_ls_568_diagnostics.json"
	(
		python quick_eval.py --header -D --gold 	MLSP_Organisers/Gold/Catalan/multilex_test_catalan_ls_labels.tsv	"${dir}/multilex_test_ca_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/German/multilex_test_german_ls_labels.tsv	"${dir}/multilex_test_de_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/English/multilex_test_english_ls_labels.tsv	"${dir}/multilex_test_en_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Spanish/multilex_test_spanish_ls_labels.tsv	"${dir}/multilex_test_es_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Filipino/multilex_test_filipino_ls_labels.tsv	"${dir}/multilex_test_fil_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/French/multilex_test_french_ls_labels.tsv	"${dir}/multilex_test_fr_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Italian/multilex_test_italian_ls_labels.tsv	"${dir}/multilex_test_it_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Japanese/multilex_test_japanese_ls_labels.tsv	"${dir}/multilex_test_ja_ls.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Portuguese/multilex_test_portuguese_ls_labels.tsv	"${dir}/multilex_test_pt_ls_568.tsv"
		python quick_eval.py -D --gold 	MLSP_Organisers/Gold/Sinhala/multilex_test_sinhala_ls_labels.tsv	"${dir}/multilex_test_si_ls.tsv"
	) | tee $out
}

eval_prompt
# eval_prompt unihd
