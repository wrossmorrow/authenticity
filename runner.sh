#!/bin/bash
SURVEY_DATA_DIR=".."
RESULTS_OUT_DIR="runs/results/bootstrap"
RESULTS_LOG_DIR="runs/logs"
for L in "12318" "12319" "12383" "12384" ; do 
	mkdir -p ${RESULTS_OUT_DIR}/${L}
	INFILE="${SURVEY_DATA_DIR}/wikisurvey_${L}_votes_2018-04-03T12_19_39Z.csv"
	python idLogit.py ${INFILE} ${L} \
		2> ${RESULTS_LOG_DIR}/${L}/error.log \
		 > ${RESULTS_LOG_DIR}/${L}/info.log &
done