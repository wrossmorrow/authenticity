#!/bin/bash
mkdir -p bootstrap/12318 ; python idLogit.py ../wikisurvey_12318_votes_2018-04-03T12_19_39Z.csv 12318 2> bootstrap/12318/error.log > bootstrap/12318/info.log &
mkdir -p bootstrap/12319 ; python idLogit.py ../wikisurvey_12319_votes_2018-04-03T12_20_13Z.csv 12319 2> bootstrap/12319/error.log > bootstrap/12319/info.log &
mkdir -p bootstrap/12383 ; python idLogit.py ../wikisurvey_12383_votes_2018-04-03T12_19_54Z.csv 12383 2> bootstrap/12383/error.log > bootstrap/12383/info.log &
mkdir -p bootstrap/12384 ; python idLogit.py ../wikisurvey_12384_votes_2018-04-02T22_29_48Z.csv 12384 2> bootstrap/12384/error.log > bootstrap/12384/info.log &