# Makefile to generate all the results.

SHELL := /bin/bash

TRAIN_DATA = ../data/annotated_train.json
TEST_DATA = ../data/annotated_test.json
MODELS = ../models

all: .results

.results: results/conv2_all_crf_max.ilp.predictions.jsonl results/conv2_all_crf_max.ilp.score results/conv2_all_crf_max.predictions.jsonl results/conv2_all_crf_max.score results/conv2_none_crf_simple.ilp.predictions.jsonl results/conv2_none_crf_simple.ilp.score results/conv2_none_crf_simple.predictions.jsonl results/conv2_none_crf_simple.score results/logistic_crf_max.ilp.predictions.jsonl results/logistic_crf_max.ilp.score results/logistic_crf_max.predictions.jsonl results/logistic_crf_max.score


gurobi810:
	wget https://packages.gurobi.com/8.1/gurobi8.1.0_linux64.tar.gz
	tar -xzf gurobi8.1.0_linux64.tar.gz

gurobi.lic:
	@echo "Please download your Gurobi license from http://www.gurobi.com/downloads/licenses/license-center"

glove/glove.6B.50.txt:
	mkdir -p glove
	cd glove
	wget -c https://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip glove.6B.50.txt

.env:
	echo "Creating virtual environment in .env";
	virtualenv -p python3 .env

install-requirements: gurobi810 requirements..env
	echo "Creating virtual environment in .env";
	virtualenv -p python3.6 .env
	source .env/bin/activate && cd gurobi810/linux64/ && python setup.py install
	source .env/bin/activate && pip install -r ./requirements.txt

results/%.predictions.jsonl: $(MODELS)/% $(TEST_DATA)
	mkdir -p results
	source .env/bin/activate
	python main.py -mp $(word 1,$^) run -d $(word 2,$^) -o $@

results/%.ilp.predictions.jsonl: results/%.predictions.jsonl
	source .env/bin/activate
	LD_LIBRARY_PATH=gurobi810/linux64/lib GRB_LICENSE_FILE=gurobi.lic python apply_ilp.py -i $< -o $@

results/%.score: results/%.predictions.jsonl $(TEST_DATA)
	source .env/bin/activate
	python evaluate.py -g $(word 1,$^) -G $(word 2,$^) -o $@


.PRECIOUS: results/%.predictions.jsonl

