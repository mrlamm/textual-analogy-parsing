# Makefile to generate all the results.

SHELL := /bin/bash

TRAIN_DATA = ../data/train.json
TEST_DATA = ../data/test.json
MODELS = ../models

all: .results

.results: results/neural_all.predictions.jsonl results/neural_all.ilp.score results/neural_none.predictions.jsonl results/neural_none.ilp.score results/neural_none.predictions.jsonl results/neural_none.score results/logistic_all.predictions.jsonl results/logistic_all.ilp.score results/logistic_all.predictions.jsonl results/logistic_all.score

.models: $(MODELS)/neural_all $(MODELS)/neural_none $(MODELS)/logistic_all

.ablation-models: .models $(MODELS)/neural_wo_ner $(MODELS)/neural_wo_deps $(MODELS)/neural_wo_crf

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

install-requirements: gurobi810 requirements.txt .env
	echo "Creating virtual environment in .env";
	virtualenv -p python3.6 .env
	source .env/bin/activate && cd gurobi810/linux64/ && python setup.py install
	source .env/bin/activate && pip install -r ./requirements.txt

results/%.predictions.jsonl: $(MODELS)/% $(TEST_DATA)
	mkdir -p results
	source .env/bin/activate && python main.py -mp $(word 1,$^) run -d $(word 2,$^) -o $@

results/%.ilp.predictions.jsonl: results/%.predictions.jsonl
	source .env/bin/activate && LD_LIBRARY_PATH=gurobi810/linux64/lib GRB_LICENSE_FILE=gurobi.lic python apply_ilp.py -i $< -o $@

results/%.score: results/%.predictions.jsonl $(TEST_DATA)
	source .env/bin/activate && python evaluate.py -g $(word 1,$^) -G $(word 2,$^) -o $@


.PRECIOUS: results/%.predictions.jsonl

# Training section

TRAIN_OPTIONS=-cvs 10 -cvi 10 -n 15 -dt $(TRAIN_DATA) -dT $(TEST_DATA)

$(MODELS)/neural_all: $(TRAIN_DATA)
	source .env/bin/activate && \
  python main.py -mp $@ train $(TRAIN_OPTIONS) \
		-f word casing ner depparse depparse_copy depparse_dists depparse_path \
	 	-x decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=max \
	 	--save && \
	python evaluate.py -G $(TRAIN_DATA) -g $@/predictions.json -o $@/predictions.score

$(MODELS)/neural_none: $(TRAIN_DATA)
	source .env/bin/activate && \
	python main.py -mp $@ train $(TRAIN_OPTIONS) \
		-f word casing ner depparse depparse_copy depparse_dists depparse_path \
		-x decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=simple \
	 	--save && \
	python evaluate.py -G $(TRAIN_DATA) -g $@/predictions.json -o $@/predictions.score

$(MODELS)/logistic_all: $(TRAIN_DATA)
	source .env/bin/activate && python main.py -mp $@ train $(TRAIN_OPTIONS) \
		-f word casing lemma pos ner depparse depparse_copy depparse_dists depparse_path \
	 	-x decode_layer=crf embed_layer=none node_model=none edge_model=none path_agg=none update_L_=false \
	 	--save && \
	python evaluate.py -G $(TRAIN_DATA) -g $@/predictions.json -o $@/predictions.score

$(MODELS)/neural_wo_ner: $(TRAIN_DATA)
	source .env/bin/activate && python main.py -mp $@ train $(TRAIN_OPTIONS) \
	 	-f word casing depparse depparse_copy depparse_dists depparse_path \
	 	-x decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=max \
		--save && \
	python evaluate.py -G $(TRAIN_DATA) -g $@/predictions.json -o $@/predictions.score

$(MODELS)/neural_wo_deps: $(TRAIN_DATA)
	source .env/bin/activate && python main.py -mp $@ train $(TRAIN_OPTIONS) \
		-f word casing ner \
	 	-x decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=none \
	 	--save && \
	python evaluate.py -G $(TRAIN_DATA) -g $@/predictions.json -o $@/predictions.score
