# A summary of all the execution commands to be run.
. run/header.sh

# Runs
STATE=state/conv2_all_crf_max
train $STATE "word casing ner depparse depparse_copy depparse_dists depparse_path" "decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=max"
train_ilp $STATE
test_ $STATE
test_ilp $STATE

STATE=state/conv2_all_crf_simple
train $STATE "word casing ner depparse depparse_copy depparse_dists depparse_path" "decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=simple"

STATE=state/conv2_all_simple_max
train $STATE "word casing ner depparse depparse_copy depparse_dists depparse_path" "decode_layer=simple embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=max"

STATE=state/conv2_ner_crf_simple
train $STATE "word casing ner" "decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=none"

STATE=state/conv2_depparse_crf_max
train $STATE "word casing depparse depparse_copy depparse_dists depparse_path" "decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=max"

STATE=state/conv2_none_crf_simple
train $STATE "word casing" "decode_layer=crf embed_layer=conv n_layers=2 node_model=simple edge_model=simple path_agg=none"
#train_ilp $STATE
test_ $STATE
#test_ilp $STATE

STATE=state/logistic_crf_max
train $STATE "word casing lemma pos ner depparse depparse_copy depparse_dists depparse_path" "decode_layer=crf embed_layer=none node_model=none edge_model=none path_agg=max update_L_=false"
#train_ilp $STATE
test_ $STATE
#test_ilp $STATE

STATE=state/logistic_crf_simple
train $STATE "word casing lemma pos ner depparse depparse_copy depparse_dists depparse_path" "decode_layer=crf embed_layer=none node_model=none edge_model=none path_agg=none update_L_=false"
