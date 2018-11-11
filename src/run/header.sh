TRAIN_DATA=../ACL_data/annotated_train.json
TEST_DATA=../ACL_data/annotated_test.json
TRAIN_OPTIONS="-cvs 10 -cvi 10 -n 15 -dt $TRAIN_DATA -dT $TEST_DATA"

function train() {
  STATE=$1
  features=$2
  model=$3

  python main.py -mp $STATE train $TRAIN_OPTIONS -f $features -x $model --save
  python evaluate.py -G $TRAIN_DATA -g $STATE/predictions.json -o $STATE/predictions.score
}

function train_ilp() {
  STATE=$1
  python apply_ilp.py -mp $STATE -i $STATE/predictions.json -o $STATE/predictions_ilp.json
  python evaluate.py -G $TRAIN_DATA -g $STATE/predictions_ilp.json -o $STATE/predictions_ilp.score
}

function test_() {
  STATE=$1
  python main.py -mp $STATE run -d $TEST_DATA -o $STATE/test_predictions.json
  python evaluate.py -G $TEST_DATA -g $STATE/test_predictions.json -o $STATE/test_predictions.score
}

function test_ilp() {
  STATE=$1
  python apply_ilp.py -mp $STATE -i $STATE/test_predictions.json -o $STATE/test_predictions_ilp.json
  python evaluate.py -G $TEST_DATA -g $STATE/test_predictions_ilp.json -o $STATE/test_predictions_ilp.score
}
