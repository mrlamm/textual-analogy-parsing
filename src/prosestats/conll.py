"""
Read conll files.
"""

from io import StringIO
from tqdm import tqdm

def read_conll(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @returns a list of examples [(col1), (col2), ...].
    """
    ret = []

    line = next(fstream).strip()
    assert len(line) > 0 and not line.startswith('-DOCSTART-') and "\t" in line, "Couldn't identify the number of columns from first line"
    current_cols = [[t] for t in line.split("\t")]

    for line in tqdm(fstream, desc="reading conll"):
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_cols[0]) > 0:
                ret.append(current_cols)
            current_cols = [[] for _ in current_cols]
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
            cols = line.split("\t")
            for i, col in enumerate(cols):
                current_cols[i].append(col)
    if len(current_cols[0]) > 0:
        ret.append(current_cols)
    return ret

def test_read_conll():
    input_ = [
        "EU	ORG",
        "rejects	O",
        "German	MISC",
        "call	O",
        "to	O",
        "boycott	O",
        "British	MISC",
        "lamb	O",
        ".	O",
        "",
        "Peter	PER",
        "Blackburn	PER",
        "",
        ]
    output = [
        ["EU rejects German call to boycott British lamb .".split(), "ORG O MISC O O O MISC O O".split()],
        ["Peter Blackburn".split(), "PER PER".split()]
        ]

    assert read_conll(iter(input_)) == output

def test_read_conll_multi():
    input_ = [
        "C.R.	n	0	0",
        "Bard	n	0	0",
        "Inc.	n	0	0",
        "yesterday	n	0	0",
        "said	n	0	0",
        "third-quarter	iv	0	1",
        "net	dl	1,2	1,2",
        "plunged	n	0	0",
        "51	n	0	0",
        "%	n	0	0",
        "to	n	0	0",
        "$	du	1	1",
        "9.9	dv	1	1",
        "million	dv	1	1",
        ".	n	0	0",
        "",
        ]
    output = [
        ["C.R. Bard Inc. yesterday said third-quarter net plunged 51 % to $ 9.9 million .".split(), "n n n n n iv dl n n n n du dv dv n".split(), "0 0 0 0 0 0 1,2 0 0 0 0 1 1 1 0".split(), "0 0 0 0 0 1 1,2 0 0 0 0 1 1 1 0".split()],
        ]
    output_ = read_conll(iter(input_))
    assert output_ == output

def write_conll(fstream, data):
    """
    Writes to an output stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @data a list of examples [(tokens), (labels), (predictions)]. @tokens, @labels, @predictions are lists of string.
    """
    for cols in data:
        for row in zip(*cols):
            row = [",".join(str(e) for e in elem) if isinstance(elem, list) else elem for elem in row]
            fstream.write("\t".join(row)) # Handles list elements.
            fstream.write("\n")
        fstream.write("\n")

def test_write_conll():
    input_ = [
        ("EU rejects German call to boycott British lamb .".split(), "ORG O MISC O O O MISC O O".split()),
        ("Peter Blackburn".split(), "PER PER".split())
        ]
    output = """EU	ORG
rejects	O
German	MISC
call	O
to	O
boycott	O
British	MISC
lamb	O
.	O

Peter	PER
Blackburn	PER

"""
    output_ = StringIO()
    write_conll(output_, input_)
    output_ = output_.getvalue()
    assert output == output_

def merge_conll_features(main_conll, feature_conll):
    tokens, labels, groups, indices = main_conll
    ixs, tokens_, lemmas, pos_tags, ner_tags, _, _ = feature_conll
    assert tokens == tokens_
    return [ixs, tokens, lemmas, pos_tags, ner_tags, labels, groups, indices]
