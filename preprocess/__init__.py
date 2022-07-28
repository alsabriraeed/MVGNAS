from constants import *
from preprocess.pre_ade import load_ade_dataset
from preprocess.pre_base import DataInstance
from preprocess.pre_helpers import tokenize     
from preprocess.pre_biorelex import load_biorelex_dataset

def load_data(dataset, split_nb, tokenizer):
    assert (dataset in DATASETS)       
    base_path = str(Path(__file__).parent.absolute()) + '/{}'.format(dataset)         
    if dataset == ADE or dataset == ADE1 :
        return load_ade_dataset(base_path, tokenizer, split_nb)
    if dataset == BIORELEX or dataset == BIORELEX1:
        return load_biorelex_dataset(base_path, tokenizer)
