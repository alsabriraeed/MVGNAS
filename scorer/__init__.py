from constants import *
from scorer.score_ade import evaluate_ade
from scorer.score_biorelex import evaluate_biorelex     

def evaluate(model, dataset, type):
    if type == ADE or type == ADE1:
        return evaluate_ade(model, dataset)
    if type == BIORELEX or type == BIORELEX1:
        return evaluate_biorelex(model, dataset)
