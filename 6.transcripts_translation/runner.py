from llama32 import Llama32Model

import argparse
from enum import StrEnum

class Model(StrEnum):
    Llama32 = 'llama32'
    Llama4S = 'llama4scout'
    Llama4M = 'llama4maverick'


parser = argparse.ArgumentParser(description='Runner script to evaluate LecTrans annotated testset')
parser.add_argument('-m', '--model', type=Model, choices=list(Model), help='Model to load')


if __name__ == "__main__":
    args = parser.parse_args()
    match args.model:
        case Model.Llama32:
            model = Llama32Model(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
        case Model.Llama4S:
            model = Llama32Model(model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct")
        case Model.Llama4M:
            model = Llama32Model(model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
        case _:
            raise NotImplementedError
    
    model.evaluate(output_suffix=f"{args.model}_pred")

