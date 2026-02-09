from llama32 import Llama32Model
from llama4 import Llama4Model
from dummy import DummyInputCheckModel

import argparse
from enum import StrEnum

class Model(StrEnum):
    Llama32 = 'llama32'
    Llama4S = 'llama4scout'
    Llama4M = 'llama4maverick'
    DummyInputCheck = 'dummy'

class Context(StrEnum):
    Vanilla = ''
    SlideText = 'slidetxt'
    SlideImage = 'slideimg'


parser = argparse.ArgumentParser(description='Runner script to evaluate LecTrans annotated testset')
parser.add_argument('-m', '--model', type=Model, choices=list(Model), default=Model.DummyInputCheck,
                    help='Model to load (default: %(default)s)')
parser.add_argument('-c', '--context', type=Context, choices=list(Context), default=Context.Vanilla,
                    help='Context to provide (default: %(default)s)')
parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'), default=None,
                    help='Provide start and end indices')


if __name__ == "__main__":
    args = parser.parse_args()
    kwargs = {
        "output_suffix": args.model,
        "context": args.context,
        "rg_idx": args.range,
    }
    match args.model:
        case Model.Llama32:
            model = Llama32Model(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
        case Model.Llama4S:
            model = Llama4Model(model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct")
        case Model.Llama4M:
            model = Llama4Model(model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
        case Model.DummyInputCheck:
            model = DummyInputCheckModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
            kwargs["dummy"] = True
        case _:
            raise NotImplementedError
    
    model.evaluate(**kwargs)

