from typing import Sequence, List
import random
from copy import deepcopy
import numpy as np 
import re 
import pdb 

from semantic_parsing_with_constrained_lm.model import DataRetriever
from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub, FullDatum
from semantic_parsing_with_constrained_lm.index.exact_match_index import ExactMatchIndex

from ambiguous_parsing.generation.fixtures.nps import PLURAL_NP_TO_SINGULAR
from ambiguous_parsing.generation.fixtures.vps import VPS_MAP, INTRANSITIVE_VPS_FOR_BOUND

class LampGeneralizationRetriever(DataRetriever[FullDatumSub, DatumSub]): 
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        # self.index: ExactMatchIndex[DatumSub, FullDatumSub] = ExactMatchIndex.create(
        #     train_data,
        #     get_content=lambda c: c.natural,  # type: ignore
        #     get_query=lambda q: q.natural,  # type: ignore
        # )
        self.baseline_type = baseline_type

        self.data: List[FullDatumSub] = list(train_data)

        self.index = {d.natural: (i, d) for i, d in enumerate(self.data)}

        self.top_k = top_k
        self.best_first = best_first
        self.shuffle = True
        self.prng = random.Random(
            seed
        )  # a random number generator to ensure deterministic behavior
        np.random.seed(seed) 

class LampGeneralizationPPRetriever(LampGeneralizationRetriever):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        super().__init__(train_data, top_k, best_first, shuffle, seed, baseline_type)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # for a given test example of a PP ambiguity, generate 3 training examples
        # 1. an example using the same verb with the same agent/patient but no PP
        # 2. an example of the same verb with a pp but no patient
        # 3. an example of the same verb with "with" but as possesive 

        # extract np1, np2, np3, vp1 from the example 
        bindings = test_datum.var_bindings
        if bindings['vp1'] == "picked_up":
            bindings['vp1'] = "picked up"

        subj_is_def = test_datum.natural.startswith("the")
        obj_is_def = not bindings['np2'][0].isupper()
        # only allow indefinite objects         
        # create templates and generate 
        if subj_is_def:
            if obj_is_def:
                surface_form_transitive = f"the {bindings['np1']} {bindings['vp1']} the {bindings['np2']}"
            else:
                surface_form_transitive = f"the {bindings['np1']} {bindings['vp1']} {bindings['np2']}"
            surface_form_instrument = f"the {bindings['np1']} {bindings['vp1']} with the {bindings['np3']}"
        else:
            if obj_is_def:
                surface_form_transitive = f"{bindings['np1']} {bindings['vp1']} the {bindings['np2']}"
            else:
                surface_form_transitive = f"{bindings['np1']} {bindings['vp1']} {bindings['np2']}"
            surface_form_instrument = f"{bindings['np1']} {bindings['vp1']} with the {bindings['np3']}"
        if obj_is_def:
            surface_form_possessive = f"the {bindings['np2']} with the {bindings['np3']}"
        else:
            surface_form_possessive = f"{bindings['np2']} with the {bindings['np3']}"

        # search to retrieve train examples 
        try:
            transitive_result = self.index[surface_form_transitive][1]
            instrument_result = self.index[surface_form_instrument][1]
            possessive_result = self.index[surface_form_possessive][1]
        except KeyError:
            raise AssertionError("Could not find all of the examples in the training data: " + surface_form_transitive + " " + surface_form_instrument + " " + surface_form_possessive) 

        if self.baseline_type == "baseline_instrument": 
            # choose ONLY one of possessive or instrument 
            results = [transitive_result, instrument_result]
        elif self.baseline_type == "baseline_possessive": 
            # choose ONLY one of possessive or instrument 
            results = [transitive_result, possessive_result]
        else: 
            assert(self.baseline_type in [None, "full"])
            # baseline is None, normal prompt with ambiguous signal 
            results = [transitive_result, instrument_result, possessive_result]

        np.random.shuffle(results)
        return results

class LampGeneralizationConjRetriever(LampGeneralizationRetriever):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        super().__init__(train_data, top_k, best_first, shuffle, seed, baseline_type)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # for a given test example of a conj ambiguity, generate 2 training examples
        # TODO: (elias): need a way to give instructions about parentheses without biasing
        # how about: 
        # 1. A and B and C: 
        #   A and (B and C)
        #   (A and B) and C
        # 2. A or B or C: 
        #    A or (B or C)
        #    (A or B) or C

        surface_form_or = re.sub(" and ", " or ", test_datum.natural)
        surface_form_and = re.sub(" or ", " and ", test_datum.natural)

        # search to retrieve train examples 
        or_result = self.index[surface_form_or][1]
        and_result = self.index[surface_form_and][1]
        results = [or_result, and_result]

        np.random.shuffle(results)
        return results

class LampGeneralizationScopeRetriever(LampGeneralizationRetriever):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        super().__init__(train_data, top_k, best_first, shuffle, seed, baseline_type)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # Test examples like: "every man hears a bird"
        # ingredients: 
        # 1. every man 
        # 2. a man hears a bird
        bindings = test_datum.var_bindings

        split_nat = test_datum.natural.split(" ")
        quant_type = split_nat[0]
        remainder = " ".join(split_nat[1:])
        quant_var = bindings['np1']

        quant_surface_form = f"{quant_type} {quant_var}"
        if re.match("[aeiou]", remainder[0]):
            article = "an"
        else:
            article = "a"
        nonquant_surface_form = f"{article} {remainder}"

        # search to retrieve train examples 

        quant_result = self.index[quant_surface_form][1]
        nonquant_result = self.index[nonquant_surface_form][1]
        results = [quant_result, nonquant_result]

        np.random.shuffle(results)
        return results

class LampGeneralizationRevscopeRetriever(LampGeneralizationRetriever):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        super().__init__(train_data, top_k, best_first, shuffle, seed, baseline_type)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # Test examples like: "a man hears every bird"
        # ingredients: 
        # 1. every bird
        # 2. a man hears a bird
        bindings = test_datum.var_bindings

        split_nat = test_datum.natural.split(" ")
        # TODO (elias): is this robust? are all final words 1 token? 
        quant_type = "every" if "every" in split_nat else "each"
        quant_idx = split_nat.index(quant_type)
        remainder = " ".join(split_nat[0:quant_idx])

        quant_var = bindings['np2']
        if quant_var in PLURAL_NP_TO_SINGULAR.keys():
            quant_var = PLURAL_NP_TO_SINGULAR[quant_var]

        quant_surface_form = f"{quant_type} {quant_var}"
        if re.match("[aeiou]", quant_var[0]):
            article = "an"
        else:
            article = "a"
        nonquant_surface_form = f"{remainder} {article} {quant_var}" 

        # search to retrieve train examples
        quant_result = self.index[quant_surface_form][1]
        nonquant_result = self.index[nonquant_surface_form][1]

        results = [quant_result, nonquant_result]


        np.random.shuffle(results)
        return results

class LampGeneralizationBoundRetriever(LampGeneralizationRetriever):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
        baseline_type: str = None,
    ):
        super().__init__(train_data, top_k, best_first, shuffle, seed, baseline_type)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # test format: Bill saw John and he waved
        # Ingredients:
        # 1. Bill saw John
        # 2. Bill waved
        # 3. John waved  
        # extract np1, np2, np3, vp1 from the example 
        bindings = test_datum.var_bindings

        # surface_form_transitive = f"{bindings['np1']} {bindings['vp1']} {bindings['np2']}"
        surface_form_transitive = test_datum.natural.split(" and ")[0]
        # verbs = [f"({v})" for v in VPS_MAP.keys()]
        # verb_gex = "|".join(verbs)
        subj, obj = re.split(bindings['vp1'], surface_form_transitive) 
        # surface_form_np1 = f"{bindings['np1']} {bindings['vp2']}"
        # surface_form_np2 = f"{bindings['np2']} {bindings['vp2']}"
        surface_form_np1 = f"{subj.strip()} {bindings['vp2']}"
        surface_form_np2 = f"{obj.strip()} {bindings['vp2']}"

        # Mary jumped and Mary frowned
        # need a surface form with 2 events so that model knows how to use 2 events 
        eligible_vps = list(set(INTRANSITIVE_VPS_FOR_BOUND) - (set([bindings['vp2']])))
        rand_verb = eligible_vps[np.random.randint(len(eligible_vps))]
        surface_form_conj = f"{subj.strip()} {rand_verb} and {bindings['vp2']}"

        # search to retrieve train examples
        try:
            transitive_result = self.index[surface_form_transitive][1]
            np1_result = self.index[surface_form_np1][1]
            np2_result = self.index[surface_form_np2][1]
            conj_result = self.index[surface_form_conj][1]
        except KeyError:
            pdb.set_trace()

        results = [transitive_result, np1_result, np2_result, conj_result]


        np.random.shuffle(results) 
        return results