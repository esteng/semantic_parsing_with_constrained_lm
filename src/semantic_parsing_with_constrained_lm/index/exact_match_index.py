import pdb 
import re
import random
from copy import deepcopy
import tempfile
from collections import defaultdict
from typing import Callable, Generic, Iterable, List, Sequence, Tuple
import numpy as np

import whoosh.index
from whoosh.fields import STORED, TEXT, SchemaClass
from whoosh.qparser import OrGroup, QueryParser

from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub, FullDatum
from semantic_parsing_with_constrained_lm.index import Candidate, DynamicIndex, Query
from semantic_parsing_with_constrained_lm.model import DataRetriever
from semantic_parsing_with_constrained_lm.index.bm25_index import PromptSchema

from ambiguous_parsing.generation.fixtures.nps import NPS_MAP, VISUAL_INSTRUMENT_NPS, TACTILE_INSTRUMENT_NPS
from ambiguous_parsing.generation.fixtures.vps import VISUAL_VPS, TACTILE_VPS, INTRANSITIVE_VPS, INTRANSITIVE_VPS_FOR_BOUND, VPS_MAP


class ExactMatchIndex(Generic[Query, Candidate], DynamicIndex[int, Query, Candidate]):
    """
    An index that does exact matching based on anonymized strings. This is used to construct ambiguous prompts
    for templates like "the man saw the boy with the telescope", where we want ~50% of the examples to be of 
    one parse, and 50% of the other. BM25Index is not as good at this because it ends up retrieving examples for 
    similar but unrelated templates, like "the man saw the boy with the mittens" 
    """
    def __init__(
        self, get_content: Callable[[Candidate], str], get_query: Callable[[Query], str]
    ):
        self.index = defaultdict(list)
        self.get_content = get_content
        self.get_query = get_query

    @classmethod
    def create(
        cls,
        candidates: Iterable[Candidate],
        get_content: Callable[[Candidate], str],
        get_query: Callable[[Query], str],
    ) -> "ExactMatchIndex":
        index = ExactMatchIndex(get_content, get_query)
        for i, candidate in enumerate(candidates):
            index.index[get_content(candidate)].append(i)
        return index

    def add(self, candidates: Iterable[Candidate]):
        index_len = len(self.index)
        for i, candidate in enumerate(candidates):
            self.index[self.get_content(candidate)].append(index_len + i)

    def search(self, query: Query, top_k: int = 5) -> List[Tuple[int, float]]:
        q = self.get_query(query)
        if q in self.index.keys():
            matches = self.index[q]
            
            # sample top k matches
            top_matches = random.sample(matches, min(top_k, len(matches)))
            return [(m, 1.0) for m in top_matches]
        else:
            # this should not happen unless the training data contains 0 examples for a given template
            return [(None, 0.0)]

class LampExactMatchRetriever(DataRetriever[FullDatumSub, DatumSub]): 
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        seed: int = 12345,
    ):

        self.index: ExactMatchIndex[DatumSub, FullDatumSub] = ExactMatchIndex.create(
            train_data,
            get_content=lambda c: self.anon(c.natural),  # type: ignore
            get_query=lambda q: self.anon(q.natural),  # type: ignore
        )
        self.data: List[FullDatumSub] = list(train_data)
        self.top_k = top_k
        self.best_first = best_first
        self.prng = random.Random(
            seed
        )  # a random number generator to ensure deterministic behavior

    def anon(self, natural: str) -> str:
        # replace NPs with <NP>
        mapping = [("<VIS_NP>", VISUAL_INSTRUMENT_NPS),
                  ("<TAC_NP>", TACTILE_INSTRUMENT_NPS),
                  ("<NP>", NPS_MAP.keys()),
                  ("<VIS_VP>", VISUAL_VPS),
                  ("<TAC_VP>", TACTILE_VPS),
                  ("<INTR_VP>", INTRANSITIVE_VPS),
                  ("<INTR_VP_BOUND>", INTRANSITIVE_VPS_FOR_BOUND),
                  ("<VP>", VPS_MAP.keys())]
        natural_split = re.split("\s+", natural)
        for i, word in enumerate(natural_split):
            for replacer, candidates in mapping:
                for cand in candidates:
                    if re.match(cand, word):
                        natural_split[i] = replacer

        return " ".join(natural_split)
            

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        result_idxs = self.index.search(test_datum, top_k=self.top_k) 
        results = [self.data[idx] for idx, _ in result_idxs]
        return results

class LampGenerator(DataRetriever[FullDatumSub, DatumSub]): 
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        seed: int = 12345,
        ratio: float = 0.5,
        shuffle: bool = True
    ):

        # self.index: ExactMatchIndex[DatumSub, FullDatumSub] = ExactMatchIndex.create(
        #     train_data,
        #     get_content=lambda c: self.anon(c.natural),  # type: ignore
        #     get_query=lambda q: self.anon(q.natural),  # type: ignore
        # )
        # data here is the "train_eval.jsonl" file that contains all the possible train pairs, not just the sampled ones 
        self.data: List[FullDatumSub] = list(train_data)
        self.top_k = top_k
        self.best_first = best_first
        self.prng = random.Random(
            seed
        )  # a random number generator to ensure deterministic behavior

        np.random.seed(seed) 
        self.ratio = ratio 
        self.shuffle = shuffle 

        # index is grouped by type, then surface form, then index
        self.index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ex in self.data:
            self.index[ex.type][ex.natural][ex.template_idx].append(ex)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        # get the type of template
        ambig_type = test_datum.type
        all_data = self.index[ambig_type]
        all_data_keys = list(all_data.keys())
        # sample some examples 
        all_data_idxs = [i for i in range(len(all_data_keys))]
        sampled_idxs = np.random.choice(all_data_idxs, size=self.top_k, replace=False)
        # split into template 0 and template 1 based on ratio
        zero_len = int(self.ratio * len(sampled_idxs))
        zero_idxs, one_idxs = sampled_idxs[0:zero_len], sampled_idxs[zero_len:]
        zero_surfaces = [all_data_keys[i] for i in zero_idxs]
        one_surfaces = [all_data_keys[i] for i in one_idxs]
        # get template 0 and template 1 
        zero_exs = [all_data[k]['0'][0] for k in zero_surfaces]
        one_exs = [all_data[k]['1'][0] for k in one_surfaces]

        results = zero_exs + one_exs
        if self.shuffle:
            np.random.shuffle(results)
        return results

class LampGeneralizationPPRetriever(DataRetriever[FullDatumSub, DatumSub]): 
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        shuffle: bool = True,
        seed: int = 12345,
    ):
        self.index: ExactMatchIndex[DatumSub, FullDatumSub] = ExactMatchIndex.create(
            train_data,
            get_content=lambda c: c.natural,  # type: ignore
            get_query=lambda q: q.natural,  # type: ignore
        )

        self.data: List[FullDatumSub] = list(train_data)
        self.top_k = top_k
        self.best_first = best_first
        self.shuffle = True
        self.prng = random.Random(
            seed
        )  # a random number generator to ensure deterministic behavior

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

        # postprocess into a datum 
        datum_dict = test_datum.__dict__
        transitive_test_datum_dict = {k:deepcopy(v) for k,v in datum_dict.items()}
        possessive_test_datum_dict = {k:deepcopy(v) for k,v in datum_dict.items()}
        instrument_test_datum_dict = {k:deepcopy(v) for k,v in datum_dict.items()}

        transitive_test_datum_dict['natural'] = surface_form_transitive
        instrument_test_datum_dict['natural'] = surface_form_instrument
        possessive_test_datum_dict['natural'] = surface_form_possessive
        transitive_test_datum = FullDatum(**transitive_test_datum_dict)
        instrument_test_datum = FullDatum(**instrument_test_datum_dict)
        possessive_test_datum = FullDatum(**possessive_test_datum_dict)

        # search to retrieve train examples 
        for i, d in enumerate(self.data):
            if d.natural == transitive_test_datum.natural:
                transitive_result_idx = i
            if d.natural == instrument_test_datum.natural:
                instrument_result_idx = i
            if d.natural == possessive_test_datum.natural:
                possessive_result_idx = i
        # transitive_result_idx = self.index.search(transitive_test_datum, top_k = 1)[0][0]
        # possessive_result_idx = self.index.search(possessive_test_datum, top_k = 1)[0][0]
        # instrument_result_idx = self.index.search(instrument_test_datum, top_k = 1)[0][0]
        # result_idxs = self.index.search(test_datum, top_k=self.top_k) 
        try:
            idxs = [transitive_result_idx, possessive_result_idx, instrument_result_idx]
            np.random.shuffle(idxs)
            results = [self.data[idx] for idx in idxs]
        except:
            pdb.set_trace()
        return results