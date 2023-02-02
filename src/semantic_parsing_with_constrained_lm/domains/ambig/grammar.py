# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import re
from typing import Iterable, List, Optional, Set

from semantic_parsing_with_constrained_lm.domains import dfa_grammar_utils
from semantic_parsing_with_constrained_lm.domains.lispress_v2.lispress_exp import (
    BooleanExpr,
    CallExpr,
    DialogueV2,
    LambdaExpr,
    LetExpr,
    LispressExpr,
    LongExpr,
    NumberExpr,
    ReferenceExpr,
    StringExpr,
    TypeName,
)

from dataflow.core.lispress import parse_lispress


def get_nt_from_type(type_name: TypeName) -> str:
    segments = (
        str(type_name)
        .replace(" ", " SP ")
        .replace("(", " LP ")
        .replace(")", " RP ")
        .replace(".", " DOT ")
        .split()
    )
    return "_".join(segments + ["NT"])


def extract_grammar_rules(lispress_expr: LispressExpr) -> Set[str]:
    lhs = get_nt_from_type(lispress_expr.type)  # type: ignore
    rules = set()

    if isinstance(lispress_expr, (NumberExpr, LongExpr, StringExpr, BooleanExpr)):
        pass
    elif isinstance(lispress_expr, ReferenceExpr):
        rules.add(f'{lhs} -> "{lispress_expr.var_name}"')

    elif isinstance(lispress_expr, LambdaExpr):
        rhs_items = [
            f'"(lambda (^{str(lispress_expr.var_type)} {lispress_expr.var_name}) "',
            get_nt_from_type(lispress_expr.main_expr.type),  # type: ignore
            '")"',
        ]
        rhs = " ".join(rhs_items)
        rules.add(f"{lhs} -> {rhs}")
        rules.update(extract_grammar_rules(lispress_expr.main_expr))

    elif isinstance(lispress_expr, LetExpr):
        var_name_expr_nts = []
        for var_name, var_expr in lispress_expr.var_assignments:
            var_name_expr_nts.extend([f'"{var_name}"', get_nt_from_type(var_expr.type)])  # type: ignore
            rules.update(extract_grammar_rules(var_expr))
        var_name_expr_nts_str = ' " " '.join(var_name_expr_nts)
        rhs = f'"(let (" {var_name_expr_nts_str} ") " {get_nt_from_type(lispress_expr.main_expr.type)} ")"'  # type: ignore
        rules.add(f"{lhs} -> {rhs}")
        rules.update(extract_grammar_rules(lispress_expr.main_expr))

    elif isinstance(lispress_expr, CallExpr):
        rhs_items: List[str] = []
        if lispress_expr.instantiation_type is not None:
            rhs_items.append(f'"^{lispress_expr.instantiation_type} "?')

        rhs_items.append(f'"{lispress_expr.name}"')

        for k, v in lispress_expr.args:
            rhs_items.extend([f'" :{k}"?', '" "', get_nt_from_type(v.type)])  # type: ignore
            rules.update(extract_grammar_rules(v))

        rhs = " ".join(rhs_items)
        rules.add(f'{lhs} -> "(" {rhs} ")"')

    return rules



"""Modified to parse new lisp format for ambig dataset
We only need this for inferring a grammar, but we'll be defining it top-down so we don't really need it?
unless we want to test that the inferred grammar is correct"""
def parse_ambig_lispress_sexp(sexp: Sexp) -> LispressExpr:
    if isinstance(sexp, str):
        return ReferenceExpr(var_name=sexp, type=None)

    assert isinstance(sexp, list) and len(sexp) == 3, f"Failed to parse {sexp}"
    # Check for literals
    if sexp[1] == NUMBER and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return NumberExpr(type=TypeName(tpe=NUMBER), value=float(sexp[2]))
    elif sexp[1] == LONG and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return LongExpr(type=TypeName(tpe=LONG), value=int(sexp[2][:-1]))
    elif sexp[1] == STRING and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return StringExpr(type=TypeName(tpe=STRING), value=sexp[2])
    elif sexp[1] == BOOLEAN and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return BooleanExpr(type=TypeName(tpe=BOOLEAN), value=sexp[2] == TRUE)
    elif isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return ReferenceExpr(type=TypeName(tpe=REFERENCE), var_name=sexp[2])

    elif sexp[0] == LAMBDA:
        assert len(sexp[1]) == 1 and len(sexp[1][0]) == 3 and sexp[1][0][0] == META_CHAR
        var_type = TypeName(tpe=sexp[1][0][1])
        var_name = sexp[1][0][2]
        main_expr = sexp[2]
        return LambdaExpr(
            var_name=var_name,  # type: ignore
            var_type=var_type,
            main_expr=parse_ambig_lispress_sexp(main_expr),
            type=None,
        )

    # Check for let expression
    elif sexp[0] == LET and isinstance(sexp[1], list) and isinstance(sexp[2], list):
        var_assignment_list = sexp[1]
        assert len(var_assignment_list) % 2 == 0
        assert all([isinstance(item, str) for item in var_assignment_list[0::2]])
        var_assignments = [
            (
                var_assignment_list[index],
                parse_ambig_lispress_sexp(var_assignment_list[index + 1]),
            )
            for index in range(0, len(var_assignment_list), 2)
        ]
        main_expr = sexp[2]
        return LetExpr(
            var_assignments=var_assignments,  # type: ignore
            main_expr=parse_ambig_lispress_sexp(main_expr),
            type=None,
        )

    else:
        # Check for CallExp
        if isinstance(sexp[2], list) and len(sexp[2]) > 0 and sexp[0] == META_CHAR:
            return_type = TypeName(tpe=sexp[1])
            function_name = None
            function_instantiation_type = None
            function_sexp, *key_value_args = sexp[2]
            if isinstance(function_sexp, str):
                function_name = function_sexp
            elif (
                isinstance(function_sexp, list)
                and len(function_sexp) == 3
                and function_sexp[0] == META_CHAR
                and isinstance(function_sexp[2], str)
            ):
                function_instantiation_type = TypeName(tpe=function_sexp[1])
                function_name = function_sexp[2]

            assert function_name is not None
            assert len(key_value_args) % 2 == 0 and all(
                [
                    isinstance(item, str) and _is_named_arg(item)
                    for item in key_value_args[0::2]
                ]
            )

            args = []
            for index in range(0, len(key_value_args), 2):
                key = _named_arg_to_key(key_value_args[index])  # type: ignore
                value = parse_ambig_lispress_sexp(key_value_args[index + 1])
                args.append((key, value))

            return CallExpr(
                type=return_type,
                name=function_name,
                args=args,
                instantiation_type=function_instantiation_type,
            )

    raise ValueError(f"Could not parse: {sexp}")

def extract_grammar(
    dataflow_dialogues: Iterable[DialogueV2],
    whitelisted_dialogue_ids: Optional[Set[str]] = None,
) -> Set[str]:
    grammar_rules = set()
    for dataflow_dialogue in dataflow_dialogues:
        if (
            whitelisted_dialogue_ids is not None
            and dataflow_dialogue.dialogue_id not in whitelisted_dialogue_ids
        ):
            continue

        for turn in dataflow_dialogue.turns:
            lispress_expr = parse_lispress(turn.lispress)
            grammar_rules.update(extract_grammar_rules(lispress_expr))
            root_type_nt = get_nt_from_type(lispress_expr.type)  # type: ignore
            grammar_rules.add(f'start -> " " {root_type_nt}')

            # find string literals
            for match in re.finditer(r'Path\.apply "([^"]*)"', turn.lispress):
                start = match.start(1)
                end = match.end(1)
                item = turn.lispress[start:end]
                # We use `repr` because the .cfg parser uses `ast.literal_eval`
                # to parse the strings, since that will handle backslash escape
                # sequences. Without `repr` the resulting grammar will have one
                # level of escaping removed.
                grammar_rules.add(f"path_literal -> {repr(item)}")
    grammar_rules.update(
        [
            'Boolean_NT -> "true"',
            'Boolean_NT -> "false"',
            r'String_NT -> "\"" (String_NT_content | path_literal | "output" | "place" | "start") "\""',
            # Lispress V2 string literals are JSON string literals, so we follow this grammar:
            # https://datatracker.ietf.org/doc/html/rfc8259#section-7
            r'String_NT_content -> ([[\u0020-\U0010FFFF]--[\u0022\u005C]] | "\\" (["\u005C/bfnrt] | u[0-9A-Fa-f]{4}))*',
            'Number_NT -> ("0" | [1-9][0-9]*) ("." [0-9]+)?',
            'Long_NT -> ("0" | [1-9][0-9]*) "L"',
        ]
    )
    return grammar_rules


create_partial_parse_builder = functools.partial(
    dfa_grammar_utils.create_partial_parse_builder,
    utterance_nonterm_name="String_NT_content",
)
