from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class NodePat:
    var: str
    label: Optional[str]
    props: Dict[str, str]


@dataclass
class RelPat:
    types: Optional[List[str]]  # allowed relationship types; None means any
    min_hops: int
    max_hops: int
    directed: bool


@dataclass
class MatchPattern:
    left: NodePat
    rel: RelPat
    right: NodePat


@dataclass
class WherePredicate:
    kind: str  # 'prop_eq' | 'similar'
    args: Tuple


@dataclass
class Query:
    match: MatchPattern
    where: List[WherePredicate]
    returns: List[str]
    limit: int


_SPACE = r"\s*"
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_LABEL = r"[A-Za-z_][A-Za-z0-9_]*"
_STRING = r"'([^']*)'"


def _parse_node(text: str) -> NodePat:
    # (var:Label {k:'v',k2:'v2'}) | (var) | (:Label) | (var:Label)
    var = None
    label = None
    props: Dict[str, str] = {}

    # remove parens
    inner = text.strip()[1:-1].strip()
    # split props
    props_part = None
    if '{' in inner and '}' in inner:
        props_part = inner[inner.index('{') + 1: inner.rindex('}')]
        inner = (inner[: inner.index('{')].strip())
        if props_part:
            for kv in props_part.split(','):
                if not kv.strip():
                    continue
                k, v = kv.split(':', 1)
                props[k.strip()] = v.strip().strip("'")

    if ':' in inner:
        parts = inner.split(':', 1)
        var = parts[0].strip() or None
        label = parts[1].strip() or None
    else:
        var = inner.strip() or None
    return NodePat(var=var or '', label=label, props=props)


def _parse_rel(text: str) -> RelPat:
    # -[:TYPE*min..max]-> or -[*min..max]-> or -[:TYPE]->
    directed = text.endswith('>')
    inner = text.strip()[1:-1].strip('-').strip('>').strip()
    rtypes: Optional[List[str]] = None
    min_hops = 1
    max_hops = 1
    if inner.startswith(':'):
        inner = inner[1:]
        if '*' in inner:
            type_part, hop = inner.split('*', 1)
        else:
            type_part, hop = inner, None
    else:
        type_part, hop = None, inner[1:] if inner.startswith('*') else None
    if type_part:
        # support alternation: TYPE1|TYPE2|TYPE3 and also forms with leading colons
        parts = [p.strip() for p in type_part.replace(':', '').split('|') if p.strip()]
        rtypes = [p.upper() for p in parts] if parts else None
    if hop:
        if '..' in hop:
            a, b = hop.split('..', 1)
            min_hops = int(a)
            max_hops = int(b)
        else:
            min_hops = max_hops = int(hop)
    return RelPat(types=rtypes, min_hops=min_hops, max_hops=max_hops, directed=directed)


def parse(query: str) -> Query:
    text = query.strip()
    # MATCH (a:Label {...})-[:REL*min..max]->(b:Label) WHERE ... RETURN a,b LIMIT 10
    m = re.search(r"MATCH\s*(\([^)]*\))\s*-(\[[^\]]*\]>?)\s*(\([^)]*\))", text, re.IGNORECASE)
    if not m:
        raise ValueError('Invalid MATCH pattern')
    left_s, rel_s, right_s = m.group(1), m.group(2), m.group(3)
    left = _parse_node(left_s)
    rel = _parse_rel(rel_s)
    right = _parse_node(right_s)
    match = MatchPattern(left=left, rel=rel, right=right)

    where_preds: List[WherePredicate] = []
    w = re.search(r"WHERE\s*(.*)\s*(RETURN|$)", text, re.IGNORECASE)
    if w:
        cond = w.group(1)
        # prop equality: var.prop='value'
        for m2 in re.finditer(rf"({_IDENT})\.({_IDENT})\s*=\s*{_STRING}", cond):
            where_preds.append(WherePredicate(kind='prop_eq', args=(m2.group(1), m2.group(2), m2.group(3)) ))
        # SIMILAR(var,'query',topK?,threshold?)
        ms = re.search(rf"SIMILAR\(\s*({_IDENT})\s*,\s*{_STRING}\s*(?:,\s*(\d+)\s*)?(?:,\s*([0-9.]+)\s*)?\)", cond, re.IGNORECASE)
        if ms:
            var = ms.group(1)
            q = ms.group(2)
            topk = int(ms.group(3)) if ms.group(3) else 50
            thr = float(ms.group(4)) if ms.group(4) else 0.2
            where_preds.append(WherePredicate(kind='similar', args=(var, q, topk, thr)))

    r = re.search(r"RETURN\s*([^L]+)(?:LIMIT|$)", text, re.IGNORECASE)
    returns = [v.strip() for v in (r.group(1) if r else '').split(',') if v.strip()] or ['*']
    lim = 25
    lm = re.search(r"LIMIT\s*(\d+)", text, re.IGNORECASE)
    if lm:
        lim = int(lm.group(1))
    return Query(match=match, where=where_preds, returns=returns, limit=lim)

