"""Parse a Brick .ttl model into per-feature semantic metadata and a graph neighbourhood.

Works on the LBNL SDAHU model (`data/raw/SDAHU/ttl/*.ttl`, Brick v1.2, namespace <bldg-59#>).
Uses rdflib. Every point (`brick:hasPoint` target) becomes a feature; we resolve:
  - measurement_type / component_type : from the Brick class of the point / of its owning component;
  - component            : the equipment/part instance that `hasPoint` the point;
  - relations            : the point's own (relation, target) edges PLUS the owning component's
                           structural edges (hasPart / feeds / partOf), which is the topology the
                           KG embedding needs and the text baseline cannot see.

The returned metadata feeds `schema.FeatureMeta`. The measurement-type string is derived from the
Brick class name by a small, transparent rule (splitting the CamelCase class), so the metadata and
KG conditions stay auditable rather than depending on a hidden mapping table.
"""
from __future__ import annotations

import re
from pathlib import Path

from rdflib import Graph, RDF, Namespace, URIRef

BRICK = Namespace("https://brickschema.org/schema/Brick#")

# structural relations that describe topology (used by the KG encoder, hidden from text/metadata)
STRUCTURAL = ("feeds", "hasPart", "isPartOf", "hasPoint", "isPointOf", "controls",
              "isFedBy", "hasLocation", "isLocationOf")

_MEAS_WORDS = ("Temperature", "Pressure", "Flow", "Power", "Speed", "Position", "Occupancy",
               "Humidity", "Energy", "Status", "Setpoint", "Command")


def _local(uri: URIRef | str) -> str:
    s = str(uri)
    return re.split(r"[#/]", s)[-1]


def _class_of(g: Graph, node: URIRef) -> str:
    for _, _, o in g.triples((node, RDF.type, None)):
        return _local(o)
    return ""


def _measurement_type(brick_class: str) -> str:
    """Coarse measurement quantity from a Brick point class, e.g.
    'Supply_Air_Temperature_Sensor' -> 'Temperature', 'Outside_Air_Flow_Sensor' -> 'Flow'.
    Falls back to the class stem with Sensor/Command/Setpoint stripped."""
    for w in _MEAS_WORDS:
        if w in brick_class:
            return w
    stem = re.sub(r"_(Sensor|Command|Setpoint|Status|Position)$", "", brick_class)
    return stem.replace("_", " ").strip()


def parse_brick(ttl_path: str | Path, dataset: str, ontology_source: str = "Brick-1.2") -> list[dict]:
    """Return one metadata dict per point (feature) in the model, ready for FeatureMeta(**d).

    A dict has: feature_id, name, dataset, unit(''), measurement_type, component, component_type,
    ontology_id, ontology_source, description, relations (tuple of (rel, target_local)).
    `unit` is left blank here; units come from the dataset's own point table and are merged in the
    dataset loader (Brick v1.2 does not carry units on these points).
    """
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    # map point -> owning component via hasPoint (and its inverse isPointOf)
    owner: dict[URIRef, URIRef] = {}
    for comp, _, pt in g.triples((None, BRICK.hasPoint, None)):
        owner[pt] = comp
    for pt, _, comp in g.triples((None, BRICK.isPointOf, None)):
        owner.setdefault(pt, comp)

    points = sorted(owner.keys(), key=lambda u: _local(u))
    out: list[dict] = []
    for pt in points:
        comp = owner.get(pt)
        pt_class = _class_of(g, pt)
        comp_class = _class_of(g, comp) if comp is not None else ""
        local = _local(pt)

        # relations of the point itself, plus the owning component's structural edges
        rels: list[tuple[str, str]] = []
        for _, p, o in g.triples((pt, None, None)):
            if p == RDF.type:
                continue
            rels.append((_local(p), _local(o)))
        if comp is not None:
            rels.append(("isPointOf", _local(comp)))
            for _, p, o in g.triples((comp, None, None)):
                pl = _local(p)
                if pl in STRUCTURAL and o != pt:
                    rels.append((f"component.{pl}", _local(o)))

        name = local.replace("_", " ").title()
        desc = f"{pt_class.replace('_', ' ')}"
        if comp_class:
            desc += f" on {comp_class.replace('_', ' ')}"
        out.append(dict(
            feature_id=f"{dataset}:{local}",
            name=name,
            dataset=dataset,
            unit="",
            measurement_type=_measurement_type(pt_class),
            component=_local(comp) if comp is not None else "",
            component_type=comp_class,
            ontology_id=f"bldg:{local}",
            ontology_source=ontology_source,
            description=desc,
            relations=tuple(rels),
        ))
    return out


if __name__ == "__main__":
    import sys
    ttl = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw/SDAHU/ttl/LBNL_FDD_Data_Sets_SDAHU_ttl.ttl"
    rows = parse_brick(ttl, dataset="LBNL_SDAHU")
    print(f"{len(rows)} points parsed from {ttl}\n")
    for r in rows:
        print(f"  {r['feature_id']:24s} type={r['measurement_type']:12s} "
              f"comp={r['component']:20s} ({r['component_type']}) | {len(r['relations'])} rels")
