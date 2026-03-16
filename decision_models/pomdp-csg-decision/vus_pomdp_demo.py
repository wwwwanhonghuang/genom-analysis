from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import math


STATES = ["benign", "pathogenic"]

ACTIONS = [
    "decide_benign",
    "decide_pathogenic",
    "defer",
]

REWARD = {
    "decide_benign": {"benign": 10.0, "pathogenic": -20.0},
    "decide_pathogenic": {"benign": -20.0, "pathogenic": 10.0},
    "defer": {"benign": -1.5, "pathogenic": -1.5},
}

# Toy likelihood tables for parse-derived and evidence-derived features
LIKELIHOODS = {
    "variant_type": {
        "synonymous": {"benign": 0.75, "pathogenic": 0.05},
        "missense": {"benign": 0.22, "pathogenic": 0.50},
        "splice": {"benign": 0.03, "pathogenic": 0.45},
    },
    "population": {
        "common": {"benign": 0.85, "pathogenic": 0.10},
        "rare": {"benign": 0.15, "pathogenic": 0.90},
    },
    "functional": {
        "neutral": {"benign": 0.80, "pathogenic": 0.15},
        "damaging": {"benign": 0.20, "pathogenic": 0.85},
    },
    "in_silico": {
        "tolerated": {"benign": 0.75, "pathogenic": 0.20},
        "deleterious": {"benign": 0.25, "pathogenic": 0.80},
    },

    # parse-derived features
    "has_start_codon": {
        True: {"benign": 0.45, "pathogenic": 0.55},
        False: {"benign": 0.55, "pathogenic": 0.45},
    },
    "has_stop_codon": {
        True: {"benign": 0.45, "pathogenic": 0.55},
        False: {"benign": 0.55, "pathogenic": 0.45},
    },
    "frameshift_like_length": {
        True: {"benign": 0.10, "pathogenic": 0.80},
        False: {"benign": 0.90, "pathogenic": 0.20},
    },
    "splice_donor_GT": {
        True: {"benign": 0.35, "pathogenic": 0.65},
        False: {"benign": 0.65, "pathogenic": 0.35},
    },
    "splice_acceptor_AG": {
        True: {"benign": 0.35, "pathogenic": 0.65},
        False: {"benign": 0.65, "pathogenic": 0.35},
    },
    "codon_count_bin": {
        "short": {"benign": 0.35, "pathogenic": 0.55},
        "medium": {"benign": 0.45, "pathogenic": 0.35},
        "long": {"benign": 0.20, "pathogenic": 0.10},
    },
}


@dataclass
class VariantSample:
    variant_id: str
    genomic: Dict[str, Any]
    evidence: Dict[str, Any]
    label_proxy: Optional[str] = None


def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    z = sum(dist.values())
    if z <= 0:
        raise ValueError("Normalization constant is zero.")
    return {k: v / z for k, v in dist.items()}


def codons(seq: str) -> List[str]:
    seq = seq.upper()
    return [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3]


def parse_sequence_features(sequence: str) -> Dict[str, Any]:
    """
    Toy parsing / ordered-sequence analysis.
    This is not a full CFG parser, but it explicitly uses sequence order.
    """
    seq = sequence.upper()
    cds = codons(seq)

    start_codons = {"ATG"}
    stop_codons = {"TAA", "TAG", "TGA"}

    has_start = len(cds) > 0 and cds[0] in start_codons
    has_stop = len(cds) > 0 and cds[-1] in stop_codons
    frameshift_like = (len(seq) % 3) != 0

    # toy splice motif checks from local sequence arrangement
    splice_donor_GT = "GT" in seq
    splice_acceptor_AG = "AG" in seq

    codon_count = len(cds)
    if codon_count <= 3:
        codon_bin = "short"
    elif codon_count <= 8:
        codon_bin = "medium"
    else:
        codon_bin = "long"

    return {
        "has_start_codon": has_start,
        "has_stop_codon": has_stop,
        "frameshift_like_length": frameshift_like,
        "splice_donor_GT": splice_donor_GT,
        "splice_acceptor_AG": splice_acceptor_AG,
        "codon_count_bin": codon_bin,
        "codons": cds,
    }


def posterior_from_sample(sample: VariantSample,
                          prior: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if prior is None:
        prior = {"benign": 0.5, "pathogenic": 0.5}

    logp = {s: math.log(prior[s]) for s in STATES}

    observations = {}

    # raw genomic features
    for k, v in sample.genomic.items():
        if k in LIKELIHOODS and v is not None:
            observations[k] = v

    # evidence features
    for k, v in sample.evidence.items():
        if k in LIKELIHOODS and v is not None:
            observations[k] = v

    # sequence parsing features
    seq = sample.genomic.get("sequence")
    if seq:
        parsed = parse_sequence_features(seq)
        for k, v in parsed.items():
            if k in LIKELIHOODS:
                observations[k] = v

    for obs_type, obs_value in observations.items():
        for s in STATES:
            p = LIKELIHOODS[obs_type][obs_value][s]
            logp[s] += math.log(p)

    m = max(logp.values())
    unnorm = {s: math.exp(logp[s] - m) for s in STATES}
    return normalize(unnorm)


def expected_reward(belief: Dict[str, float], action: str) -> float:
    return sum(belief[s] * REWARD[action][s] for s in STATES)


def choose_action(belief: Dict[str, float]) -> Dict[str, Any]:
    utilities = {a: expected_reward(belief, a) for a in ACTIONS}
    best_action = max(utilities, key=utilities.get)
    return {"best_action": best_action, "utilities": utilities}


def explain_sample(sample: VariantSample) -> None:
    belief = posterior_from_sample(sample)
    decision = choose_action(belief)
    parsed = parse_sequence_features(sample.genomic["sequence"])

    print("=" * 72)
    print(f"Variant: {sample.variant_id}")

    print("\n[Raw genomic observation]")
    for k, v in sample.genomic.items():
        print(f"  {k}: {v}")

    print("\n[Sequence-derived parse features]")
    for k, v in parsed.items():
        print(f"  {k}: {v}")

    print("\n[Evidence observations]")
    for k, v in sample.evidence.items():
        print(f"  {k}: {v}")

    print("\n[Posterior belief]")
    print(f"  P(benign | o)     = {belief['benign']:.4f}")
    print(f"  P(pathogenic | o) = {belief['pathogenic']:.4f}")

    print("\n[Expected utility]")
    for a, u in decision["utilities"].items():
        print(f"  {a:<18}: {u:.4f}")

    print("\n[Recommended action]")
    print(f"  {decision['best_action']}")

    if sample.label_proxy is not None:
        print("\n[Label proxy]")
        print(f"  {sample.label_proxy}")

    print("=" * 72)
    print()


def make_demo_dataset() -> List[VariantSample]:
    return [
        VariantSample(
            variant_id="var_seq_001",
            genomic={
                "gene": "GENE_A",
                "variant_type": "missense",
                "sequence": "ATGGCCAAATGA",   # ATG GCC AAA TGA
            },
            evidence={
                "population": "rare",
                "functional": "damaging",
                "in_silico": "deleterious",
            },
            label_proxy="pathogenic",
        ),
        VariantSample(
            variant_id="var_seq_002",
            genomic={
                "gene": "GENE_B",
                "variant_type": "synonymous",
                "sequence": "ATGGCCAAACCC",   # no stop codon at end
            },
            evidence={
                "population": "common",
                "functional": "neutral",
                "in_silico": "tolerated",
            },
            label_proxy="benign",
        ),
        VariantSample(
            variant_id="var_seq_003",
            genomic={
                "gene": "GENE_C",
                "variant_type": "splice",
                "sequence": "CAGGTAAAGT",     # splice-like motifs
            },
            evidence={
                "population": "rare",
                "functional": None,
                "in_silico": "deleterious",
            },
            label_proxy=None,
        ),
        VariantSample(
            variant_id="var_seq_004",
            genomic={
                "gene": "GENE_D",
                "variant_type": "missense",
                "sequence": "ATGGCCAATG",     # length % 3 != 0
            },
            evidence={
                "population": "common",
                "functional": "damaging",
                "in_silico": "deleterious",
            },
            label_proxy=None,
        ),
    ]


def main():
    dataset = make_demo_dataset()
    for sample in dataset:
        explain_sample(sample)


if __name__ == "__main__":
    main()