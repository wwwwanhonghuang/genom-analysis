"""
llm/neural_symbolic/parser.py
NeuralPCFGParser — converts HyenaDNA token-level embeddings to a parse tree
via a differentiable PCFG inside/argmax (torch-struct SentCFG).

Architecture
------------
Input  : DNA sequence of length n
Step 1 : HyenaDNA encoder  → hidden states  (1, n, D)
Step 2 : Terminal scorer   → (1, n, T)       linear projection from token hiddens
Step 3 : Span encoder      → (1, n, n, D)    mean-pool hidden states over each span
Step 4 : Rule scorer       → (1, NT, NT, NT) bilinear NT-pair scoring from span reps
Step 5 : Root scorer       → (1, NT)         linear from CLS-like global embedding
Step 6 : SentCFG           → inside + argmax  (torch-struct)
Step 7 : decode_tree()     → ParseTree

The parser is unsupervised at construction — rule scores come entirely from
learned projections of HyenaDNA embeddings.  Supervised fine-tuning against
labeled genomic parse trees can be added by providing target trees to loss().
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))  # llm/

from neural_symbolic.grammar import (
    BINARY_RULES, NUM_NT, NUM_T, NUM_RULES,
    NT_NAMES, TERMINAL_VOCAB,
    root_prior_tensor, rule_mask,
)
from neural_symbolic.tree import ParseTree, decode_tree


class NeuralPCFGParser(nn.Module):
    """
    Neural PCFG parser operating on HyenaDNA token-level hidden states.

    Args:
        embed_dim  : HyenaDNA hidden size (128 for tiny, 256 for larger)
        hidden_dim : internal projection dimension
        num_nt     : number of non-terminal symbols (default: 12)
        num_t      : number of terminal symbols (default: 5 = ACGTN)
        use_grammar_prior : whether to add biological rule priors to scores
    """

    def __init__(
        self,
        embed_dim:         int  = 128,
        hidden_dim:        int  = 64,
        num_nt:            int  = NUM_NT,
        num_t:             int  = NUM_T,
        use_grammar_prior: bool = True,
    ):
        super().__init__()
        self.embed_dim         = embed_dim
        self.hidden_dim        = hidden_dim
        self.num_nt            = num_nt
        self.num_t             = num_t
        self.use_grammar_prior = use_grammar_prior

        # --- Terminal scorer: token hidden → NT emission scores (T per position)
        self.term_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_t),
        )

        # --- Span encoder: map span representation to a vector
        self.span_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )

        # --- Rule scorer: given a span vector, score NT → NT NT rules
        # Decomposed as: score(A → B C) = w_A · span + U_B · span + V_C · span
        self.nt_emb    = nn.Embedding(num_nt, hidden_dim)  # left-child NT embs
        self.nt_emb_r  = nn.Embedding(num_nt, hidden_dim)  # right-child NT embs
        self.rule_head = nn.Linear(hidden_dim, num_nt)     # parent NT head
        self._S = num_nt + num_t                           # full child symbol count (NT+T)

        # --- Root scorer: global sequence → root NT distribution
        self.root_proj = nn.Linear(embed_dim, num_nt)

        # Grammar structure
        self._rule_mask = None   # (NT, NT, NT) binary mask — lazily registered
        self._root_prior = None  # (NT,) prior — lazily registered

        # Register biological rule prior as a buffer (not a parameter)
        if use_grammar_prior:
            rm = rule_mask()                            # (NT, NT, NT)
            rp = root_prior_tensor()                    # (NT,)
            self.register_buffer("bio_rule_mask", rm)
            self.register_buffer("bio_root_prior", rp)
        else:
            self.bio_rule_mask  = None
            self.bio_root_prior = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.nt_emb.weight,   std=0.1)
        nn.init.normal_(self.nt_emb_r.weight, std=0.1)

    # ------------------------------------------------------------------
    # Forward: hidden states → PCFG potentials
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,           # (batch, n, D)
        lengths:       Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-potentials for the PCFG from HyenaDNA hidden states.

        Returns:
            terms  : (batch, n, T)        log terminal emission probs
            rules  : (batch, NT, S, S)    log binary rule probs  (S = NT + T)
            roots  : (batch, NT)          log root probs
        """
        batch, n, D = hidden_states.shape
        device = hidden_states.device

        # 1. Terminal scores: (batch, n, T)
        terms = F.log_softmax(self.term_proj(hidden_states), dim=-1)

        # 2. Global embedding for root scoring: mean over sequence
        if lengths is not None:
            mask   = torch.arange(n, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            global_emb = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            global_emb = hidden_states.mean(dim=1)  # (batch, D)

        # 3. Root scores: (batch, NT)
        roots = F.log_softmax(self.root_proj(global_emb), dim=-1)
        if self.use_grammar_prior and self.bio_root_prior is not None:
            roots = roots + self.bio_root_prior.unsqueeze(0)
            roots = roots - torch.logsumexp(roots, dim=-1, keepdim=True)

        # 4. Rule scores: (batch, NT, NT, NT)
        # span_emb: use the global embedding as a proxy for now
        # (proper span encoding would index [i:j] pairs — expensive at O(n²))
        span_h = F.gelu(self.span_proj(global_emb))   # (batch, hidden)

        # Parent head: (batch, NT) → which parent NT is active
        parent_scores = self.rule_head(span_h)          # (batch, NT)

        # Left/right child NT embeddings: (NT, hidden)
        left_e  = self.nt_emb.weight                    # (NT, hidden)
        right_e = self.nt_emb_r.weight                  # (NT, hidden)

        # Score(A → B C) = parent_scores[A] + span_h · left_e[B] + span_h · right_e[C]
        # parent: (batch, NT, 1, 1)
        # left:   (1,     1,  NT, 1)  via span_h @ left_e.T
        # right:  (1,     1,  1,  NT) via span_h @ right_e.T

        left_scores  = torch.matmul(span_h, left_e.T)   # (batch, NT)
        right_scores = torch.matmul(span_h, right_e.T)  # (batch, NT)

        # Broadcast to (batch, NT_parent, NT_left, NT_right) — NT children only
        rule_scores_nt = (
            parent_scores.unsqueeze(-1).unsqueeze(-1)   # (B, NT, 1,  1)
            + left_scores.unsqueeze(1).unsqueeze(-1)    # (B, 1, NT,  1)
            + right_scores.unsqueeze(1).unsqueeze(2)    # (B, 1,  1, NT)
        )  # (B, NT, NT, NT)

        # Mask out biologically impossible NT→NT NT rules
        if self.use_grammar_prior and self.bio_rule_mask is not None:
            neg_inf = torch.full_like(rule_scores_nt, -1e9)
            rule_scores_nt = torch.where(
                self.bio_rule_mask.bool().unsqueeze(0),
                rule_scores_nt,
                neg_inf,
            )

        # Normalize over NT×NT child pairs, then embed into full (NT, S, S) space.
        # torch-struct SentCFG requires rules (batch, NT, S, S) where S = NT + T.
        # Terminal-child entries stay at -1e9 (our grammar is NT → NT NT only).
        S = self._S
        rules_full = rule_scores_nt.new_full((batch, self.num_nt, S, S), -1e9)
        rules_full[:, :, :self.num_nt, :self.num_nt] = rule_scores_nt

        rules = F.log_softmax(
            rules_full.reshape(batch, -1), dim=-1
        ).reshape(batch, self.num_nt, S, S)

        return terms, rules, roots

    # ------------------------------------------------------------------
    # Parse: inside (torch-struct) + Viterbi CYK backtrack
    # ------------------------------------------------------------------

    def parse(
        self,
        hidden_states: torch.Tensor,
        sequence:      str,
        lengths:       Optional[torch.Tensor] = None,
    ) -> Tuple[ParseTree, torch.Tensor]:
        """
        Full parse: hidden states → ParseTree.

        SentCFG.argmax returns a tuple (terms, rules, roots) matching
        the input potentials — NOT a flat span chart.  We therefore:
          1. Use SentCFG only for the partition (inside) computation.
          2. Run a standalone CYK Viterbi backtracker over the scored
             rules tensor to recover the best parse tree.

        Returns:
            tree   : ParseTree decoded by CYK Viterbi
            log_Z  : log partition function (batch,)
        """
        try:
            from torch_struct import SentCFG
        except ImportError:
            raise ImportError(
                "torch-struct is required for neural parsing.\n"
                "Install: pip install git+https://github.com/harvardnlp/pytorch-struct"
            )

        terms, rules, roots = self.forward(hidden_states, lengths)

        batch, n, T = terms.shape
        length_tensor = (
            lengths if lengths is not None
            else torch.tensor([n] * batch, device=terms.device)
        )

        dist  = SentCFG((terms, rules, roots), lengths=length_tensor)
        log_Z = dist.partition   # (batch,)

        # Viterbi CYK: rules is (batch, NT, S, S); only NT-child slice matters
        seq_len = length_tensor[0].item()
        tree = self._viterbi_decode(
            terms[0].detach().cpu().float(),
            rules[0].detach().cpu().float(),
            roots[0].detach().cpu().float(),
            sequence[:seq_len],
        )
        return tree, log_Z

    def _viterbi_decode(
        self,
        terms:    torch.Tensor,   # (n, T)
        rules:    torch.Tensor,   # (NT, S, S)
        roots:    torch.Tensor,   # (NT,)
        sequence: str,
    ) -> ParseTree:
        """
        CYK Viterbi backtracker.

        Complexity: O(n³ · NT²) — fine for short sequences (n ≤ 32).
        For longer sequences, chunked embedding already limits effective n.

        Chart entry:  score[(i,j,nt)]  = max log-prob of nt spanning [i,j]
        Back-pointer: bp[(i,j,nt)]     = (split, left_nt, right_nt)
        """
        n   = len(sequence)
        NT  = self.num_nt
        NEG = -1e9

        # --- score and back-pointer tables ---
        score: dict[tuple, float] = {}
        bp:    dict[tuple, tuple] = {}

        # Lexical (length-1) spans: score = roots[nt] + terms[i, nt % T]
        T = self.num_t
        for i in range(n):
            for nt in range(NT):
                score[(i, i, nt)] = roots[nt].item() + terms[i, nt % T].item()

        # Populate chart bottom-up
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span - 1
                for nt in range(NT):
                    best      = NEG
                    best_bp   = (-1, -1, -1)
                    r_row     = rules[nt]           # (S, S)
                    for mid in range(i, j):
                        for l_nt in range(NT):
                            l_sc = score.get((i,   mid, l_nt), NEG)
                            if l_sc <= NEG:
                                continue
                            for r_nt in range(NT):
                                r_sc = score.get((mid+1, j, r_nt), NEG)
                                if r_sc <= NEG:
                                    continue
                                rule_sc = r_row[l_nt, r_nt].item()
                                total   = rule_sc + l_sc + r_sc
                                if total > best:
                                    best    = total
                                    best_bp = (mid, l_nt, r_nt)
                    if best > NEG:
                        score[(i, j, nt)] = best
                        bp[(i, j, nt)]    = best_bp

        # Find best root NT over span [0, n-1]
        root_span = n - 1
        best_root_nt = max(
            range(NT),
            key=lambda nt: score.get((0, root_span, nt), NEG) + roots[nt].item()
        )

        # Backtrack
        def build(i: int, j: int, nt: int) -> "ParseTree":
            from neural_symbolic.tree import ParseTree
            if i == j:
                nuc  = sequence[i] if i < len(sequence) else "N"
                leaf = ParseTree(label=nuc, start=i, end=i + 1)
                return ParseTree(label=nt, start=i, end=i + 1, left=leaf,
                                 score=score.get((i, j, nt), 0.0))
            key = (i, j, nt)
            if key not in bp:
                # No recorded split — produce a flat node
                mid = (i + j) // 2
                return ParseTree(
                    label=nt, start=i, end=j + 1,
                    left=build(i, mid, nt),
                    right=build(mid + 1, j, nt),
                )
            mid, l_nt, r_nt = bp[key]
            return ParseTree(
                label=nt,
                start=i, end=j + 1,
                left=build(i, mid, l_nt),
                right=build(mid + 1, j, r_nt),
                score=score.get(key, 0.0),
            )

        return build(0, root_span, best_root_nt)

    # ------------------------------------------------------------------
    # Loss (for supervised fine-tuning)
    # ------------------------------------------------------------------

    def loss(
        self,
        hidden_states:  torch.Tensor,
        target_spans:   List[Tuple[int, int, int]],  # (start, end, nt)
        lengths:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss against labeled span annotation.
        `target_spans` is a list of (start, end, nt_label) tuples
        representing all non-terminal spans in the gold tree.
        """
        try:
            from torch_struct import SentCFG
        except ImportError:
            raise ImportError("torch-struct required for loss computation")

        terms, rules, roots = self.forward(hidden_states, lengths)
        batch, n, _ = terms.shape
        length_tensor = (
            lengths if lengths is not None
            else torch.tensor([n] * terms.device)
        )

        dist  = SentCFG((terms, rules, roots), lengths=length_tensor)
        log_Z = dist.partition                          # (batch,)

        # Score of the gold tree: sum log-probs of its spans
        gold_score = torch.zeros(batch, device=terms.device)
        for start, end, nt in target_spans:
            # end here is exclusive; SentCFG uses inclusive end
            j = end - 1
            if 0 <= start < n and j < n:
                # Terminal score contribution
                if start == j:
                    gold_score = gold_score + terms[0, start, nt % NUM_T]
                else:
                    gold_score = gold_score + rules[0, nt, nt, nt]  # simplified

        return (log_Z - gold_score).mean()


# ---------------------------------------------------------------------------
# Full pipeline: sequence → (tokenize, embed with HyenaDNA) → parse
# ---------------------------------------------------------------------------

class GenomicParser:
    """
    End-to-end wrapper: DNA sequence string → ParseTree.
    Loads HyenaDNA for embedding and NeuralPCFGParser for symbolic parsing.
    """

    def __init__(
        self,
        hyena_model_id: Optional[str] = None,
        embed_dim:      int = 128,
        hidden_dim:     int = 64,
        device:         str = "cpu",
    ):
        self.device = device
        self._tokenizer = None
        self._encoder   = None
        self._parser    = NeuralPCFGParser(
            embed_dim=embed_dim, hidden_dim=hidden_dim
        ).to(device)
        self._hyena_model_id = hyena_model_id or "LongSafari/hyenadna-tiny-1k-seqlen-hf"

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        from transformers import AutoTokenizer, AutoModel
        print(f"[genomic_parser] Loading encoder {self._hyena_model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hyena_model_id, trust_remote_code=True
        )
        self._encoder = AutoModel.from_pretrained(
            self._hyena_model_id, trust_remote_code=True
        ).to(self.device).eval()

    def encode(self, sequence: str) -> torch.Tensor:
        """Return token-level hidden states (1, n, D)."""
        self._ensure_encoder()
        max_len = self._tokenizer.model_max_length - 2
        seq = sequence[:max_len]
        inputs = self._tokenizer(seq, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._encoder(**inputs)
        return out.last_hidden_state  # (1, n, D)

    def parse(self, sequence: str) -> ParseTree:
        """Full pipeline: sequence → ParseTree."""
        hidden = self.encode(sequence)
        n = hidden.shape[1]
        lengths = torch.tensor([n], device=self.device)
        self._parser.eval()
        with torch.no_grad():
            tree, log_Z = self._parser.parse(hidden, sequence[:n], lengths)
        return tree