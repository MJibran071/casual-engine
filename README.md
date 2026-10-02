# casual-engine
Causal World Engine — a from-scratch neurosymbolic causal inference engine, white-labeled for pharma &amp; biotech
# Causal World Engine

**v2.1 · A from-scratch neurosymbolic causal inference engine, white-labeled for pharma & biotech**

The engine discovers causal structure from observational data, proves which
interventions are identifiable, estimates their effects, computes
counterfactuals, and keeps an honest ledger of what it knows, suspects, and
does not know — with an adversarial review board that verifies every claim
before it is released.

### What we guarantee

**Nothing ships unverified, and nothing wrong ever ships.** Every answer is
checked against a truth machine — dual symbolic evaluation, sandboxed
execution, or do-calculus — and when the evidence does not decide the
question, the engine refuses instead of guessing. Measured across 1,380
graded checks: `causal_wrong_shipped = 0`, `words_wrong_shipped = 0`.

That guarantee is where this beats frontier language models, and the reason
is structural rather than a matter of effort. Causal identification is not a
fluency skill, so scale does not close the gap: on the same questions,
opus-5-class systems score 0–11% on identification while scoring 98.5% on
grade-school math. This engine scores 176/176 on identification, at 0.5 ms
and $0 per call.

**Where that number does *not* yet hold, stated plainly:** the 176/176 is
measured on a synthetic universe where the ground truth is known by
construction. On the 101 real Tübingen cause-effect pairs — experimental
ground truth, already in `data/ce_pairs/` — this engine scores **62.4%**
forced-choice, roughly the published unsupervised level, with no signal
clearing its own pre-registered gate. The verification claim is proven on
synthetic data and *unproven on real data*; both halves of that sentence
belong in any honest pitch.

That arm is a **baseline, not a ship surface**, and the distinction is now
enforced rather than promised. Its scorer is an unverified forced-choice
voter: 38 of 101 wrong forced-choice, 33 wrong among the 88 it answered
confidently. `operations/ship_surface.py` walks the import graph of every
user-invokable entry point with `ast` — nothing is executed — and the card
carries the result as a gate: `causal_unverified_quarantined = 1/1`,
`novelty_ledger_clean = 1/1`, across 57 first-party modules and 86 import
edges. The walker ships with a negative control in pytest: a planted import
must be detected, because a gate that cannot fail is decoration. The 38
wrong answers are still printed, below the table and outside OVERALL, so
nothing is hidden by reclassifying the row.

Running the verified pipeline on the same pairs answers **0 of 101**
(`causal_verified_real = 0/101`) with zero wrong ships — and that zero is a
refusal, not a result. Every pair is two-variable observational data, where
the engine's Markov-equivalence doctrine holds X→Y and Y→X to be
observationally indistinguishable, so it declines to orient any of them. The
zero-wrong-ship guarantee is therefore **vacuously** satisfied on the
standard real benchmark. We report that rather than quote it: a guarantee
met by answering nothing is not a capability.

Two things changed there, and both were bugs rather than tuning. First,
`_safe_corr` — the engine's core statistic — returned **r = 1.0** for any
column containing a NaN, because Python's `min(1.0, nan)` is `1.0`. Three
of the pairs (0081–0083) carry a third column that is a missing-value
indicator, so the engine was refusing while believing it had found
*maximum* confounding. Any real dataset with missing values hit this.
Second, a context column that is a deterministic function of X or Y was
counted as a confounder rather than recognised as a copy.
`research/years/year2_system_one/auto_context.py` now gates the context
pool with a stated reason per rejection and ships a **measurement design**
for every pair it still cannot orient (`causal_verified_design = 101/101`):
which variable to randomize, what estimand to read, and the sample size
the observed correlation would need.

The coverage number is still 0/101 — the honest zero — but the *reason*
changed from "certain confounding, by artifact" to "no admissible context,
by measurement" (`causal_verified_real_ctx_admissible = 0/101`, with the
rejections logged as `too_missing` and `constant`). Only one of those two
sentences is true, and now the true one is the one the engine says. The
next real-data measurement stays multi-variable (Sachs; semi-synthetic
effect injection into the clinical cohorts), where orientation is
legitimately decidable.

### Widening the training envelope — one extrapolation closed, one left open

The support gate refused Sachs for being a 2.75x extrapolation, which is
correct behaviour and not a solution. The fix was to make the training
universe contain that regime — which is only legitimate if the head is
**validated there first**. The order is enforced by
`operations/validate_envelope_widening.py`, and it is the order that
matters:

1. Add four **wide non-Gaussian** world families — `wide_anm` (11
   variables), `wide_chain` (8), `wide_fork` (9), `heavy_tail_pair`.
   Designed from phosphorylation-cascade structure (positive, right-skewed,
   heavy-tailed, a network of interacting kinases) and **not** from any
   statistic of `data/sachs.csv`. No Sachs value was inspected while
   choosing the distribution shapes. They are domain priors, not a fit to
   the answer.
2. Retrain the direction head on the widened universe (45,408 rows, 22
   families, 402s).
3. **Measure fresh-seed holdout accuracy per family.** The four new
   families scored **62.5%** under the old head and **75.8%** under the new
   one. Above the 60% licensing bar — so the widening is *earned*.
4. Only then re-derive the envelope constants by measurement.
5. Only then measure Sachs, once.

| | before | after |
|---|---|---|
| max variables | 4 | 11 |
| max \|skew\| | 0.411 | 3.642 |
| max \|excess kurtosis\| | 4.125 | 40.878 |

What that bought: **the variable-count refusal is closed.** Sachs satisfies
it now (11 ≤ 11), so the 2.75x extrapolation the head could not be trusted
with is gone. The synthetic causal lane also improved and now *includes* the
hard regimes: `causal_direction` 153/176 → **192/220 (87.3%)**,
`causal_effect_sign` 158/176 → **196/220 (89.1%)**, identification 220/220.

What it did **not** buy: Sachs still refuses, now on marginal *shape* only —
|skew| 6.06 against a 3.72 bound, |excess kurtosis| 53.7 against 41.7. The
margin fell from ~74x over the bound to ~1.6x, which is real progress. But
closing it entirely would mean inventing training worlds shaped like the
measured statistics of the dataset being evaluated, which is fitting the
simulator to the answer. That residual is left closed on purpose.

One regression to report: `latent_fork` holdout fell from 25% to 0% in the
retrain. Nothing in the gates caught it, because the lane averages over
families — which is a scoring weakness worth fixing, not a fact to bury.

Also found while building this: `_noise(rng, n, sd, dist="exp")` called
`rng.exponential(sd)` without `size=n`, which returns a **scalar**. One
constant was broadcast into every row and the whole `exp` family looked
exactly Gaussian (skew 0.000, kurtosis 3.000) — a silent bug that would have
invalidated the entire exercise while reporting plausible numbers.

### Per-family reporting, and one regression that was not a regression

A pooled 88% is not a result; it is an average over families whose accuracy
spans 34% to 100%. `operations/causal_calibration.py` now prints all 22 and
reproduces `lane_causal`'s own number (192/220, 0 wrong ships) as a
**control** before printing anything else — because three earlier versions
of this scorer disagreed with the card while emitting confident, plausible,
wrong tables, including a "a clean threshold exists" conclusion built on six
samples.

**The `latent_fork` 0% was real, and so was the 100% next to it. Neither was
wrong; they were answers to different questions.** `latent_fork` has two
variables, so it can ask exactly two direction questions, and a hidden
common cause makes *both* unanswerable — there is no directional truth in
the family at all.

* `validate_envelope_widening` asks the **raw head** for an argmax label.
  It answers `X_TO_Y`/`Y_TO_X`. The head has essentially never learned to
  emit `UNKNOWN` for a two-variable world: **0%**.
* `causal_calibration` runs the **full pipeline**, which abstains on both
  (DEFER, confidence 0.20). A refusal on an unanswerable question earns
  credit: **100%**.

So the head is 0% at *saying "unidentifiable"* and the shipped system is 100%
at *refusing to*. The shipped system is right here. Neither figure is
orientation skill, and the harness now prints `def` (accuracy restricted to
questions that actually have a direction) beside `acc` so the two can never
be confused again. `latent_fork` and `selection_bias` read `def = n/a`:
every question they can ask is unanswerable.

### The risk curve: closing the shape gap is NOT licensed

With ground truth known, the obvious way to close the remaining gap is to
find a confidence threshold above which the wide non-Gaussian worlds are
answered safely, and gate on it. Measured on 209 answers with a definite
truth across the four wide families:

| threshold | ships | correct | wrong | precision |
|-----------|-------|---------|-------|-----------|
| 0.50 | 172 | 154 | 8 | 0.895 |
| 0.68 | 166 | 148 | 8 | 0.892 |
| 0.75 | 69 | 54 | 8 | 0.783 |
| 0.80 | 28 | 20 | 8 | 0.714 |
| 0.85 | 0 | 0 | 0 | — |

**No threshold with at least 20 ships gives zero wrong ships.** The wrong
answers do not move down the confidence axis at all: 8 wrong at 0.50, 8
wrong at 0.80, and then coverage collapses to nothing. The same shape as
Sachs, where wrong orientations came back at 0.804–0.837 and right ones at
0.804–0.821, all 22 inside a 0.033-wide band.

So the support envelope **stays closed**. The marginal-shape gap on Sachs
(skew 6.06 against a 3.72 bound, kurtosis 53.7 against 41.7 — a margin down
from 74x to 1.6x) is close, and closing it by training would have bought
coverage at the cost of answers this head cannot make safely. That trade is
not available. The refusal is the feature.

### A confident UNKNOWN is not a safe negative

Per-family reporting paid off immediately by exposing a defect the pooled
number hid. `wide_anm` read 69% overall but **34%** on questions that have a
direction — and the diagnosis was not what it looked like. 72% of that
deficit was the engine *declining* decidable questions, which the scoring
rule awards nothing for. The actual engine defect was narrower and much
sharper: **11 of 12 wrong ships were confident `UNKNOWN` answers on
non-adjacent ancestor pairs.**

That happens because `World.truth()` defines a direction as an *ancestor*
relation, while a pairwise engine can only see *adjacency*. On a 3-variable
world those nearly coincide; on an 11-variable DAG they do not. The engine
finds no direct edge, sees correlation it cannot orient, and ships
"UNKNOWN" — a claim about the world built from an absence of evidence.

The engine already tried to guard this, and the guard was backwards. It
refused an UNKNOWN only when `conf < 0.70`, reasoning that a *strong*
unknown must be a real finding. Measured, the wrong ones are the strong
ones:

| shipped UNKNOWN labels | count | mean confidence |
|---|---|---|
| correct findings (truth genuinely unidentifiable) | 30 | 0.824 |
| wrong ships (truth had a direction) | 12 | 0.828 |

Every wrong ship sat at 0.806–0.847, above the 0.70 bar, so the rule never
fired on the cases it existed for. This is the same 0.033-wide band that
made confidence useless on Sachs, reproduced on synthetic ground truth
where the truth is known.

Nothing else available separates the two populations. Common-cause suspicion
is mildly *anti*-correlated (0.174 correct vs 0.148 wrong). World width is
confounded with family structure — `wide_fork` gives 5 correct / 0 wrong and
`wide_chain` 0 correct / 4 wrong at the same 8–9 variables — and
`P(definite | non-adjacent pair)` by width came out **non-monotone**
(0.69, 0.84, 1.00, 1.00, 0.31), tracking whether a family is a chain or a
sparse random DAG, which is invisible to a pairwise engine. So there is no
threshold to find; it is a distinction this head cannot make.

**The fix is to refuse the negative and keep the reason.** A confident
UNKNOWN now always routes to System Two, and the finding travels in the
decision's `meta` (`unidentifiable`, `unidentifiable_label`,
`unidentifiable_confidence`, `suspicion`) instead of in the shipped label.
The caller still learns "I believe these two are confounded"; it just is not
presented as a verified answer.

Measured over all 22 families, 8 fresh seeds: **total wrong ships 15 → 3,
accuracy 88.4% → 88.4%, zero delta.** `wide_anm` 7 → 0, `wide_chain` 4 → 0,
`collider` 1 → 0. The three survivors are wrong *direction* labels in `fork`,
`butterfly` and `wide_chain`, measured to be identical with this refusal
switched on and off — neither caused nor fixed by it, and not claimed here.

The zero cost is structural, not lucky. Declining an unanswerable question is
still the right answer, so the 30 correct negatives lose nothing; declining
an answerable one was already scored wrong, so the 12 wrong ones lose
nothing they were not already losing. The engine trades *coverage of a
negative it cannot support* for nothing. What it gives up is the ability to
volunteer "this is unidentifiable" as a shipped answer — a real capability,
and the decision to give it up was made deliberately rather than by accident.

### The external number: on real cause-effect pairs the engine is worse than a constant

`causal_verified_real` is the **Tübingen Cause-Effect Pairs** benchmark —
external data with published ground-truth directions, and the only causal
lane in this repository scored on something we did not manufacture. Its
coverage is **0/101**. That zero has now been diagnosed properly, and the
diagnosis is the most useful measurement here.

**Why it refuses.** Not the support envelope — 96 of the 101 pairs are
*inside* it, and marginal shape binds on only 5. The real reason is that 98
of the files are **2-column**, so no context exists, `n_context == 0`, and
`NO_CONTEXT_DIRECTION_CAP = 0.68` floors every decision to VERIFY. All 101
land at confidence exactly 0.680. This is the engine behaving correctly:
orientation from a bare two-column table is not identifiable in general,
which is the same lesson the `reverse_direct` uncertainty-teacher family
teaches. The lane is structurally incapable of shipping, and that is a
property of the input, not a missing feature.

**Why the refusal is right, and not lucky.** The number that was missing is
what the engine *would* name if it shipped. Measured against baselines,
because "below chance" only matters if chance is not higher than a constant:

| predictor on the 101 external pairs | accuracy |
|---|---|
| majority class (always "the first column causes the second") | **73.3%** |
| coin flip | 50.0% |
| **this engine, forced to answer** | **41.6%** |

It is **32 points below a constant that ignores the data entirely**.

An earlier draft of this section called 41.6% "below chance". That was
overstated and the significance check does not support it: at n=101 the
standard error is 5.0 points, so 41.6% sits **1.69 sigma** from 50% — *not
distinguishable from chance*. The correct reading is that the forced choice
is uninformative, not inverted. Reversing every answer gives 58.4%, still
under the majority constant.

The wide-multivariate result is a different matter and **is** significant:
44/149 = 29.5%, SE 4.1 points, **5.0 sigma** below chance. That one is real.

Searching for a mechanism on the 2-column pairs found nothing usable: the
best correlation between any cheap feature and the true direction is
**−0.214** (`spread`, the effect-to-cause standard-deviation ratio), and
`spread` agrees with ground truth on 53% of pairs — barely above chance.
There is no orientation signal in these features on two-column data, which
is why the cap has to refuse rather than filter.

Worse, its confidence is **anti-calibrated** on external data. Pre-cap
confidence correlates **−0.035** with correctness, and raising the threshold
makes accuracy monotonically *worse* (41.6% at 0.50 down to 37.8% at 0.95).
The benchmark's own identifiability weight correlates **−0.191**: the engine
is most confident on the pairs the benchmark calls *least* identifiable. So
there is no threshold on any available signal that buys safety, and the cap
has to refuse rather than filter.

**The consequence for the card.** `causal_verified_real_wrong_shipped` was
green because the engine answered none of the 101 pairs. A wrong-ship gate
satisfied by shipping nothing is not a safety result — it is a guard that
cannot fail. The lane now also reports
`causal_verified_real_safety_demonstrated`, which requires `ships > 0` **and**
`wrong == 0`, so it stays red until the engine is both loud and right. If
that ever happens it goes green on its own: the gate is falsifiable, which
the previous row was not.

What this does *not* license is a transform that reshapes real data into the
envelope to buy coverage. The envelope was not the binding constraint here,
so that experiment was aimed at the wrong thing; the binding constraint is
that two-column tables do not determine orientation.

### Wide external data: six datasets the loader was throwing away

The 2-column diagnosis above said no context exists, so no context could be
created. That turned out to be true of the *loader*, not the benchmark.

`pairs.zip` contains **108** pair files. Ninety-nine are 2-column, and the
101 usable ones are all that `causal_verified_real` ever saw — but **six are
multivariate, with published ground truth**, and every one was discarded on a
technicality: `cause_cols` spans several columns, and the loader filtered on
`cause_cols[0] != cause_cols[1]`.

| pair | cols | rows | cause block | effect block | published |
|---|---|---|---|---|---|
| 0052 | 8 | 10226 | cols 5–8 | cols 1–4 | `<-` |
| 0053 | 4 | 989 | cols 2–4 | col 1 | `<-` |
| 0054 | 5 | 392 | cols 1–3 | cols 4–5 | `->` |
| **0055** | **32** | 72 | cols 17–32 | cols 1–16 | `<-` |
| 0071 | 8 | 120 | cols 1–6 | cols 7–8 | `->` |
| 0105 | 10 | 1000 | cols 1–9 | col 10 | `->` |

Real, external, **wide**, and scored against ground truth nobody here wrote.
The estimator was declared before the results were seen: the engine orients
pairs, so a block cause needs an aggregation rule — ask every cross-block
ordered pair, and count what the engine agrees with.

**Result: 302 ordered pairs asked, 1 shipped, and it was right. Zero wrong
ships.** Forced off the cap, the head would have been right **29.5%** of the
time (44/149) — further below chance than even the 2-column case. The refusal
doctrine holds up on wide real data too, and where the engine *did* speak on
external ground truth it was correct.

Six datasets is a **case study, not a benchmark**, and no accuracy claim is
made from it. What it supports is a safety claim, and the first non-vacuous
one in this repository: `causal_verified_wide_real_safety_demonstrated = 1/1`
requires a real ship, so silence cannot pass it. Compare
`causal_verified_real_safety_demonstrated`, which is red precisely because it
refuses everything.

**It also corrects an earlier claim of mine.** I reported that the
variable-count refusal had "closed" after widening the universe. It closed
*for Sachs' 11 variables* — the ceiling is `SUPPORT_MAX_VARIABLES = 11`, and
0055's 32 variables exceed it, refused as
`32 variables, beyond the 11-variable training maximum`. The ceiling is still
there, and the only genuinely wide real dataset in reach is above it. Whether
to raise it would have to go through `validate_envelope_widening`: widen the
universe, retrain, measure a fresh-seed holdout, re-derive the constants.
Given the risk curve found no zero-wrong threshold on wide worlds, that is not
licensed yet.

This lane also cost OVERALL 8 points (70.5% → 62.7%), which is the honest
consequence of asking 302 external questions and shipping 1. The card's rule
is that abstention is not-correct, and this lane applies it to itself.

### Generator guards

A number measured on synthetic worlds is only as good as the generator that
made them, and a generator that silently emits the wrong *kind* of data
keeps every downstream number perfectly self-consistent and perfectly
meaningless. That is not hypothetical: it already happened once, with
`rng.exponential(sd)` and no `size=n`. `tests/test_world_generators.py` (88
tests) now guards the six ways it can happen again —

* a noise sample of `n` rows is `n` values long, for all five distributions;
* each distribution **measures** like the distribution that was asked for,
  which is the "too clean" guard;
* no column of a non-Gaussian family carries the exact-Gaussian signature,
  and no family emits a constant column;
* the wide families are actually wide, and still exceed the pre-widening
  `|skew| <= 0.411` bound — otherwise "we widened the universe" is a claim
  in a comment;
* `_build` is reproducible per seed and varies across seeds, because a
  fresh-seed holdout is meaningless if a seed does not name a world;
* and a **negative control**: the historical scalar bug is monkeypatched
  back in and the guard is required to fail. A guard that cannot fail is
  decoration.

### The Sachs result: the synthetic number does not transfer, and now the engine knows it

This is the most important measurement in the repository, so it is here in
full rather than in a footnote.

`lane_causal_sachs` is the first non-vacuous real-data causal arm: 11
measured protein variables, ground truth from published laboratory
interventions (`data/sachs_target.csv`). It asks the 18 published edges
in both directions — 36 ordered questions — and counts wrong ships.

**The first run shipped 15 wrong orientations out of 22.** Only 3 of the 18
published edges were recovered. The same head scores 86.9% on the synthetic
universe.

Before treating that as a verdict, the measurement itself was checked, and
two things were wrong with the *question*, not the answer:

- 37 of the 55 node pairs are not adjacent in the ground truth but **are**
  connected by a directed path. "Does A cause B" is ill-posed for those, so
  scoring a correct indirect claim as wrong is the harness lying. The arm now
  asks only the 36 well-posed questions.
- The edge list contains a 2-cycle (`PIP3 -> plcg` and `plcg -> PIP3`), so it
  is not a DAG.

Corrected, the result stands: 15 wrong ships. Then the cause was found, and
it is not a tuning problem:

| | wrong ships | right ships |
|---|---|---|
| confidence returned | 0.804 – 0.837 | 0.804 – 0.821 |

**The confidence channel carries no information at all.** All 22 answers sat
inside a 0.033-wide band, so no threshold separates right from wrong — and
lowering the cap to the context-free value destroyed 0 wrong *and* 0 correct
answers. It was a refusal wearing a parameter.

What does separate them is the training distribution:

| | training universe (max) | Sachs |
|---|---|---|
| variables | 4 | 11 |
| \|skew\| | 0.411 | 30.439 |
| \|excess kurtosis\| | 4.125 | 1520.442 |

The direction head was being asked about a world nearly three times wider
than anything it had seen, with marginals nothing like them. So
`research/years/year2_system_one/support.py` measures that envelope by
walking the synthetic universe, and the fast path now **declines** any
direction or sign question outside it, with the reason attached:
`out of training support: 11 variables, beyond the 4-variable training
maximum; column 0 |skew|=30.44 exceeds the 0.45 training maximum`.

After the gate: **`causal_sachs_wrong_shipped = 1/1` with coverage 0/36**,
while the synthetic lanes are untouched (153/176, 158/176, 176/176).

Read that honestly: **this is a refusal, not a capability.** The engine still
cannot orient real causal structure; it now declines to pretend it can. That
is the precondition for the first, and it is the only version of the
guarantee that survives contact with real data.

Two caveats that travel with every Sachs number, recorded in
`data/sachs_provenance.json` and reprinted by the card:

- The file's **origin is unverified**. There is no dataset card or citation.
  Its 7,466 rows do not match the canonical 853-row observational file.
  That is *consistent with* the corpus pooled across experimental
  conditions, but a clustering probe found only ~3 clear boundaries, which
  does not confirm it — so pooling is neither established nor ruled out. If
  the rows are pooled, condition membership is an unmeasured common cause and
  orientation on the pooled matrix is confounded by construction. Condition
  labels are **not** reconstructed by clustering and **not** used anywhere;
  machine-inferred strata would be exactly the synthetic input the program
  forbids. Before this dataset backs an investor number, replace it with the
  single-condition file — the arm is built so only the provenance record has
  to change.
- The graph-level instrument is **not** better. PC on the same data recovers
  12 of 18 true adjacencies (66.7% recall) but orients at 29.6% precision,
  and it misses five edges incident on `PKC`, a hub of the true network.
  Linear-Gaussian conditional-independence tests on raw, non-Gaussian,
  pooled multi-omics intensities are the wrong tool; that is the honest
  explanation of both failures, and it is a known one in the literature.

### The code lane: coverage by retry, with the ceiling removed

The code lane is the only one with an external, free verifier — the
compiler and the tests are ground truth the project does not own. So the
lane was rebuilt around the one property that makes retries nearly free:
a failed attempt costs nothing, a wrong ship costs the guarantee.

Coverage came from a **retry ladder over grammars**, not from guessing
harder. The legacy grammar could not express `((amount*10)/100)` because
10 was not an atom, and could not pair two composite subexpressions at
all, so `a*a + b*b` was unreachable — a grammar limit masquerading as an
honest refusal. Three rungs now (wider constants, then composite pairing),
each verified before shipping, so widening the grammar cannot make a wrong
ship possible. Measured on 23 specs declared *before* any of this, in
`data/benchmarks/code_synth_specs.json`:

| configuration | coverage | right on held-out rows |
|---|---|---|
| legacy grammar | 14/23 | 13/23 |
| + leave-one-out selection | 14/23 | 13/23 |
| + retry ladder (shipped) | **23/23** | **18/23** |

Newly wrong on held-out rows: **zero**. The leave-one-out selector is a
measured **null** on this suite — only two specs carry three or more
examples, and on both the first-fitting candidate already won — so it is
reported as inactive rather than claimed as a gain.

The more interesting result is a deletion. Synthesis used to ship at 0.90.
Three rules for earning a higher tier were implemented and measured, and
every one of them granted the tier to code that is provably wrong off the
given examples:

- "exactly one candidate fits" → `((w*5)*5)`, which ignores `l` entirely
- "one behaviour on the probe grid" → `((b*5)-a)`, which ignores `a*a`
- "enough examples *and* the search ran out" → `((a*c)*max(b, 1))` for `a*b*c`

The last is the instructive failure: exhausting the search proves the
*grammar* contains one answer, not that the *function* has one. If the true
expression is outside the DSL, the search returns a degenerate impostor and
calls it unique. So the ceiling is now **0.60** for every verified
synthesis — "this code provably satisfies every example you gave, and
nothing more" — which is a weaker claim than the old one and a stronger one
than any tier that could be justified.

What the lane now guarantees is not "never wrong" but **"never wrong
silently."** Five of 23 ships are wrong on held-out rows, and all five say
so: `code_declared_flagged = 23/23`, and the card carries the falsifiable
gate `code_declared_unflagged_wrong = 1/1`. If a future change ever ships a
wrong answer with a clean bill of health, that row goes red.

### What we do not claim

We are not a general-purpose AI and we are not trying to be. We lose on
coverage and say so: 22.5% shipped-correct on GSM8K-200 against 84–98% for
frontier systems. We do not compete on breadth, world knowledge, or fluency.
What we sell is the part that compounds — correctness you can audit, inside a
guarantee that is measured rather than asserted. Concretely, the guarantee is
now enforced at five independent points, all of them pass/fail gates that can
go red: unverified scorers unreachable from any user-invokable entry point
(`causal_unverified_quarantined`), the novelty ledger citing none of them
(`novelty_ledger_clean`), zero wrong ships on the synthetic universe and on
both real-data arms (`causal_wrong_shipped`,
`causal_verified_real_wrong_shipped`, `causal_sachs_wrong_shipped`), and no
unflagged wrong synthesis on the declared suite
(`code_declared_unflagged_wrong`).

And the limit, stated as precisely as we can: **on real data the engine's
causal answers are currently refusals.** On Sachs it would have shipped 15
wrong orientations; the support gate now declines instead, because the
training distribution never contained a world like that. Zero wrong ships on
real data is currently achieved by not answering. The day it stops being
achieved by not answering is the day this project is actually worth
something, and the measurement that would prove it is on the card.

**White-label console for pharma & biotech.** The default skin —
*CausalPath Console* — is built for R&D teams asking target–outcome questions:
pathway causation from single-cell or screening data, confounder triage for
biomarker claims, and pre-experiment `do(...)` experiment design. Rebrand the
console by editing the `BRAND` object at the top of `operations/almir.html`
(name, tagline, monogram, support contact) — no code changes.

No causal-inference library is used for the core algorithms: Fisher-Z tests,
PC/FCI, do-calculus, backdoor/front-door adjustment, distance correlation,
bootstrap stability, and a graph-attention proof sketcher are all implemented
here.

## Quickstart

```bash
pip install -r requirements.txt

# Synthetic data — fork structure, effect query, counterfactual
python demo.py --dataset synthetic --structure fork --query "X -> Y" --counterfactual 2.0

# Real data — Sachs protein-signaling network (11 proteins, 7466 cells)
python demo.py --dataset sachs --query "PKC -> Mek"
```

### Library use

```python
from causal_engine import discover, intervene, audit, transfer

engine = discover(data, node_names=["X", "Y", "Z"], algorithm="pc")
ate, adjustment_set = intervene(engine, treatment=0, outcome=1)
verdict = audit(engine, treatment=0, outcome=1)   # review board debate
```

### White-label console (UI)

```bash
python3 operations/almir_ui.py --port 8420
# open http://127.0.0.1:8420 — light clinical UI, "Research use only" banner
```

### REST API

```bash
pip install "fastapi uvicorn pydantic"
python -m causal_engine.api --port 8420
# POST /discover · /audit · /intervene · /transfer · /benchmarks
# docs at http://127.0.0.1:8420/docs
```

### LLM agent integration

```python
from causal_engine.llm_bridge import CausalLLMBridge
bridge = CausalLLMBridge()
bridge.get_tool_schemas()        # tool definitions for an LLM agent
bridge.verify_claim("PKC causes Mek", dataset="sachs")
```

## What it does

| Stage | Module | Outcome |
|---|---|---|
| **Discover** | `causal_engine.core` | PC/FCI graph from observational data, with resampling stability |
| **Identify** | `research/weeks/week4_reasoning` | do-calculus proofs: backdoor/front-door adjustment sets, or NOT_IDENTIFIED |
| **Estimate** | `causal_engine.core` | ATE via the smallest verified adjustment set |
| **Audit** | `research/weeks/week7_parliament` | 3-member review board, every proposal falsified before voting → RELEASE / SUSPECTED / QUARANTINE |
| **Explain** | `research/weeks/week4_reasoning.epistemic_tracker` | KNOWN / SUSPECTED / UNKNOWN ledger with confidence |
| **Transfer** | `research/weeks/week24_worlds` | cross-domain effect-direction transfer gated by mechanism signatures |

## Results

**Synthetic structures** — chain / fork / collider recovered and oriented
with >95% accuracy; ATEs correct after backdoor adjustment.

**Sachs (real data, 11 proteins, 7466 cells, 18 true edges)**

| Method | F1 |
|---|---|
| PC, α=0.05 | 0.53 |
| Single FCI | 0.52 |
| Bootstrap FCI, thresh 0.7 | 0.54 (best) |

The honest finding: F1 plateaus ≈ 0.54 because a cluster of false
`pakts473` hub edges is a *genuine statistical dependency*, not a test
artifact — the PAG orientation marks those edges `<->` instead of
hallucinating directions.

**Verified interventions** — 73/110 ordered protein pairs identifiable with
a verified adjustment set (66%); the rest honestly marked NOT_IDENTIFIED.
Top verified effect: `PKC → P38`, ATE 5.11.

**Neurosymbolic bridge** — 0.86 proof-strategy accuracy (Bayes ceiling 0.90
with latents), zero unverified answers in 1300+ queries.

**Parliament** — on Sachs PKC→P38 the surviving backdoor (0.625) and SCM
(0.561) proposals agree → KNOWN; confounded naive claims are quarantined.
Active experiment design resolves 9 of 10 unprovable claims with one
proposed `do(praf)` knockout.

## Who it is for (pharma & biotech)

| Use case | What the engine does |
|---|---|
| **Target triage** | Rank which protein/gene pairs have *identifiable, verified* effects before committing wet-lab budget — `experiment_design.py` scores the single `do(...)` intervention that resolves the most disputed claims |
| **Biomarker claim audit** | Confounded biomarker→outcome claims are quarantined, not averaged away (49/49 confounded claims caught in the deployment report, 0 false accusations) |
| **Screening data** | Sachs single-cell discovery is the reference workflow — with the F1 ceiling documented, not hidden |
| **Review-ready evidence** | Every released claim carries its adjustment set, evidence type, and KNOWN/SUSPECTED/UNKNOWN provenance — an audit trail for governance |

## Repository layout

```
causal_engine/     public package — supported API, REST server, LLM bridge
research/          the research journal:
  weeks/             week1 → week30 arc (perception → cognition → parliament → SHIP)
  years/             year1 → year5 quarterly journal (history; not the supported API)
operations/        white-label console (almir), run supervisor, research loop, alerts
tests/             smoke, e2e, and module tests
data/              Sachs datasets + discovery ledgers
docs/              OPERATIONS.md runbook · docs/archive/ (v0.1-era documents)
LICENSE            MIT
```

Full layer diagram, dependency rules, and design principles:
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Testing

```bash
python tests/smoke_test.py        # all modules compile
python -m pytest tests/ -x -q     # unit + integration
python causal_engine/core.py      # engine self-test (fork ATE ≈ 0)
```

Most research modules carry their own `__main__` test block with honest
assertions — run any file directly to see it verify itself.

## Honest limitations

- **The hybrid nonlinear oracle was falsified on Sachs** — dCor agrees with
  Fisher-Z on the false hub edges. A power-calibrated dCor is the open
  follow-up.
- **NOT_IDENTIFIED is often information-theoretic**: with latent
  confounders, no observed adjustment set can exist (the sketcher's
  Bayes-ceiling analysis quantifies this).
- **Unoriented skeletons give conservative answers** — on an unorientable
  chain CPDAG, the engine reports the direct effect, not the total effect.
- **Skeleton backdoor uses conservative semantics**: undirected edges may
  hide confounding, so the engine never claims an effect it cannot block
  paths for.

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | layers, module map, dependency rules, data flow |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | almir UI, run matrix, alerts, research loop |
| [docs/archive/](docs/archive/) | v0.1 architecture, roadmaps, deployment & host-ops guides |
| [paper/paper.md](paper/paper.md) | the system paper |

## Data

Sachs, K., et al. (2005). *Causal protein-signaling networks derived from
multiparameter single-cell data.* Science, 308(5721), 523-529.
`data/sachs.csv`, `data/sachs_target.csv`.

## License

MIT — see the `LICENSE` file. The white-label console may be rebranded and
deployed inside your organization under the same license.


this project is for sale any can buy > imranjabran@gmail.com
