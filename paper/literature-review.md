# Literature review: Hanner-Hydra

Search date 2026-11-08. This review informs the related-work section and positioning of the
Hanner-Hydra paper (`paper/main.tex`). Citation keys refer to `paper/references.bib`.

**How the search was done.** About 35 web searches (log at the end), plus close reading of
the two sources the paper depends on most: the NanoHydra paper (arXiv HTML, read in full)
and the OpenAI Family 087 Mahler manuscript (PDF, first pages read; repository index and
community notes checked). Every entry has one of these marks:

- **[read]**: the primary source was read for the details given here.
- **[checked]**: bibliographic details and main claim confirmed from the publisher or
  arXiv listing, not read in full.
- **[secondary]**: known only from abstracts or secondary summaries; verify before relying
  on details.

Numbers quoted from other papers are their own reported numbers under their own protocols.

---

## 1. Summary and positioning

1. **No prior work uses structured or polytope-based kernel groups in time series
   classification.** We found no paper that picks Hydra's competing groups for their
   geometry, or that finds a group's winner faster than evaluating every kernel. The
   closest work, NanoHydra, uses binary kernels in unstructured random groups and still
   evaluates all of them (§3). This is the paper's main novelty claim. It rests on a web
   search, not an exhaustive review, so phrase it as "to our knowledge".
2. **The nearest conceptual relatives are in hashing, not in time series classification.**
   A cube group's winner is the sign pattern of the patch, which is hyperplane hashing
   (SimHash). A cross-polytope group's winner is the largest coordinate with its sign,
   which is cross-polytope LSH. Hanner cotrees interpolate between the two (§5). This
   framing is new as far as we found, and it explains *what* the features compute.
3. **The accuracy result (equivalent to Hydra, not better) fits the field.** Binary or
   ternary weights cost little or nothing in random-kernel methods: MiniRocket uses two
   weight values, and NanoHydra's binary kernels cost about 1.5 points on UCR in its own
   table. Our pre-registered result, −0.06 points with matched controls, is a
   stronger and cleaner statement than either.
4. **The state of the art has moved on in accuracy, not in cost.** MultiRocket+Hydra, HC2,
   QUANT and 2026 in-context methods (RocketPFN, MASHT) lead on accuracy (§2, §8). Hanner-
   Hydra's case is cost: single-core speed, memory, multiplication-free inference. The
   paper should present it as an efficiency result, and position edge deployment
   (NanoHydra's setting) as the natural application.
5. **The Mahler motivation needs careful wording.** The symmetric Mahler conjecture with
   Hanner equality cases is claimed in an OpenAI manuscript (2026). It is not peer
   reviewed, and its Lean index has review status "unchecked" (§6). Hanner polytopes were
   already known as conjectured minimizers, proven in special classes and proven local
   minimizers. The method needs none of this; it uses the combinatorial structure only.

---

## 2. Random convolutional kernel methods

| Work | Key | Status | What matters for us |
| :--- | :--- | :--- | :--- |
| ROCKET (Dempster, Petitjean, Webb 2020), DMKD 34(5) 1454–1495 | `dempster2020rocket` | [checked] | 10,000 random kernels, PPV and max pooling, linear classifier. The family's origin. |
| MiniRocket (Dempster, Schmidt, Webb, KDD 2021) | `dempster2021minirocket` | [checked] | Fixed kernels of length 9 with two weight values (−1, 2); up to 75× faster than ROCKET. Precedent for few-valued, multiplication-free kernels. |
| MultiRocket (Tan et al. 2022), DMKD 36(5) 1623–1646 | `tan2022multirocket` | [checked] | Adds pooling operators and first differences; more accurate than MiniRocket. Published comparator in our secondary analysis (Hanner −0.75 points, practically worse). |
| Hydra (Dempster, Schmidt, Webb 2023), DMKD 37(5) 1779–1805 | `dempster2023hydra` | [checked] | Our baseline. Groups of k competing Gaussian kernels; counts winners; one parameter moves it between dictionary-like and ROCKET-like. Hydra + MultiRocket concatenation is the strongest pipeline in that paper. |
| QUANT (Dempster, Schmidt, Webb 2024), DMKD 38(4) 2377–2402 | `dempster2024quant` | [checked] | Fixed intervals, quantiles, extremely randomised trees; state-of-the-art accuracy on 142 UCR datasets in under 15 min on one core. Another efficiency-first method; possible combination partner. |
| S-Rocket (Salehinejad et al. 2022), arXiv 2203.03445 | `salehinejad2022srocket` | [secondary] | Prunes up to 99% of kernels in some datasets without noticeable loss. |
| Detach-ROCKET (Uribarri et al. 2024), DMKD | `uribarri2024detach` | [checked] | Sequential feature detachment keeps about 10% of features at equal or better accuracy. Shows the features are highly redundant; suggests pruning as future work for Hanner features too. |

**Takeaways.** Speed in this family has come from fewer or cheaper kernels (MiniRocket),
from pruning (S-Rocket, Detach-ROCKET), and from simpler features (QUANT). Hanner-Hydra adds
a fourth route: *structure inside a group*, so that the cost of finding the winner no longer
grows with the group size.

---

## 3. Closest work: NanoHydra and edge deployment

### NanoHydra [read]

Cioflan, Fonseca, Wang, Benini, "NanoHydra: Energy-Efficient Time-Series Classification at
the Edge", arXiv 2510.20038 (v1, 22 Oct 2025), accepted at IJCNN 2025 (arXiv comment). ETH
Zurich. Code: github.com/pulp-platform/nanohydra. Key `cioflan2025nanohydra`.

- **Kernels:** binary weights from {−1, 1}, length 9, binary in training and on-device.
  Groups of K kernels; the paper does not state H (groups) or K (kernels per group), nor the
  dilation values.
- **Features:** counts both the max-response and the min-response kernel per group, hard
  and soft counting; first differences; two configurations (A-NanoHydra: 5 dilations, 2
  difference orders; E-NanoHydra: 3 and 1).
- **Arithmetic:** int16 convolution operands, int8 post-training-quantised logistic-
  regression classifier; Hydra's square root and division in the scaler are replaced by
  shifts. Dot products use SIMD; the paper describes **no** way to find the group winner
  without evaluating all K kernels.
- **Accuracy:** UCR average (Table I, int16 model): NanoHydra 83.81% vs HYDRA 85.31%,
  MiniRocket 86.09%, MultiRocket 86.84%. NanoHydra's numbers are the best of 20 runs per
  dataset (Table I caption) and per hyperparameter pair (Section IV); the related-work
  numbers, Hydra's included, are copied from the WEASEL 2.0 paper ("as reported by Schäfer
  et al."). The paper says only "over 100 datasets"; its code (`pretrain/data/datasets.csv`,
  `run_ucr_benchmark.py` in pulp-platform/nanohydra, checked 2026-11-08) lists exactly the
  112 UCR datasets we use (aeon's `univariate_equal_length`), loads the default train/test
  split, and runs k=8, g=16 with best-of-10 (best-of-25 in the quantised script; the paper
  says 20). Best-of-20 favours NanoHydra, so the 1.5-point gap to
  Hydra is if anything an underestimate. ECG5000:
  94.13% (E) and 94.47% (A).
- **Hardware:** GAP9 MCU. E-NanoHydra uses 3.82 kB of memory; 0.33 ms and 7.69 µJ per
  inference at 240 MHz (the abstract's headline). The "18× more efficient" claim matches
  NAS-TCN's 140 µJ, but the paper does not name the baseline.

**Relation to Hanner-Hydra.**
- Same goal (cheaper Hydra), different mechanism. NanoHydra makes each kernel cheaper;
  Hanner-Hydra makes the *group* cheaper. A Hanner cube group contains *all* 2^l sign
  vectors on its taps and finds the winner in O(l); NanoHydra's binary groups are K random
  sign vectors evaluated one by one.
- NanoHydra counts hard-min as well as hard-max. For centrally symmetric groups this is
  redundant: the hard-min counts are a permutation of the hard-max counts, and our
  ablation gives identical predictions. Binary {−1, 1} groups are not centrally symmetric
  unless drawn in pairs, so for NanoHydra the min counts can carry information.
- Accuracy: NanoHydra is 1.5 points below Hydra on UCR in its own Table I, despite
  best-of-20 numbers. Our
  pre-registered comparison shows −0.06 points (practically equivalent). The protocols
  differ, so do not compare the numbers directly; it is fair to say that ternary
  structured groups need not cost accuracy.
- Natural future work and a strong reviewer question: a Hanner kernel on GAP9 or another
  MCU, against NanoHydra on the same hardware.

### MiniRocket on microcontrollers [secondary]

An ETH Zurich report on IoT asset tracking ports MiniRocket to a low-power sensor node in
floating and fixed point, within 1% of floating-point accuracy (the extracted figures,
about 8.6 ms and 72 µJ, need checking in the PDF). NanoHydra's introduction cites ROCKET on
an nRF52810 MCU. Not yet in the `.bib`; add if the paper discusses deployment.

---

## 4. Dictionary, symbolic and pattern-based methods

| Work | Key | Status | Relevance |
| :--- | :--- | :--- | :--- |
| SAX (Lin et al. 2007), DMKD 15(2) 107–144 | `lin2007sax` | [checked] | Symbolic words; SAX-style word counts were a planned comparator (dropped, prereg §8). |
| WEASEL 2.0 (Schäfer, Leser 2023), Machine Learning 112(12) 4763–4788 | `schaefer2023weasel2` | [checked] | Dilated, randomised dictionary transform with fixed memory. Hanner is on par (inconclusive, +0.01 points). |
| 1D local binary patterns (e.g. Kaya et al. 2014, EEG) | — | [secondary] | Sign-of-neighbour patterns on 1D signals. A cube group's winner is a sign pattern of the centred patch, a close relative. |
| Ordinal patterns (Bandt, Pompe 2002; ordinal-pattern classification, Discovery Science 2018) | — | [secondary] | Rank patterns of short windows. Related but not identical: ordinal patterns are permutations, Hanner winners are sign vectors. |

**Takeaway.** The description "close to a bag of binary symbolic words" is
supported: Hanner features are counts of sign patterns on blocks of taps, weighted by the
soft-max response. Cite SAX and WEASEL 2.0 for the dictionary side; mention 1D-LBP only
briefly, or add a verified reference.

---

## 5. Hashing and max-affine views (the conceptual frame)

| Work | Key | Status | Relevance |
| :--- | :--- | :--- | :--- |
| Charikar 2002, STOC 380–388 | `charikar2002simhash` | [checked] | Random-hyperplane rounding as LSH (SimHash): the sign pattern of projections. A cube group's winner is exactly the sign pattern of its taps. |
| Andoni et al. 2015, NIPS 28, 1225–1233 ("Practical and optimal LSH for angular distance") | `andoni2015crosspolytope` | [checked] | Cross-polytope LSH: rotate, then map to the nearest signed basis vector, which is argmax ±e_i. A cross-polytope group's winner is exactly this. Asymptotically optimal and practical. |
| Yagnik et al. 2011, ICCV ("The power of comparative reasoning") | `yagnik2011wta` | [secondary] | Winner-take-all hashing: the index of the maximum among K permuted coordinates. Hydra-style counting is a convolutional WTA code. |
| Maxout polytopes (arXiv 2509.21286); tropical views of networks (e.g. Charisopoulos, Maragos, arXiv 1805.08749) | — | [secondary] | max_v ⟨x, v⟩ is the support function of conv(V), a max-affine (tropical) function; Hydra's group winner is a maxout unit. |

**Takeaway, for the method section.** Hanner groups interpolate between hyperplane LSH (cube)
and cross-polytope LSH (cross-polytope). Mixed cotrees give sign patterns on some blocks and
"largest block" choices on others. This gives readers an established vocabulary for the
features, and it links the cotree to known optimality results in hashing (cross-polytope
LSH is optimal for angular distance). Present it as interpretation, not as a theorem about
classification accuracy.

---

## 6. Convex geometry: Hanner polytopes and the Mahler conjecture

### Hanner polytopes and their combinatorics

| Work | Key | Status | Content |
| :--- | :--- | :--- | :--- |
| Hanner 1956, Math. Scand. 4, 65–87 | `hanner1956intersections` | [checked] | Origin of the recursive spaces (3.2 intersection property). |
| Hansen, Lima 1981, Acta Math. 146, 1–23 | `hansen1981structure` | [checked] | Classification of finite-dimensional spaces with the 3.2 intersection property; used as the final input of the OpenAI equality proof. |
| Corneil, Lerchs, Stewart Burlingham 1981, Discrete Appl. Math. 3(3) 163–174 | `corneil1981cographs` | [checked] | Cographs and cotrees. The cotree is our data structure; cite for the term. |
| Sanyal, Werner, Ziegler 2009, DCG | `sanyal2009kalai` | [checked] | Kalai's 3^d conjecture: Hanner polytopes have exactly 3^d nonempty faces; conjecture A holds for d ≤ 4. |
| Freij, Henze, Schmitt, Ziegler 2013, Electron. J. Combin. 20(2) P32 | `freij2013hansen` | [checked] | A Hansen polytope is Hanner iff its graph is threshold; Hanner and Hansen classes differ. |
| Kim 2014, J. Funct. Anal. 266(4); arXiv 1212.2544 | `kim2014hanner` | [checked] | Every Hanner polytope is a strict local minimiser of the volume product (Banach–Mazur). Uses a graph representation without induced P4, i.e. cographs. |

### The Mahler conjecture: state of knowledge

- Mahler 1939 (`mahler1939uebertragung`, [checked]) posed the volume-product problem.
- Proven classes: unconditional bodies (Saint-Raymond 1981, `saintraymond1981`; equality
  cases Meyer 1986, `meyer1986caracterisation`, and Reisner), zonoids (Reisner 1986,
  `reisner1986zonoids`) [secondary for details].
- Dimension 3: Iriyeh, Shibata 2020, Duke 169(6) 1077–1134 (`iriyeh2020mahler`, [checked]);
  shorter proof by Fradelizi, Hubard, Meyer, Roldán-Pensado, Zvavitch, Amer. J. Math.
  144(5) 1201–1219 (`fradelizi2022mahler`, [checked]; title to confirm).
- Local minimality: cube (Nazarov, Petrov, Ryabogin, Zvavitch 2010, Duke 154(3) 419–430,
  `nazarov2010mahler`, [checked]); all Hanner polytopes (Kim 2014).
- 2026 preprints: Chen, Li, Xi, Xu claim the non-symmetric 3D case (arXiv 2605.09334)
  [secondary]; several other 2026 preprints on local minimality.

### OpenAI Family 087 [read: abstract, Theorem 1.1, introduction; repository index]

- "The symmetric Mahler conjecture and its equality cases", OpenAI, dated September 22,
  2026, in github.com/openai/math (released in October 2026; pinned to commit
  `4eaee87`). Key `openai2026mahler`; the repository asks to cite each manuscript with the
  BibTeX in its directory.
- **Theorem 1.1** (verbatim in substance): for every n ≥ 1 and every origin-symmetric convex
  body K ⊂ R^n, |K||K°| ≥ 4^n/n!, with equality iff K is an invertible linear image of a
  Hanner polytope. The paper defines Hanner polytopes through ℓ∞ products and ℓ1 sums,
  exactly our cotree nodes.
- **Formalisation:** `lean/formalization.yaml` lists Lean declarations for both the
  inequality (`OAI.SymmetricMahler.symmetric_mahler`) and the equality characterisation
  (`OAI.SymmetricMahler.symmetric_mahler_equality`), with review status "unchecked". A
  companion manuscript claims the general (non-symmetric) conjecture; it does not appear
  in the formalisation index.
- **Status:** not peer reviewed. The README calls the manuscripts "raw research data" and
  says that some unformalised results "could have issues". Community notes
  (mathcommunity.github.io/oai-math-revised) list no corrections for Family 087.
- **Licence:** Apache-2.0 (repository).

**How to cite it in the paper.** "Hanner polytopes are the equality cases of the symmetric
Mahler conjecture, a claim recently made in an unrefereed, partly Lean-formalised manuscript
[openai2026mahler]; they were known before as the minimisers among unconditional bodies
and as local minimisers [saintraymond1981, meyer1986caracterisation, kim2014hanner]. Our
method uses only their combinatorial structure." This keeps the motivation without making
the paper depend on an unrefereed result. Reviewers in DMKD will not judge the Mahler proof,
but some will question it.

---

## 7. Low-precision and multiplication-free networks

| Work | Key | Status | Relevance |
| :--- | :--- | :--- | :--- |
| Ternary Weight Networks (Li et al. 2016), arXiv 1605.04711 | `li2016ternary` | [secondary] | Weights in {−1, 0, 1} with a scale; learned, not random. |
| XNOR-Net (Rastegari et al., ECCV 2016) | `rastegari2016xnor` | [checked] | Binary weights and inputs; XNOR and popcount. |
| AdderNet (Chen et al., CVPR 2020) | `chen2020addernet` | [secondary] | Replaces multiplications with ℓ1 distances. |

**Takeaway.** These reduce the cost of each weight. Hanner-Hydra goes further for groups:
the winner of up to 2^l ternary kernels costs O(l) additions, with no learning. One
sentence in related work is enough.

---

## 8. Benchmarks, evaluation methodology and recent context

### Archives and bake-offs [checked]

- UCR archive (Dau et al. 2019, IEEE/CAA J. Autom. Sinica 6(6) 1293–1305, `dau2019ucr`);
  UEA archive (Bagnall et al. 2018, arXiv 1811.00075, `bagnall2018uea`).
- Multivariate bake-off (Ruiz et al. 2021, DMKD 35(2) 401–449, `ruiz2021multivariate`);
  bake-off redux (Middlehurst, Schäfer, Bagnall 2024, DMKD 38(4) 1958–2031,
  `middlehurst2024bakeoff`): MultiRocket+Hydra and HC2 lead on UCR.
- HIVE-COTE 2.0 (Middlehurst et al. 2021, Machine Learning 110, 3211–3243,
  `middlehurst2021hc2`).
- MONSTER (Dempster et al. 2025, DMLR vol. 2, arXiv 2502.15122, `dempster2025monster`):
  argues UCR/UEA are small (median 217 and 255 cases), favouring low-variance methods; a
  collection of large datasets (10K–50M) with baselines including Hydra and QUANT. A
  reviewer may ask how Hanner-Hydra scales; MONSTER is the natural follow-up.
- aeon toolkit (Middlehurst et al. 2024, JMLR 25(289), `middlehurst2024aeon`): our
  baseline implementation and estimator interface.

### Statistics and protocol

- Bayesian signed-rank test with ROPE (Benavoli et al. 2017, JMLR 18(77),
  `benavoli2017bayesian`) [checked]; Demšar 2006 (`demsar2006statistical`) for
  Wilcoxon/Friedman and CD diagrams [checked].
- Multi-Comparison Matrix (Ismail-Fawaz et al. 2023, arXiv 2305.11921,
  `ismailfawaz2023mcm`) [secondary]: argues CD diagrams depend on the set of compared
  methods. Supports our choice of pairwise tests as primary and CD diagrams as secondary.
- Pre-registration in ML: NeurIPS 2020 workshop "The pre-registration experiment" and a
  2021 follow-up (`prereg2020neurips`) [checked]. Few ML papers are pre-registered; ours is
  frozen in a git tag rather than a public registry, which the paper should say.

### Recent work (2025–2026) [secondary]

- Hydra + QUANT ensembles on 10 MONSTER datasets (Maniar, arXiv 2512.06666,
  `maniar2025metalearning`): prediction-level combination captures little of the oracle
  gain; feature concatenation works better.
- RocketPFN (O'Rourke, Trisovic, Bertsimas, arXiv 2606.21786, `orourke2026rocketpfn`):
  ROCKET features with TabPFN; reports parity with HC2 on 92 UCR datasets.
- MASHT (arXiv 2607.19234, NeurIPS 2026 listing): MultiRocket + Hydra features into a
  tabular in-context model; reports a lower average rank than HC2 on univariate data.
- **Implication:** Hydra-style features are now used as inputs to stronger classifiers.
  Hanner features are a cheaper drop-in for that role, which is worth one sentence in the
  discussion (future work), not a claim.

---

## 9. What this means for the paper

**Claims we can make** (supported by the literature and our results):
- First structured (polytope-based) competing kernel groups; O(ℓ) winner for groups of up
  to 2^ℓ ternary kernels (to our knowledge).
- Accuracy practically equivalent to Hydra under a pre-registered non-inferiority rule,
  11.2× faster and 23.8× less memory on one core, with theory-matched controls showing the
  speed comes from the cotree.
- Interpretation through hyperplane and cross-polytope hashing.

**Claims to avoid or qualify:**
- "Optimal" or "extremal" kernels: the Mahler extremality is affine-invariant and says
  nothing about classification.
- Any dependence on the Mahler theorem being proven: cite it as a claim.
- State-of-the-art accuracy: HC2, MultiRocket(+Hydra), QUANT and in-context methods are
  more accurate.
- Direct numeric comparison with NanoHydra: different protocols and hardware.

**Reviewer questions to prepare for:**
1. Microcontroller or integer implementation, and comparison with NanoHydra.
2. Scaling to large datasets (MONSTER).
3. Combination with MultiRocket or QUANT (the strongest pipelines use Hydra in a
   combination).
4. The UEA gap and the inconclusive balanced accuracy and macro-F1.
5. Why Hanner polytopes rather than other structured families with fast winners (any
   product of cross-polytopes, or plain sign patterns). The controls answer part of this;
   a cube-only or cross-polytope-only ablation would answer more.

**Open checks before submission:** resolve every `VERIFY` note in `references.bib`; read
the MiniRocket MCU report and S-Rocket if they are cited; decide whether to cite 1D-LBP
and ordinal patterns with verified references; confirm the Fradelizi et al. title.

---

## 10. Search log

Hydra and its successors: "NanoHydra time series classification Hydra microcontroller";
"MultiRocket-Hydra MR-Hydra"; "QUANT interval quantiles"; "MONSTER Monash scalable time
series evaluation repository"; "Hydra competing convolutional kernels 2025 2026 extension";
"The Meta-Learning Gap: Combining Hydra and Quant"; "In-Context Time Series Classification
with Random Convolutional Features MASHT"; "RocketPFN"; "bake off redux 30 resamples".

Cheap kernels and edge: "binary OR ternary random convolutional kernels MiniRocket
microcontroller FPGA"; "Detach-ROCKET"; "TinyML time series classification microcontroller
ROCKET MiniRocket energy"; "ternary weight networks multiplication-free AdderNet XNOR-Net".

Benchmarks and methods: "WEASEL 2.0"; "MiniRocket KDD 2021"; "ROCKET DMKD 2020";
"MultiRocket DMKD 2022"; "aeon toolkit JMLR 2024"; "UCR time series archive 2019";
"Multi-Comparison Matrix"; "preregistration machine learning NeurIPS workshop".

Patterns and hashing: "one-dimensional local binary patterns"; "ordinal patterns Bandt
Pompe classification"; "cross-polytope LSH Andoni 2015"; "Charikar 2002 SimHash";
"winner-take-all hashing Yagnik 2011"; "tropical geometry maxout support function".

Convex geometry: "Hanner polytopes Kalai 3^d Sanyal Werner Ziegler"; "Mahler conjecture
Hanner local minimizers Kim Reisner"; "Mahler conjecture proof 2026 equality cases Lean";
"github openai/math Mahler Family 087"; "Iriyeh Shibata Duke 2020"; "Hansen polytopes
threshold graphs"; "Hansen Lima 1981 Acta Mathematica"; "Corneil Lerchs Stewart Burlingham
1981"; "Saint-Raymond unconditional Meyer Reisner zonoids"; "Kim minimal volume product near
Hanner polytopes"; "Nazarov Petrov Ryabogin Zvavitch local minimality cube"; "Mahler 1939".

---

## 11. Sources

Random-kernel methods and successors
- [ROCKET, arXiv 1910.13051](https://arxiv.org/pdf/1910.13051v1.pdf)
- [MiniRocket, arXiv 2012.08791](https://arxiv.org/abs/2012.08791)
- [MultiRocket, arXiv 2102.00457](https://arxiv.org/pdf/2102.00457)
- [Hydra, arXiv 2203.13652](https://arxiv.org/pdf/2203.13652) · [DMKD version](https://link.springer.com/10.1007/s10618-023-00939-3) · [code](https://github.com/angus924/hydra)
- [QUANT, arXiv 2308.00928](https://arxiv.org/pdf/2308.00928) · [code](https://github.com/angus924/quant)
- [S-Rocket, arXiv 2203.03445](https://arxiv.org/pdf/2203.03445)
- [Detach-ROCKET, arXiv 2309.14518](https://arxiv.org/html/2309.14518v3)
- [NanoHydra, arXiv 2510.20038](https://arxiv.org/abs/2510.20038) · [HTML](https://arxiv.org/html/2510.20038) · [code](https://github.com/pulp-platform/nanohydra)
- [ETH report: MiniRocket on a low-power tag](https://www.research-collection.ethz.ch/server/api/core/bitstreams/b5bd8918-f595-4997-b36d-20649b13816f/content)
- [Meta-learning gap (Hydra + QUANT), arXiv 2512.06666](https://arxiv.org/pdf/2512.06666)
- [RocketPFN, arXiv 2606.21786](https://arxiv.org/pdf/2606.21786)
- [MASHT, arXiv 2607.19234](https://arxiv.org/pdf/2607.19234) · [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/162153)

Dictionary, patterns, hashing, low precision
- [WEASEL 2.0, arXiv 2301.10194](https://arxiv.org/pdf/2301.10194v2.pdf) · [Machine Learning version](https://link.springer.com/doi/10.1007/s10994-023-06395-w)
- [1D-LBP for EEG (Kaya et al.)](https://avesis.uludag.edu.tr/yayin/9591f8b2-158c-4222-8c9c-ea4adcac9bc1/1d-local-binary-pattern-based-feature-extraction-for-classification-of-epileptic-eeg-signals)
- [Multivariate ordinal patterns, Discovery Science 2018](https://link.springer.com/10.1007/978-3-030-01771-2_15) · [20 years of ordinal patterns](https://www.ifisc.uib-csic.es/en/publications/20-years-of-ordinal-patterns-perspectives-and-chal)
- [Charikar 2002](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p380-charikar.pdf)
- [Cross-polytope LSH, arXiv 1509.02897](https://arxiv.org/pdf/1509.02897) · [NeurIPS proceedings](https://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance)
- [Winner-take-all hashing (Yagnik et al. 2011)](https://www.cs.toronto.edu/~dross/YagnikStrelowRossLin_ICCV2011.pdf)
- [Maxout polytopes, arXiv 2509.21286](https://arxiv.org/html/2509.21286) · [Charisopoulos, Maragos, arXiv 1805.08749](https://www.arxiv.org/abs/1805.08749)
- [Ternary Weight Networks, arXiv 1605.04711](https://alphaxiv.org/abs/1605.04711) · [XNOR-Net, arXiv 1603.05279](https://arxiv.org/pdf/1603.05279) · [AdderNet, arXiv 1912.13200](https://arxiv.org/pdf/1912.13200)

Convex geometry
- [OpenAI math repository](https://github.com/openai/math) · [community notes](https://mathcommunity.github.io/oai-math-revised/) · [community Lean scope for 087](https://github.com/mathcommunity/oai-math-revised/blob/main/lean/docs/087.md)
- [Sanyal, Werner, Ziegler, arXiv 0708.3661](https://arxiv.org/pdf/0708.3661)
- [Freij, Henze, Schmitt, Ziegler, Electron. J. Combin.](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v20i2p32)
- [Kim, arXiv 1212.2544](https://arxiv.org/pdf/1212.2544)
- [Nazarov, Petrov, Ryabogin, Zvavitch, arXiv 0905.0867](https://arxiv.org/pdf/0905.0867)
- [Iriyeh, Shibata, Duke](https://projecteuclid.org/journals/duke-mathematical-journal/volume-169/issue-6/----Custom-HTML----Symmetric/10.1215/00127094-2019-0072.short) · [arXiv 1904.10765](https://arxiv.org/pdf/1904.10765)
- [Hansen, Lima 1981 (MaRDI record)](https://portal.mardi4nfdi.de/wiki/Item:Q1163720)
- [Corneil, Lerchs, Stewart Burlingham 1981 (dblp)](https://dblp.uni-trier.de/rec/journals/dam/CorneilLB81.html)
- [Mahler 1939 (DML-CZ)](https://dml.cz/handle/10338.dmlcz/109441?show=full)
- [Mahler volume (overview)](https://en.wikipedia.org/wiki/Mahler_volume) · [Chen, Li, Xi, Xu, arXiv 2605.09334](https://arxiv.org/pdf/2605.09334)

Benchmarks and methodology
- [UCR archive, arXiv 1810.07758](https://arxiv.org/pdf/1810.07758)
- [Bake off redux, arXiv 2304.13029](https://arxiv.org/pdf/2304.13029)
- [HIVE-COTE 2.0](https://link.springer.com/doi/10.1007/s10994-021-06057-9)
- [MONSTER, arXiv 2502.15122](https://arxiv.org/pdf/2502.15122)
- [aeon, JMLR 25](https://www.jmlr.org/papers/v25/23-1444.html)
- [Multi-Comparison Matrix, arXiv 2305.11921](https://arxiv.org/pdf/2305.11921) · [code](https://github.com/MSD-IRIMAS/Multi_Comparison_Matrix)
- [NeurIPS 2020 pre-registration workshop](https://neurips.cc/virtual/2020/workshop/16158) · [2021 follow-up](https://nips.cc/virtual/2021/workshop/21885)
