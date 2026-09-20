# Objective

To develop a python package that can be used to generate alignments of cognates within an comparative dictionary formatted as CLDF. The alignment algorithm will generate a CLDF-compliant output file with phoneme alignments expressed with `|` delimiters.

# Overview

- Alignments will be based on the 24 articulatory features represented in [PanPhon](https://github.com/dmort27/panphon)
- These features will be used to compute the substitution cost of aligning phonemes within cognate sets.
- Weights for each of these features will be learned empirically, via expectation-maximization (EM).
- Alignments involve multiple different languages (and are multi-source)
- It is probably not computationally feasible to compute all $n$ alignments (for $n$ languages, where $n$ is arbitrarily large)
- Instead, it is desirable to choose one language as an anchor, using a heuristic, and then align each of the other languages to this language
- This should be repeated for all the languages and the anchor language that maximizes the mean probability of the alignments should be chosen, along with the associated weights.
- You can experiment with both aggregating the weights across non-anchor languages and learning a separate set of weights for each anchor-non-anchor pair.
- There should be both a library implementing this functionality and a script, allow it to be employed at the command line.

## Initialization

- The agent should experiment with various kinds of initialization, both for the weights and for the alignments.
- Weights can be initialized as $\frac{1}{24}$ plus a small random value, centered around `0`
- Alignments can be initialized using the Levenshtein dynamic programming algorithm, but with the substitution cost proportional to $w \cdot (x_1 - x_2)$ where $w$ is the weights and $x_1 - x_2$ is the element-wise difference between the features in phones $p_1$ and $p_2$.

## Optimization

- Optimization should proceed as an integrative process, following the EM algorithm.
- The agent should try various algorithms for updating the weights, but it should start with SGD.

# Inputs

The input to the primary function of the library is a comparative dictionary in [CLDF format](https://cldf.clld.org/). The forms are in IPA. An example is provided in `data/tangkhulic_cldf`. CLDF data consists of CSV data and JSON metadata. The following tables and fields are important for this task:

- `forms.ID`: the ID for a form (keys into `cognates.Form_ID`
- `forms.Language_ID`: the identifier of the language (keys into 
- `forms.Form`: the IPA representation of the form, with morphs segmented using `+`
- `cognates.Form_ID`: the ID for a form (keys into `forms.ID`
- `cognates.Cognateset_ID`: the ID of the cognate set to which `cognates.Form_ID` is asserted to belong.
- `cognates.Morph_Index`: the index (starting at zero) of the morph that belongs to `cognates.Cognateset_ID`
- `Languages.ID`: the language to which a form belongs (keys into `forms.Language_ID`.

The data are sparse, meaning that many cognate sets will not include forms for every language.

# Outputs

The output of the primary function in the library and of the script should be a CLDF-compliant CSV file `alignments` with the following fields:

- `alignments.ID`: a (unique) primary key for `alignments`.
- `alignments.Form_ID`: the form ID for the aligned form.
- `alignments.Cognateset_ID`: the ID of the cognate set with respect to which the form is aligned.
- `alignments.Aligned_Form`: the aligned form, with segments indicated with pipe (`|`)

# Constraints

- All code should be in clear, well-commented, idiomatic Python
- The agent should review its code critically, as if it were Linus Torvalds reviewing a code commit to the Linux kernel.
- At the end of each cycle of edits, the agent should go through the code and look for areas of possible code reuse or areas of redundancy

# Tools

The agent should build Python tools for exploring and evaluating the alignments when such functionality does not make sense as part of the library (for example, when it needs to be executed only as a script).

# Workflow

1. The agent should implement the project, following general guidance in `AGENTS.md` and specific requests from the user.
2. Tests should be run on the resulting code.
3. If all tests pass, the agent should make a pull request
4. The user retains responsibility for approving the pull request


