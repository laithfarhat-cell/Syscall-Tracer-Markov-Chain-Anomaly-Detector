# Statistical model and security interpretation

## 1. From a finite-state automaton to a DTMC

The input is a sequence of normalized symbols \(x_1,\ldots,x_T\) from one recording stream. A finite-state automaton captures which transitions can occur; it does not distinguish a common path from a rarely traveled legal path. A **discrete-time Markov chain** attaches a categorical distribution to each state's outgoing transitions. Time is counted in syscalls, not seconds. The transition rule is fitted to an application version, mode of operation, and telemetry scope; it is not a claim that a full operating system is Markovian or time-homogeneous.

For a first-order model the factorization follows from the chain rule and the Markov assumption:

$$
\begin{aligned}
\Pr(X_{1:T}=x_{1:T})
 &= \Pr(X_1=x_1)\prod_{t=1}^{T-1}\Pr(X_{t+1}=x_{t+1}\mid X_{1:t}=x_{1:t}) \\
 &\overset{\text{assumption}}{=} \Pr(X_1=x_1)\prod_{t=1}^{T-1}\Pr(X_{t+1}=x_{t+1}\mid X_t=x_t).
\end{aligned}
$$

The vocabulary \(S\) contains every token observed in training and one reserved \(\texttt{<UNK>}\) symbol. At calibration and inference out-of-vocabulary observations map to \(\texttt{<UNK>}\) for probability calculations. Alerts preserve the original normalized tokens. No initial-state distribution is fitted: the detector conditions on the initial context inside every window.

## 2. Order \(k\) as a first-order chain on contexts

Replace the first-order assumption with conditioning on the preceding \(k\) tokens. Define the augmented state \(C_t=(X_{t-k+1},\ldots,X_t)\), for \(t\ge k\). An edge from \(c=(u_1,\ldots,u_k)\) to \((u_2,\ldots,u_k,s)\) has probability \(P(s\mid c)\); other edges between augmented states are structurally impossible. This is a first-order DTMC on \(S^k\), often called an \((k+1)\)-gram model because a count includes both context and destination. It need not store a \(|S|^k \times |S|^k\) dense matrix. The profile stores only positive counts keyed by observed contexts and destinations. The diagnostic heatmap is context × destination, not a dense augmented-state matrix.

Counts never bridge different PIDs/TIDs, stream IDs, sources, files, or the automatic holdout boundary. Interleaving multiple threads into one chain would model scheduler order, not the application's per-thread behavior. More history can distinguish different phases reaching the same syscall, but longer contexts increase estimation variance and the number of unobserved contexts. Start at \(k=1\) and compare candidate orders on multiple benign sessions.

## 3. Maximum likelihood and additive smoothing

Let \(N_{cs}\) count the training transitions from context \(c\) to symbol \(s\), and \(N_c=\sum_s N_{cs}\). Maximizing the conditional multinomial log-likelihood \(\sum_s N_{cs}\log P(s\mid c)\) subject to \(\sum_s P(s\mid c)=1\) gives the unsmoothed MLE. In a nonempty row, the Lagrange condition for positive counts is \(N_{cs}/P(s\mid c)=\lambda\); imposing the simplex constraint gives \(\lambda=N_c\), hence \(\widehat P_{\text{MLE}}(s\mid c)=N_{cs}/N_c\). Zero-count destinations receive zero MLE mass; the data provide no estimate for a wholly unobserved row.

For scoring, add a strictly positive pseudo-count \(\alpha\) to every destination. This is the posterior mean under a symmetric Dirichlet prior with parameters \((\alpha,\ldots,\alpha)\), not the unsmoothed MLE:

$$
P_\alpha(s\mid c)=\frac{N_{cs}+\alpha}{N_c+\alpha |S|},\qquad
\sum_{s\in S}P_\alpha(s\mid c)=\frac{N_c+\alpha |S|}{N_c+\alpha |S|}=1.
$$

An unseen row is uniform; a missing edge in a well-observed row has small, positive mass. This avoids infinite surprise from a single unseen observation. It also means an unseen context can assign a **larger** probability than a zero-count edge in a highly specialized context, so the absence of a count matters separately to the critical rule. Warm-up critical checks use the actual available suffix of length 1 to \(k-1\), with its own fitted counts. They are not backoff scores for full windows.

Incremental fitting adds counts and may grow \(S\). Even rows with no new observations can change because \(\alpha |S|\) changes. A fitted threshold is invalidated by any nonempty update: validate and recalibrate before using the updated profile. For numeric stability the implementation accepts \(10^{-100}\le\alpha\le10^{100}\); typical tuning is many orders of magnitude narrower.

## 4. Window surprise and information theory

For \(W>k\), the conditional likelihood of a window is the product over its \(W-k\) conditional transitions:

$$
\mathcal L(W)=-\frac{1}{W-k}\sum_{i=k+1}^{W}\log_2 P_\alpha(x_i\mid x_{i-k:i-1}).
$$

With \(k=1\), the denominator is \(W-1\). The units are bits per conditional transition. Computation uses sums of log-probabilities instead of multiplying small numbers. A rolling queue retains the last \(W-k\) full-order transition costs. At the window's end these costs start at its \((k+1)\)-th event, so **no context preceding the window** enters the sum.

Let \(q(c,s)\) be the empirical context-destination distribution of the window, \(q(c)\) its context marginal, and \(q(s\mid c)\) its conditional. Grouping terms by \((c,s)\) gives conditional cross-entropy against the baseline:

$$
\begin{aligned}
\mathcal L(W)
 &= -\sum_c q(c)\sum_s q(s\mid c)\log_2 P_\alpha(s\mid c) \\
 &= \sum_c q(c)\bigl[H_2(q(\cdot\mid c)) + D_{\rm KL}(q(\cdot\mid c)\,\|\,P_\alpha(\cdot\mid c))\bigr].
\end{aligned}
$$

The score is not a probability of intrusion or an entropy estimate for the full application. It rises for surprising edges and also reflects how varied the workload in the window is. In particular, high entropy or mode changes in a legitimate workload can produce drift.

## 5. Calibration and dual decisions

1. Fit the model on trusted training captures. Score every **complete** held-out benign window using the same normalization, order, window length, and stream separation as monitoring. Set \(\tau\) to the empirical 99th percentile by default (linear interpolation between adjacent order statistics at rank \((n-1)\cdot0.99\)). Record the sample count, median, and MAD alongside it.
2. Flag **sequence drift** when surprise exceeds \(\tau\), allowing a tiny rounding tolerance at the threshold. A 99th percentile is not an independent-event p-value, a guarantee of 1% future alerting, or a threshold for incident severity. Overlapping windows share transitions; estimate operational false alarm rates and latency with independent captures and tag changes in workload.
3. Independently flag a **critical transition** when the newest edge involves a configured high-risk endpoint or completes the configured pair, and either its empirical count is zero or its smoothed probability is below or equal to the cutoff. This catches short events diluted by averaging and events before the first full window. It may also flag normal loader, debugger, or privilege-management behavior missing from the baseline. The first observed call in a stream has no incoming transition and cannot be scored as one. Critical alerts are evaluated on the newest edge only; drift alerts may repeat as overlapping windows shift. Severity is a rule-based triage score, not a calibrated posterior.

Window-level false positives, scenario-level detection, and elapsed detection latency are distinct metrics. The provided lab exercises both rules with labeled **synthetic records**, not with actual exploits. Quantify precision and recall only against representative, independently labeled observations in the intended deployment.

## 6. Security boundary and mimicry

The detector observes one projection of behavior. If two executions have the same normalized syscall sequence and stream partition, a detector using only those observations has no basis for distinguishing intent. This is the **mimicry** limitation. Argument abstraction can hide the difference between two destinations, files, or privilege contexts; even a larger order cannot distinguish identical observations. Missing syscall coverage, telemetry loss, compromised logs or sensor, multiprocessing behavior outside the modeled stream, and legitimate software drift also limit inference. A high score does not identify code injection, privilege escalation, or control-flow hijacking by itself.

Defensive improvements include profiles scoped to workload and software version; a small, reviewed set of security-relevant argument categories; correlation with authenticated audit, process ancestry, identity, file integrity, and network telemetry; and independent least-privilege controls. Protect profile integrity and the sensor's access to telemetry. Rebaseline only after reviewing benign changes, never automatically on suspicious streams. Keep case review and enforcement separate: this tool reports evidence, not an enforcement decision.

### Further reading

- S. Forrest et al., *A Sense of Self for Unix Processes*, IEEE Symposium on Security and Privacy, 1996.
- D. Wagner and P. Soto, *Mimicry Attacks on Host-Based Intrusion Detection Systems*, ACM CCS, 2002.
