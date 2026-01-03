

# Interactive Bayesian Inference Explorer (Shiny)

This repository contains a Shiny application for interactively exploring Bayesian inference in canonical one-parameter models. The purpose is not merely to “compute a posterior”, but to expose (i) **the algebraic structure of Bayes’ rule**, (ii) **conjugacy as an identity of kernels**, and (iii) the way in which priors act as **regularisers / pseudo-data**.

The current Shiny app implements, in executable form:

- **Binomial likelihood** with either
  - **discrete prior** over $\theta$, or
  - **continuous conjugate Beta prior**.
- **Poisson likelihood** with either
  - **discrete prior** over $\lambda$, or
  - **continuous conjugate Gamma prior**.

Tabs titled **Normal** and **MCMC Simulations** exist in the UI, but their server-side inference logic is currently a scaffold / placeholder and should be treated as **work-in-progress**.

-----

# Mathematical foundations

## Notation

Let $y$ denote observed data, $\theta$ a parameter, and $\pi(\theta)$ a prior density (or mass function). We write:

- Likelihood: $L(\theta; y) = p(y \mid \theta)$
- Prior: $\pi(\theta)$
- Posterior: $\pi(\theta \mid y)$
- Evidence (marginal likelihood): $p(y)$

### The Bayesian model as a joint distribution

The Bayesian paradigm begins by specifying a **joint distribution** for all unknown and observed quantities. In the simplest parametric setting:

$
p(\theta,y) = p(y\mid\theta)\,\pi(\theta).
$

The posterior is a conditional distribution derived from this joint:

$
\pi(\theta\mid y) = \frac{p(\theta,y)}{p(y)}.
$

In hierarchical models, one introduces hyperparameters $\eta$ and potentially latent variables $z$, yielding

$
p(\theta,\eta,z,y) = p(y\mid z,\theta,\eta)\,p(z\mid\theta,\eta)\,p(\theta\mid\eta)\,p(\eta).
$

The posterior becomes $p(\theta,\eta,z\mid y)$ and is typically high-dimensional, motivating MCMC.

### The evidence (marginal likelihood) and its role

The evidence is defined by marginalising the joint:

$
p(y)=\int p(y\mid\theta)\,\pi(\theta)\,d\theta.
$

It is both:

- a **normalising constant** making $\pi(\theta\mid y)$ integrate to 1, and
- a fundamental quantity for **Bayesian model comparison**, e.g. Bayes factors.

For two models $\mathcal{M}_1,\mathcal{M}_2$ with priors $p(\mathcal{M}_k)$ and evidences $p(y\mid\mathcal{M}_k)$,

$
\frac{p(\mathcal{M}_1\mid y)}{p(\mathcal{M}_2\mid y)}
= \frac{p(y\mid\mathcal{M}_1)}{p(y\mid\mathcal{M}_2)}\cdot\frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}.
$

The ratio $\mathrm{BF}_{12}=p(y\mid\mathcal{M}_1)/p(y\mid\mathcal{M}_2)$ is the **Bayes factor**.

### Posterior predictive distribution

Bayesian inference is often ultimately about predicting new data $\tilde y$. The posterior predictive is

$
p(\tilde y\mid y)=\int p(\tilde y\mid\theta)\,\pi(\theta\mid y)\,d\theta.
$

This integral is analytically tractable in many conjugate models (see below) and is a core reason conjugate families are pedagogically powerful.

### Bayesian decision theory: point estimates as Bayes actions

Given a loss function $\ell(a,\theta)$ for an action/estimate $a$, the Bayes action minimises posterior expected loss:

$
a^*(y)=\arg\min_a\ \mathbb{E}[\ell(a,\theta)\mid y].
$

Common special cases:

- Squared error loss: $\ell(a,\theta)=(a-\theta)^2$ gives $a^*(y)=\mathbb{E}[\theta\mid y]$.
- Absolute error loss: $\ell(a,\theta)=|a-\theta|$ gives $a^*(y)=\mathrm{median}(\theta\mid y)$.
- 0–1 loss (classification): Bayes action is the posterior mode (MAP) for discrete $\theta$.

These identities are not mere heuristics: they are theorems from decision theory.

### Bayes’ theorem as conditioning (measure-theoretic form)

In its most general form, Bayes’ rule is a statement about **conditional probability measures**.

Let $(\Omega,\mathcal{F},\mathbb{P})$ be a probability space and suppose $\theta$ and $y$ are random elements on measurable spaces $(\Theta,\mathcal{B}_\Theta)$ and $(\mathcal{Y},\mathcal{B}_\mathcal{Y})$. If $\mathbb{P}$ admits a joint distribution on $\Theta\times\mathcal{Y}$, then a posterior is a **regular conditional distribution** $\mathbb{P}(\theta\in\cdot\mid y)$.

When densities exist with respect to a base measure (Lebesgue for continuous; counting for discrete), Bayes’ rule reduces to the familiar density identity. When densities do not exist, Bayes’ rule is still valid at the level of measures.

### Posterior as penalised likelihood and MAP as regularisation

For continuous $\theta$, MAP estimation is often presented as

$
\theta_{\mathrm{MAP}}\in \arg\max_\theta\ \pi(\theta\mid y)
=\arg\max_\theta\ \big[\log p(y\mid\theta)+\log\pi(\theta)\big].
$

Thus the prior contributes an additive penalty $-\log\pi(\theta)$ to the log-likelihood. Many classical regularisers are MAP priors:

- Gaussian prior on coefficients $\beta$ corresponds to $\ell_2$ (ridge) penalty.
- Laplace prior corresponds to $\ell_1$ (lasso) penalty.

### Kullback–Leibler characterisation of the posterior

The posterior can be characterised as the distribution $q$ that minimises a variational objective:

$
\pi(\theta\mid y) = \arg\min_q\ \mathrm{KL}\big(q(\theta)\ \|\ \pi(\theta\mid y)\big),
$

and equivalently maximises the evidence lower bound (ELBO):

$
\log p(y) = \mathcal{L}(q) + \mathrm{KL}\big(q(\theta)\ \|\ \pi(\theta\mid y)\big),
\qquad
\mathcal{L}(q)=\mathbb{E}_q[\log p(y,\theta)]-\mathbb{E}_q[\log q(\theta)].
$

Variational Bayes approximates $\pi(\theta\mid y)$ by restricting $q$ to a tractable family (e.g. mean-field), trading exactness for speed.

### Exchangeability and de Finetti (why i.i.d. models appear)

Many Bayesian models are justified by **exchangeability**: a sequence $Y_1,Y_2,\dots$ is exchangeable if its joint distribution is invariant under finite permutations.

De Finetti’s theorem (for Bernoulli sequences) states that an infinite exchangeable Bernoulli sequence is conditionally i.i.d. given a latent $\theta$:

$
\Pr(Y_{1:n}=y_{1:n}) = \int \prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i}\,d\Pi(\theta).
$

Thus, the Bayesian “parameter” $\theta$ can be viewed as a representation of exchangeable structure rather than a physical constant.

### Prior construction: conjugate, weakly-informative, reference priors

Common approaches to choosing $\pi(\theta)$:

- **Conjugate priors** for analytic tractability and interpretability (pseudo-data updates).
- **Weakly informative priors** (e.g. Normal with large scale) to stabilise inference without overwhelming the likelihood.
- **Reference / objective priors** such as Jeffreys’ prior $\pi_J(\theta)\propto \sqrt{I(\theta)}$, where $I(\theta)$ is Fisher information. These aim at invariance properties but can be improper.

-----

All Bayes updates in this app are of the form:

$
\pi(\theta \mid y) = \frac{p(y \mid \theta)\,\pi(\theta)}{\int p(y \mid \theta)\,\pi(\theta)\,d\theta}.
$

When $\theta$ is discrete (finite support), the integral is replaced by a sum.

## Bayes’ rule (discrete vs continuous) and normalisation

### Discrete parameter space

If $\Theta = \{\theta_1,\dots,\theta_m\}$ and $\pi_i = \Pr(\theta=\theta_i)$, then

$
\Pr(\theta=\theta_i \mid y)
= \frac{p(y \mid \theta_i)\,\pi_i}{\sum_{j=1}^m p(y \mid \theta_j)\,\pi_j}.
$

This is exactly what the “Discrete” modes compute and display as a table (prior, likelihood, product, posterior).

### Continuous parameter space

If $\theta$ is continuous with density $\pi(\theta)$, then

$
\pi(\theta\mid y) \propto p(y\mid\theta)\,\pi(\theta),
\qquad
\text{with normaliser } p(y)=\int p(y\mid\theta)\,\pi(\theta)\,d\theta.
$

In conjugate families, this integral has a closed form because the posterior kernel remains in the same parametric family.

## Exponential families and conjugate priors (unifying view)

Many classical likelihoods used in this project belong to the exponential family:

$
p(y\mid\theta)=h(y)\exp\big(\eta(\theta)^\top T(y)-A(\theta)\big),
$

where $T(y)$ is a sufficient statistic and $\eta(\theta)$ the natural parameter. A conjugate prior is typically of the form

$
\pi(\theta\mid\lambda,\nu)\propto \exp\big(\eta(\theta)^\top\lambda-\nu A(\theta)\big),
$

so that the posterior updates by **adding sufficient statistics**:

$
\lambda' = \lambda + T(y),\qquad \nu' = \nu + 1 \quad (\text{or } \nu' = \nu + n \text{ for } n \text{ i.i.d. observations}).
$

The Beta–Binomial and Gamma–Poisson updates are concrete instances of this general mechanism.

## Sufficient statistics and Bayesian updating as information accumulation

In the models implemented here, the data influence the posterior only through low-dimensional sufficient statistics:

- Binomial: $x$ (successes) and $n$ (trials).
- Poisson i.i.d.: $S=\sum_i y_i$ and $n$ (number of observations).

This is not an accident; it is a consequence of exponential family structure and is one of the most important practical insights in Bayesian modelling: high-dimensional raw data are compressed into statistics that fully determine the likelihood.

-----

# Model 1: Binomial likelihood

## Likelihood and sufficient statistic

Let $X \mid \theta \sim \text{Binomial}(n,\theta)$, with observed successes $x\in\{0,\dots,n\}$. The likelihood is

$
p(x\mid\theta) = \binom{n}{x}\theta^x(1-\theta)^{n-x},\qquad 0<\theta<1.
$

The likelihood kernel in $\theta$ is

$
p(x\mid\theta) \propto \theta^x(1-\theta)^{n-x}.
$

## A) Discrete prior on $\theta$

Choose a finite grid $\Theta=\{\theta_1,\dots,\theta_m\}\subset(0,1)$ and prior masses $\pi_i$ (with $\sum_i \pi_i=1$). Then

$
\Pr(\theta=\theta_i\mid x)
= \frac{\binom{n}{x}\theta_i^x(1-\theta_i)^{n-x}\,\pi_i}{\sum_{j=1}^m \binom{n}{x}\theta_j^x(1-\theta_j)^{n-x}\,\pi_j}
= \frac{\theta_i^x(1-\theta_i)^{n-x}\,\pi_i}{\sum_{j=1}^m \theta_j^x(1-\theta_j)^{n-x}\,\pi_j}.
$

### Posterior moments on a grid

With discrete posterior masses $\tilde\pi_i$, posterior expectations are direct:

$
\mathbb{E}[\theta\mid x] = \sum_{i=1}^m \theta_i\,\tilde\pi_i,
\qquad
\mathrm{Var}(\theta\mid x)=\sum_{i=1}^m (\theta_i-\mathbb{E}[\theta\mid x])^2\tilde\pi_i.
$

These are not currently displayed in the UI, but the table is sufficient to compute them.

## B) Continuous conjugate prior: $\theta\sim\mathrm{Beta}(\alpha,\beta)$

### Prior definition and normalising constant

The Beta density is

$
\pi(\theta) = \frac{1}{\mathrm{B}(\alpha,\beta)}\,\theta^{\alpha-1}(1-\theta)^{\beta-1},
\qquad 0<\theta<1,
$

where the Beta function

$
\mathrm{B}(\alpha,\beta)=\int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\,dt
= \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}.
$

### Conjugacy proof (kernel argument)

Bayes’ rule gives

$
\pi(\theta\mid x)
\propto p(x\mid\theta)\pi(\theta)
\propto \big[\theta^x(1-\theta)^{n-x}\big]\,\big[\theta^{\alpha-1}(1-\theta)^{\beta-1}\big]
= \theta^{\alpha+x-1}(1-\theta)^{\beta+n-x-1}.
$

The RHS is the kernel of a Beta distribution. Therefore,

$
\theta\mid x \sim \mathrm{Beta}(\alpha+x,\;\beta+n-x).
$

This algebraic update is exactly what the Binomial tab implements.

### Posterior mean, variance, and MAP

For $\theta\mid x \sim \mathrm{Beta}(\alpha',\beta')$ with $\alpha'=\alpha+x$, $\beta'=\beta+n-x$:

$
\mathbb{E}[\theta\mid x] = \frac{\alpha'}{\alpha'+\beta'},
\qquad
\mathrm{Var}(\theta\mid x)=\frac{\alpha'\beta'}{(\alpha'+\beta')^2(\alpha'+\beta'+1)}.
$

If $\alpha',\beta'>1$, the posterior mode (MAP) is

$
\theta_{\mathrm{MAP}} = \frac{\alpha'-1}{\alpha'+\beta'-2}.
$

### Interpretation as pseudo-counts

The Beta prior can be interpreted as contributing pseudo-successes $\alpha-1$ and pseudo-failures $\beta-1$ (with caveats near the boundary). After observing $x$ successes in $n$ trials, the posterior adds them additively.

### Credible intervals

An equal-tailed $(1-\gamma)$ credible interval is

$
\big[q_{\gamma/2},\; q_{1-\gamma/2}\big],
\qquad
q_p = F^{-1}_{\mathrm{Beta}(\alpha',\beta')}(p).
$

The UI computes these via Beta quantiles.

### Posterior predictive distribution (Beta–Binomial)

Let $\tilde X\mid\theta \sim \mathrm{Binomial}(m,\theta)$ be future data. The posterior predictive is

$
p(\tilde x\mid x)
= \int_0^1 p(\tilde x\mid\theta)\,\pi(\theta\mid x)\,d\theta
= \binom{m}{\tilde x}\,\frac{\mathrm{B}(\alpha'+\tilde x,\beta'+m-\tilde x)}{\mathrm{B}(\alpha',\beta')}.
$

Thus $\tilde X\mid x\sim\mathrm{BetaBinomial}(m,\alpha',\beta')$.

-----

# Model 2: Poisson likelihood

## Likelihood and sufficient statistic

Assume counts $Y_1,\dots,Y_n\mid\lambda$ are conditionally i.i.d.

$
Y_i\mid\lambda \sim \mathrm{Poisson}(\lambda),
\qquad
p(y_i\mid\lambda)=e^{-\lambda}\frac{\lambda^{y_i}}{y_i!}.
$

Let $S=\sum_{i=1}^n y_i$. Then the joint likelihood is

$
p(y_{1:n}\mid\lambda)
= \prod_{i=1}^n e^{-\lambda}\frac{\lambda^{y_i}}{y_i!}
= e^{-n\lambda}\lambda^{S}\prod_{i=1}^n \frac{1}{y_i!}.
$

So the likelihood kernel is

$
p(y_{1:n}\mid\lambda)\propto \lambda^{S}e^{-n\lambda}.
$

In the current UI implementation, inputs are provided as:

- total count $S$ (named “Total Counts”), and
- number of observations $n$ (named “Number of Observations”).

## A) Discrete prior on $\lambda$

Choose a grid $\Lambda=\{\lambda_1,\dots,\lambda_m\}\subset (0,\infty)$ and prior masses $\pi_i$. Then

$
\Pr(\lambda=\lambda_i\mid y)
= \frac{p(y\mid\lambda_i)\pi_i}{\sum_{j=1}^m p(y\mid\lambda_j)\pi_j}.
$

In the UI, the likelihood for the discrete mode is computed using a Poisson PMF with parameter $\lambda_i$ at the observed “total counts” input. For full correctness under the i.i.d. model, one typically uses the joint likelihood above with $S$ and $n$ (kernel $\lambda^{S}e^{-n\lambda}$). The continuous Gamma–Poisson mode below follows the standard $n$-replicate derivation.

## B) Continuous conjugate prior: $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$

### Parameterisation

This app uses the **shape–rate** parameterisation:

$
\pi(\lambda)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda},
\qquad \lambda>0.
$

Moments:

$
\mathbb{E}[\lambda]=\frac{\alpha}{\beta},
\qquad
\mathrm{Var}(\lambda)=\frac{\alpha}{\beta^2}.
$

### Moment-matching: (mean, sd) $\to$ (shape, rate)

The Poisson tab asks for a prior mean $\mu_0$ and prior standard deviation $\sigma_0$. Solving

$
\mu_0=\frac{\alpha}{\beta},\qquad \sigma_0^2=\frac{\alpha}{\beta^2}
$

gives

$
\alpha = \frac{\mu_0^2}{\sigma_0^2},
\qquad
\beta = \frac{\mu_0}{\sigma_0^2}.
$

This is exactly the conversion used internally.

### Conjugacy proof

Let $S=\sum_{i=1}^n y_i$. Then

$
\pi(\lambda\mid y)
\propto p(y\mid\lambda)\pi(\lambda)
\propto \big[\lambda^{S}e^{-n\lambda}\big]\,\big[\lambda^{\alpha-1}e^{-\beta\lambda}\big]
= \lambda^{\alpha+S-1}e^{-(\beta+n)\lambda}.
$

Thus

$
\lambda\mid y \sim \mathrm{Gamma}(\alpha+S,\;\beta+n).
$

### Posterior mean and credible intervals

For $\lambda\mid y\sim\mathrm{Gamma}(\alpha',\beta')$ with $\alpha'=\alpha+S$, $\beta'=\beta+n$:

$
\mathbb{E}[\lambda\mid y]=\frac{\alpha'}{\beta'},
\qquad
\mathrm{Var}(\lambda\mid y)=\frac{\alpha'}{\beta'^2}.
$

Equal-tailed credible intervals are computed via Gamma quantiles:

$
\big[q_{\gamma/2},\;q_{1-\gamma/2}\big],
\qquad
q_p = F^{-1}_{\mathrm{Gamma}(\alpha',\beta')}(p).
$

### Posterior predictive distribution (Gamma–Poisson $\Rightarrow$ Negative Binomial)

For a new count $\tilde Y\mid\lambda\sim\mathrm{Poisson}(\lambda)$,

$
p(\tilde y\mid y)
= \int_0^\infty p(\tilde y\mid\lambda)\pi(\lambda\mid y)\,d\lambda
= \int_0^\infty \Big[e^{-\lambda}\frac{\lambda^{\tilde y}}{\tilde y!}\Big]
\Big[\frac{\beta'^{\alpha'}}{\Gamma(\alpha')}\lambda^{\alpha'-1}e^{-\beta'\lambda}\Big]\,d\lambda.
$

Collecting terms,

$
p(\tilde y\mid y)
= \frac{\beta'^{\alpha'}}{\tilde y!\,\Gamma(\alpha')}
\int_0^\infty \lambda^{\alpha'+\tilde y-1}e^{-(\beta'+1)\lambda}\,d\lambda
= \frac{\beta'^{\alpha'}}{\tilde y!\,\Gamma(\alpha')}
\frac{\Gamma(\alpha'+\tilde y)}{(\beta'+1)^{\alpha'+\tilde y}}.
$

This is a Negative Binomial form (with parameterisation depending on convention).

-----

# Additional Bayesian models (theory extensions beyond the current Shiny implementation)

This section intentionally goes beyond what is currently implemented server-side. The goal is to make the README a mathematically complete reference for canonical Bayesian updates.

## Bernoulli likelihood (special case of Binomial)

Let $X\mid\theta\sim\mathrm{Bernoulli}(\theta)$, $x\in\{0,1\}$. Then

$
p(x\mid\theta)=\theta^x(1-\theta)^{1-x}.
$

With $\theta\sim\mathrm{Beta}(\alpha,\beta)$, the posterior is

$
\theta\mid x\sim\mathrm{Beta}(\alpha+x,\beta+1-x).
$

For i.i.d. data $x_{1:n}$ with $S=\sum_i x_i$, we recover $\mathrm{Beta}(\alpha+S,\beta+n-S)$.

## Multinomial likelihood with Dirichlet prior

Let $x=(x_1,\dots,x_K)$ be counts with $\sum_k x_k = n$, and

$
x\mid\theta \sim \mathrm{Multinomial}(n,\theta),
\qquad \theta=(\theta_1,\dots,\theta_K),\ \sum_k\theta_k=1.
$

Likelihood:

$
p(x\mid\theta)=\frac{n!}{\prod_k x_k!}\prod_{k=1}^K \theta_k^{x_k}.
$

Dirichlet prior:

$
\pi(\theta)=\frac{1}{\mathrm{B}(\alpha)}\prod_{k=1}^K \theta_k^{\alpha_k-1},
\qquad \alpha_k>0,
$

where $\mathrm{B}(\alpha)=\frac{\prod_k\Gamma(\alpha_k)}{\Gamma(\sum_k\alpha_k)}$. Posterior:

$
\theta\mid x \sim \mathrm{Dirichlet}(\alpha_1+x_1,\dots,\alpha_K+x_K).
$

Posterior predictive for a single future draw is categorical with probabilities $\mathbb{E}[\theta_k\mid x]=(\alpha_k+x_k)/(\sum_j \alpha_j + n)$, and for future counts it is Dirichlet–Multinomial.

## Exponential likelihood with Gamma prior

Let $Y_i\mid\lambda\sim\mathrm{Exponential}(\lambda)$ with density $\lambda e^{-\lambda y_i}$, $y_i\ge 0$. For i.i.d. data, the likelihood kernel is

$
p(y_{1:n}\mid\lambda)\propto \lambda^{n}\exp\Big(-\lambda\sum_{i=1}^n y_i\Big).
$

With $\lambda\sim\mathrm{Gamma}(\alpha,\beta)$ (shape–rate),

$
\lambda\mid y \sim \mathrm{Gamma}\Big(\alpha+n,\ \beta+\sum_{i=1}^n y_i\Big).
$

## Negative Binomial likelihood with Gamma prior (overdispersion modelling)

One convenient parameterisation of the Negative Binomial is a Poisson–Gamma mixture:

$
Y\mid\lambda \sim \mathrm{Poisson}(\lambda),\qquad \lambda\mid r,p \sim \mathrm{Gamma}\Big(r,\ \frac{p}{1-p}\Big).
$

Marginally, $Y\sim\mathrm{NegBin}(r,p)$. This representation explains why Gamma priors are natural when modelling count overdispersion.

## Normal likelihood with known variance: posterior predictive

Under Normal–Normal conjugacy, the posterior predictive for $\tilde Y\mid y$ is Normal:

$
\tilde Y\mid y \sim \mathcal{N}(\mu_n,\ \sigma^2+\tau_n^2).
$

The predictive variance decomposes into observation noise $\sigma^2$ plus parameter uncertainty $\tau_n^2$.

## Multivariate Normal with known covariance

Let $y_i\mid\mu\sim\mathcal{N}_d(\mu,\Sigma)$ with $\Sigma$ known. With prior $\mu\sim\mathcal{N}_d(\mu_0,\Lambda_0)$, the posterior is

$
\mu\mid y \sim \mathcal{N}_d(\mu_n,\Lambda_n),
$

$
\Lambda_n^{-1}=\Lambda_0^{-1}+n\Sigma^{-1},
\qquad
\mu_n=\Lambda_n\big(\Lambda_0^{-1}\mu_0+n\Sigma^{-1}\bar y\big).
$

## Unknown covariance: Inverse-Wishart conjugacy (cautionary)

For multivariate Normal models, a classical conjugate prior for $\Sigma$ is the inverse-Wishart. While conjugate, it can be overly informative and is often replaced by modern alternatives (LKJ priors on correlations, half-t priors on scales). It is included here for completeness because it leads to analytic Gibbs updates in many textbook models.

## Hierarchical models: partial pooling and shrinkage

In hierarchical Bayesian models, we often have group-specific parameters $\theta_j$ with a shared hyperprior:

$
y_j\mid\theta_j \sim p(y_j\mid\theta_j),
\qquad
\theta_j\mid\eta \sim p(\theta_j\mid\eta),
\qquad
\eta\sim p(\eta).
$

This induces **partial pooling**: each $\theta_j$ is shrunk toward a population-level distribution governed by $\eta$. Many real applications (health, education, small-area estimation) benefit from this structure.

## Logistic regression and data augmentation (Polya–Gamma Gibbs sampling)

Binary regression is a canonical example where conjugacy is not straightforward.

Let $y_i\in\{0,1\}$ and

$
\Pr(y_i=1\mid\beta) = \sigma(x_i^\top\beta),
\qquad \sigma(t)=\frac{1}{1+e^{-t}}.
$

With a Gaussian prior $\beta\sim\mathcal{N}(0,\Sigma_0)$, the posterior is not conjugate.

A powerful augmentation identity (Polya–Gamma) introduces latent $\omega_i$ such that the conditional posterior becomes Gaussian:

$
p(\beta\mid\omega,y) \propto \exp\Big(-\frac{1}{2}\beta^\top(X^\top\Omega X+\Sigma_0^{-1})\beta + \beta^\top X^\top\kappa\Big),
$

where $\Omega=\mathrm{diag}(\omega_1,\dots,\omega_n)$ and $\kappa_i=y_i-1/2$. Therefore

$
\beta\mid\omega,y \sim \mathcal{N}(m,V),
\qquad V=(X^\top\Omega X+\Sigma_0^{-1})^{-1},\quad m=V X^\top\kappa.
$

The augmented Gibbs sampler alternates:

- $\omega_i\mid\beta,y\sim\mathrm{PG}(1,x_i^\top\beta)$
- $\beta\mid\omega,y\sim\mathcal{N}(m,V)$

This is a paradigmatic example of using augmentation to recover tractable full conditionals.

## Gaussian process regression (nonparametric Bayes)

Gaussian processes (GPs) place a prior on functions $f$:

$
f\sim\mathrm{GP}(m(\cdot),k(\cdot,\cdot)).
$

With regression model $y_i=f(x_i)+\epsilon_i$, $\epsilon_i\sim\mathcal{N}(0,\sigma^2)$, the posterior over $f$ is again a GP, and the posterior predictive at new $x_*$ is analytic:

$
f_*\mid y \sim \mathcal{N}\big(m_*,\,\Sigma_*\big),
$

where $m_*$ and $\Sigma_*$ depend on the kernel matrix $K(X,X)$ and cross-covariances $K(X,x_*)$. GPs illustrate that “Bayesian models” need not be finite-dimensional.

## Dirichlet process (DP) mixture models (Bayesian clustering)

For flexible density modelling and clustering, a DP prior is common:

$
G\sim\mathrm{DP}(\alpha,G_0),
\qquad
\theta_i\mid G\sim G,
\qquad
y_i\mid\theta_i\sim p(y_i\mid\theta_i).
$

Integrating out $G$ yields the Chinese Restaurant Process predictive structure, making DP mixtures a canonical example where Gibbs sampling (or collapsed Gibbs) is central.

-----

## Normal likelihood with known variance (Normal–Normal conjugacy)

Assume $Y_i\mid\mu\sim\mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known and $\bar y$ the sample mean. The likelihood kernel in $\mu$ is

$
p(y\mid\mu)\propto \exp\Big(-\frac{n}{2\sigma^2}(\mu-\bar y)^2\Big).
$

Choose prior $\mu\sim\mathcal{N}(\mu_0,\tau_0^2)$. Then the posterior is Normal:

$
\mu\mid y \sim \mathcal{N}(\mu_n,\tau_n^2),
$

where

$
\tau_n^2 = \Big(\frac{1}{\tau_0^2}+\frac{n}{\sigma^2}\Big)^{-1},
\qquad
\mu_n = \tau_n^2\Big(\frac{\mu_0}{\tau_0^2}+\frac{n\bar y}{\sigma^2}\Big).
$

This exhibits the precision-weighted averaging structure.

## Normal likelihood with unknown mean and variance (Normal–Inverse-Gamma)

Let $Y_i\mid\mu,\sigma^2\sim\mathcal{N}(\mu,\sigma^2)$. A standard conjugate prior is

$
\sigma^2 \sim \mathrm{Inv\text{-}Gamma}(a_0,b_0),
\qquad
\mu\mid\sigma^2 \sim \mathcal{N}\Big(\mu_0,\frac{\sigma^2}{\kappa_0}\Big).
$

Then the posterior remains Normal–Inverse-Gamma with updates (writing $\bar y$ and $S=\sum_i(y_i-\bar y)^2$):

$
\kappa_n = \kappa_0+n,
\qquad
\mu_n = \frac{\kappa_0\mu_0+n\bar y}{\kappa_0+n},
$

$
a_n = a_0+\frac{n}{2},
\qquad
b_n = b_0+\frac{1}{2}S+\frac{\kappa_0 n}{2(\kappa_0+n)}(\bar y-\mu_0)^2.
$

Marginally, $\mu\mid y$ is Student-$t$ and the posterior predictive for $\tilde y$ is also Student-$t$, demonstrating how uncertainty in $\sigma^2$ thickens tails.

## Bayesian linear regression (matrix form)

Let $y\in\mathbb{R}^n$, $X\in\mathbb{R}^{n\times p}$, model

$
y\mid\beta,\sigma^2 \sim \mathcal{N}(X\beta,\sigma^2 I_n).
$

With conjugate prior

$
\beta\mid\sigma^2 \sim \mathcal{N}(\beta_0,\sigma^2 V_0),
\qquad
\sigma^2\sim\mathrm{Inv\text{-}Gamma}(a_0,b_0),
$

the posterior is again Normal–Inverse-Gamma with

$
V_n^{-1}=V_0^{-1}+X^\top X,
\qquad
\beta_n = V_n\big(V_0^{-1}\beta_0 + X^\top y\big),
$

and

$
a_n=a_0+\frac{n}{2},
\qquad
b_n=b_0+\frac{1}{2}\Big(y^\top y+\beta_0^\top V_0^{-1}\beta_0-\beta_n^\top V_n^{-1}\beta_n\Big).
$

This connects Bayesian inference to ridge-like regularisation and provides analytic predictive distributions.

-----

# Computational Bayesian inference (MCMC) — expert-level overview

Analytic posteriors are the exception rather than the rule. Once the posterior has no closed form normalising constant or is high-dimensional (latent variables, hierarchical priors, non-conjugate likelihoods), we typically rely on **Monte Carlo** approximations.

## Monte Carlo estimators of posterior expectations

For an integrable function $f$, we often want

$
\mathbb{E}[f(\theta)\mid y]=\int f(\theta)\pi(\theta\mid y)\,d\theta.
$

If we can generate i.i.d. draws $\theta^{(s)}\sim\pi(\theta\mid y)$, then the Monte Carlo estimator

$
\hat\mu_f = \frac{1}{S}\sum_{s=1}^S f\big(\theta^{(s)}\big)
$

converges by the law of large numbers, and has standard error $\mathrm{sd}(f(\theta)\mid y)/\sqrt{S}$. However, i.i.d. sampling from $\pi(\theta\mid y)$ is usually impossible.

## Markov chain Monte Carlo: core idea

MCMC constructs a Markov chain $\{\theta^{(t)}\}_{t\ge 0}$ with:

- a transition kernel $K(\theta,\cdot)$, and
- stationary distribution equal to the desired posterior $\pi(\theta\mid y)$.

Under ergodicity conditions, the chain satisfies a Markov chain law of large numbers:

$
\frac{1}{T}\sum_{t=1}^T f(\theta^{(t)}) \xrightarrow[T\to\infty]{} \mathbb{E}[f(\theta)\mid y].
$

In practice, samples are **correlated**, so the effective information is less than $T$ independent draws.

### Geometric intuition: why mixing is hard

Posteriors can have:

- strong correlations (narrow ridges),
- heavy tails,
- multiple modes,
- funnel geometries (common in hierarchical models).

Poorly tuned random-walk samplers may explore these regions extremely slowly.

### Detailed balance (reversibility)

A sufficient (not necessary) condition to ensure $\pi$ is stationary is **detailed balance**:

$
\pi(\theta)K(\theta,\theta') = \pi(\theta')K(\theta',\theta)
\quad \text{for all } \theta,\theta'.
$

If detailed balance holds and the chain is irreducible and aperiodic, then the chain is ergodic with stationary distribution $\pi$.

### Stationarity vs convergence

Having $\pi$ as stationary means: if $\theta^{(0)}\sim\pi$, then $\theta^{(t)}\sim\pi$ for all $t$. Convergence is stronger: for arbitrary initial $\theta^{(0)}$, the distribution of $\theta^{(t)}$ approaches $\pi$ as $t\to\infty$. This requires irreducibility and aperiodicity (and, in continuous spaces, typically Harris recurrence).

### Markov chain CLT and Monte Carlo standard errors

Under suitable conditions, a Markov chain central limit theorem holds:

$
\sqrt{T}\big(\hat\mu_f-\mathbb{E}[f(\theta)\mid y]\big)\xrightarrow{d}\mathcal{N}(0,\sigma_f^2),
$

where $\sigma_f^2$ depends on the autocorrelation structure (not the i.i.d. variance). Practical MCSE estimation methods include batch means and spectral variance estimators.

## Metropolis–Hastings (MH)

### Algorithm

Given current state $\theta$, propose $\theta'\sim q(\theta'\mid\theta)$. Accept with probability

$
\alpha(\theta,\theta') = \min\Big\{1,\ \frac{\pi(\theta'\mid y)\,q(\theta\mid\theta')}{\pi(\theta\mid y)\,q(\theta'\mid\theta)}\Big\}.
$

If accepted set $\theta^{(t+1)}=\theta'$; otherwise $\theta^{(t+1)}=\theta$.

### Why the acceptance ratio works (sketch via detailed balance)

Let

$
K(\theta,\theta')=q(\theta'\mid\theta)\alpha(\theta,\theta')\quad (\theta'\neq\theta).
$

Then one can show

$
\pi(\theta)q(\theta'\mid\theta)\alpha(\theta,\theta') = \pi(\theta')q(\theta\mid\theta')\alpha(\theta',\theta),
$

which establishes detailed balance and hence $\pi$ as stationary.

### Symmetric proposals

If $q(\theta'\mid\theta)=q(\theta\mid\theta')$ (e.g. Gaussian random walk), MH reduces to

$
\alpha(\theta,\theta') = \min\Big\{1,\ \frac{\pi(\theta'\mid y)}{\pi(\theta\mid y)}\Big\}.
$

### Random-walk scaling and optimal acceptance (high-level)

In high dimensions for approximately Gaussian targets, classical results suggest an optimal acceptance rate around $0.234$ for random-walk MH, with step size scaling like $O(d^{-1/2})$. The practical implication: naive MH becomes inefficient as dimension grows, motivating gradient-based samplers.

### Adaptive MCMC (with caution)

Adaptive algorithms tune proposal parameters during sampling (e.g., covariance adaptation). Care is required: naive adaptation can break the Markov property and invalidate ergodicity unless diminishing adaptation conditions are satisfied.

## Gibbs sampling (a special case and a central workhorse)

### Setup and full conditionals

Suppose $\theta=(\theta_1,\dots,\theta_d)$. The Gibbs sampler requires the **full conditional distributions**

$
\pi(\theta_i\mid \theta_{-i}, y),\qquad i=1,\dots,d,
$

where $\theta_{-i}$ denotes all components except $\theta_i$.

### Algorithm (systematic scan)

Starting from $\theta^{(t)}$, update sequentially:

$
\theta_1^{(t+1)}\sim \pi(\theta_1\mid \theta_2^{(t)},\dots,\theta_d^{(t)},y),
$

$
\theta_2^{(t+1)}\sim \pi(\theta_2\mid \theta_1^{(t+1)},\theta_3^{(t)},\dots,\theta_d^{(t)},y),
$

and so on until

$
\theta_d^{(t+1)}\sim \pi(\theta_d\mid \theta_1^{(t+1)},\dots,\theta_{d-1}^{(t+1)},y).
$

### Why Gibbs targets the correct posterior

Each coordinate update is a Markov transition that leaves $\pi(\theta\mid y)$ invariant. Intuitively, if $\theta\sim\pi$, then sampling $\theta_i$ from its conditional keeps the joint distribution as $\pi$. More formally, the Gibbs kernel can be shown to satisfy invariance by integrating out the updated coordinate.

### Worked Gibbs example: Normal–Inverse-Gamma model

For the Normal model with unknown $(\mu,\sigma^2)$, the Normal–Inverse-Gamma posterior admits a simple two-step Gibbs sampler:

- Sample $\mu\mid\sigma^2,y$ from a Normal distribution.
- Sample $\sigma^2\mid\mu,y$ from an inverse-gamma distribution.

Concretely (with prior hyperparameters $\mu_0,\kappa_0,a_0,b_0$):

$
\mu\mid\sigma^2,y \sim \mathcal{N}\Big(\mu_n,\frac{\sigma^2}{\kappa_n}\Big),
\qquad
\kappa_n=\kappa_0+n,\ \mu_n=\frac{\kappa_0\mu_0+n\bar y}{\kappa_0+n},
$

and

$
\sigma^2\mid\mu,y \sim \mathrm{Inv\text{-}Gamma}\Big(a_0+\frac{n}{2},\ b_0+\frac{1}{2}\sum_{i=1}^n (y_i-\mu)^2+\frac{\kappa_0}{2}(\mu-\mu_0)^2\Big).
$

This illustrates the key Gibbs idea: even when the joint posterior is complex, *full conditionals can be standard distributions*.

### Worked Gibbs example: Bayesian linear regression (conditionally conjugate)

In Bayesian regression with $y\mid\beta,\sigma^2\sim\mathcal{N}(X\beta,\sigma^2 I)$ and conjugate Normal–Inverse-Gamma prior, a standard Gibbs sampler alternates:

- $\beta\mid\sigma^2,y\sim\mathcal{N}(\beta_n,\sigma^2 V_n)$
- $\sigma^2\mid\beta,y\sim\mathrm{Inv\text{-}Gamma}(a_n,b_n)$

where $\beta_n,V_n,a_n,b_n$ are as defined in the conjugate posterior formulas above.

### Gibbs as Metropolis–Hastings with acceptance probability 1

For a single-component update, consider a proposal that draws $\theta_i'$ from the full conditional $q(\theta_i'\mid \theta_{-i})=\pi(\theta_i'\mid\theta_{-i},y)$. In the MH ratio, the target and proposal terms cancel, yielding $\alpha\equiv 1$. This is why Gibbs can be viewed as MH with a cleverly chosen proposal.

### Blocked Gibbs, collapsed Gibbs, and data augmentation

- **Blocked Gibbs** updates groups $\theta_B$ jointly from $\pi(\theta_B\mid\theta_{-B},y)$, improving mixing when parameters are strongly correlated.
- **Collapsed Gibbs** analytically integrates out some variables to reduce dependence (Rao–Blackwellisation).
- **Data augmentation** introduces latent variables to make conditionals conjugate (e.g., mixture models), trading dimensionality for tractable updates.

### Rao–Blackwellisation (variance reduction)

If a conditional expectation is available analytically, replacing a raw Monte Carlo estimator with its conditional expectation can reduce variance. For example, instead of estimating $\mathbb{E}[f(\theta)]$ by $f(\theta^{(t)})$, one can use $\mathbb{E}[f(\theta)\mid \phi^{(t)}]$ when $\theta$ can be integrated out given $\phi$. Collapsed Gibbs samplers are often viewed through this lens.

## Convergence, mixing, and diagnostics

### Autocorrelation and effective sample size (ESS)

Because MCMC draws are correlated, uncertainty is governed by the integrated autocorrelation time

$
\tau_{\mathrm{int}} = 1 + 2\sum_{k=1}^{\infty}\rho_k,
$

where $\rho_k$ is lag-$k$ autocorrelation of $f(\theta^{(t)})$. The effective sample size is approximately

$
\mathrm{ESS} \approx \frac{T}{\tau_{\mathrm{int}}}.
$

### Burn-in and thinning (nuanced guidance)

- **Burn-in** discards early iterations before the chain reaches its stationary regime. It is a pragmatic mitigation for poor initialisation, not a substitute for diagnosing convergence.
- **Thinning** reduces storage by keeping every $k$-th draw, but usually does not increase information per unit compute; it is mainly a memory/IO strategy.

### Multiple chains and $\hat R$

Running multiple chains from dispersed initial points allows convergence assessment. The split-$\hat R$ statistic compares between-chain and within-chain variance; values near 1 suggest convergence (subject to the limitations of any scalar diagnostic).

### Trace plots, rank plots, and posterior geometry

Diagnostics are not optional. At minimum, inspect:

- Trace plots for non-stationarity and poor mixing.
- Autocorrelation functions (ACF) for slow exploration.
- Rank plots across chains for subtle non-convergence.

If the posterior has strong correlations or funnel geometry (common in hierarchical models), naive random-walk MH can fail catastrophically; reparameterisation (centred vs non-centred) or gradient-based methods are often required.

## Beyond basic MH/Gibbs: modern samplers

### Slice sampling

Slice sampling introduces an auxiliary variable $u$ and samples uniformly from the region under the unnormalised density. It can adapt step sizes automatically in 1D and some multivariate extensions, reducing tuning burden relative to MH.

### Hamiltonian Monte Carlo (HMC) and NUTS

HMC augments parameters $\theta$ with momenta $p$ and simulates Hamiltonian dynamics using gradients of $\log \pi(\theta\mid y)$. This allows long-distance proposals with high acceptance probability in high dimensions.

The No-U-Turn Sampler (NUTS) adaptively chooses trajectory lengths, dramatically improving usability (as in Stan).

These methods are not yet implemented in this Shiny app, but they are the modern standard for many continuous models.

### Monte Carlo standard error (MCSE)

The uncertainty in $\hat\mu_f$ should be reported via MCSE, which depends on ESS rather than raw iteration count.

## Bayesian workflow: posterior predictive checks and model criticism

Bayesian analysis is incomplete without checking whether the model can reproduce salient features of the data.

### Posterior predictive checks (PPC)

Generate replicated data $y^{\mathrm{rep}}$ from the posterior predictive:

$
y^{\mathrm{rep}}\sim p(\tilde y\mid y)=\int p(\tilde y\mid\theta)\,\pi(\theta\mid y)\,d\theta.
$

Compare test quantities $T(y)$ to $T(y^{\mathrm{rep}})$. Systematic discrepancies indicate model misfit.

### Predictive evaluation

For out-of-sample evaluation, one commonly uses approximate leave-one-out cross-validation (LOO) or information criteria such as WAIC, both built around the log pointwise predictive density.

## Alternatives and complements to MCMC

### Importance sampling

If $\theta^{(s)}\sim g(\theta)$ for a proposal density $g$, then

$
\mathbb{E}[f(\theta)\mid y] = \frac{\int f(\theta)\,\tilde\pi(\theta)\,d\theta}{\int \tilde\pi(\theta)\,d\theta}
\approx
\frac{\sum_s f(\theta^{(s)})w_s}{\sum_s w_s},
\qquad w_s=\frac{\tilde\pi(\theta^{(s)})}{g(\theta^{(s)})},
$

where $\tilde\pi$ is an unnormalised posterior. Importance sampling suffers in high dimensions unless $g$ closely matches the target.

### Sequential Monte Carlo (SMC)

SMC methods propagate a weighted particle approximation through a sequence of distributions (e.g., tempered posteriors), resampling to avoid weight degeneracy. SMC is particularly useful for multimodal targets and for estimating normalising constants $p(y)$.

### Variational inference (VI)

VI replaces sampling with optimisation by maximising the ELBO. It scales well but can underestimate uncertainty depending on the approximation family.

## Practical note for this repository

The app UI contains an “MCMC Simulations” tab with controls (iterations, burn-in, thinning, number of chains). The mathematical content above explains what those controls mean conceptually. The modular server-side sampler is not yet implemented in the current codebase, but the intended direction is consistent with standard MCMC workflows (potentially via JAGS, Stan, or custom samplers).

-----

# What the Shiny app currently matches (implementation notes)

## Implemented (server-side)

- **Binomial tab**
  - Discrete $\theta$ grid prior update.
  - Beta conjugate update with $\alpha' = \alpha + x$, $\beta' = \beta + n - x$.
  - Prior/posterior plotting and credible interval plotting.
- **Poisson tab**
  - Gamma conjugate update with $\alpha' = \alpha + S$, $\beta' = \beta + n$, where $\alpha,\beta$ are derived from (mean, sd).
  - Discrete grid prior update.
  - Prior/posterior plotting and a 95% interval plot.

## Present in UI but not yet fully implemented

- **Normal** tab
  - UI inputs exist for a Normal model, but the module server logic is currently incomplete.
- **MCMC Simulations** tab
  - UI scaffold exists (data upload, model selection, chain settings), but sampling logic is not yet wired into the modular app.

-----

# Reproducibility and running locally

## Dependencies

The project includes an `renv.lock` for package reproducibility.

## Run the application

From the repository root in R:

```r
renv::restore()
shiny::runApp()
```

-----

# Technical Stack

- **Language:** R
- **Framework:** Shiny
- **Core Bayesian/teaching utilities:** `TeachBayes` (tabulation/plot helpers)
- **Plotting:** `ggplot2`, `plotly`
- **Data manipulation:** `dplyr`, `tidyr`
- **UI theming:** `bslib`, `shinythemes`

-----

# Acknowledgments and provenance

## Fork / original author credit

This directory was forked/adapted from work originally authored by **Godwill Zulu** (see the earliest commits in the git history: `godwillA33peo <godwillzulu51@gmail.com>`). This repository retains and extends that original educational Shiny app idea and implementation.

## External references

- `TeachBayes` package and associated teaching materials (used for several plotting/tabulation utilities).

-----

# About Me

My name is **Donnie**, and I am a **PhD student at Trinity College Dublin**. My research interests include Bayesian modelling, computational statistics, and the translation of principled probabilistic methods into practical tools.

This project functions as a living lab notebook: a place where Bayesian identities are not only stated, but made tangible through interactive computation and visualisation.

-----

### Predictive evaluation

For out-of-sample evaluation, one commonly uses approximate leave-one-out cross-validation (LOO) or information criteria such as WAIC, both built around the log pointwise predictive density.

## Alternatives and complements to MCMC

### Importance sampling

If \(\theta^{(s)}\sim g(\theta)\) for a proposal density \(g\), then

\[
\mathbb{E}[f(\theta)\mid y] = \frac{\int f(\theta)\,\tilde\pi(\theta)\,d\theta}{\int \tilde\pi(\theta)\,d\theta}
\approx
\frac{\sum_s f(\theta^{(s)})w_s}{\sum_s w_s},
\qquad w_s=\frac{\tilde\pi(\theta^{(s)})}{g(\theta^{(s)})},
\]

where \(\tilde\pi\) is an unnormalised posterior. Importance sampling suffers in high dimensions unless \(g\) closely matches the target.

### Sequential Monte Carlo (SMC)

SMC methods propagate a weighted particle approximation through a sequence of distributions (e.g., tempered posteriors), resampling to avoid weight degeneracy. SMC is particularly useful for multimodal targets and for estimating normalising constants \(p(y)\).

### Variational inference (VI)

VI replaces sampling with optimisation by maximising the ELBO. It scales well but can underestimate uncertainty depending on the approximation family.

## Practical note for this repository

The app UI contains an “MCMC Simulations” tab with controls (iterations, burn-in, thinning, number of chains). The mathematical content above explains what those controls mean conceptually. The modular server-side sampler is not yet implemented in the current codebase, but the intended direction is consistent with standard MCMC workflows (potentially via JAGS, Stan, or custom samplers).

-----

# What the Shiny app currently matches (implementation notes)

## Implemented (server-side)

- **Binomial tab**
  - Discrete \(\theta\) grid prior update.
  - Beta conjugate update with \(\alpha' = \alpha + x\), \(\beta' = \beta + n - x\).
  - Prior/posterior plotting and credible interval plotting.
- **Poisson tab**
  - Gamma conjugate update with \(\alpha' = \alpha + S\), \(\beta' = \beta + n\), where \(\alpha,\beta\) are derived from (mean, sd).
  - Discrete grid prior update.
  - Prior/posterior plotting and a 95% interval plot.

## Present in UI but not yet fully implemented

- **Normal** tab
  - UI inputs exist for a Normal model, but the module server logic is currently incomplete.
- **MCMC Simulations** tab
  - UI scaffold exists (data upload, model selection, chain settings), but sampling logic is not yet wired into the modular app.

-----

# Reproducibility and running locally

## Dependencies

The project includes an `renv.lock` for package reproducibility.

## Run the application

From the repository root in R:

```r
renv::restore()
shiny::runApp()
```

-----

# Technical Stack

- **Language:** R
- **Framework:** Shiny
- **Core Bayesian/teaching utilities:** `TeachBayes` (tabulation/plot helpers)
- **Plotting:** `ggplot2`, `plotly`
- **Data manipulation:** `dplyr`, `tidyr`
- **UI theming:** `bslib`, `shinythemes`

-----

# Acknowledgments and provenance

## Fork / original author credit

This directory was forked/adapted from work originally authored by **Godwill Zulu** (see the earliest commits in the git history: `godwillA33peo <godwillzulu51@gmail.com>`). This repository retains and extends that original educational Shiny app idea and implementation.

## External references

- `TeachBayes` package and associated teaching materials (used for several plotting/tabulation utilities).

-----

# About Me

My name is **Donnie**, and I am a **PhD student at Trinity College Dublin**. My research interests include Bayesian modelling, computational statistics, and the translation of principled probabilistic methods into practical tools.

This project functions as a living lab notebook: a place where Bayesian identities are not only stated, but made tangible through interactive computation and visualisation.

-----