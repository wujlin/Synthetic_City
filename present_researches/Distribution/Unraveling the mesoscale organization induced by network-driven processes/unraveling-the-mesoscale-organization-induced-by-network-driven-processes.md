# Unraveling the mesoscale organization induced by network-driven processes

Giacomo Barzon $^{a,b}$ , Oriol Artime $^{c,d,e}$ , Samir Suweis $^{a,f,g}$ , and Manlio De Domenico $^{a,f,g,h,1}$

Affiliations are included on p. 10.

Edited by Giorgio Parisi, Università degli Studi di Roma La Sapienza, Roma, Italy; received October 23, 2023; accepted May 21, 2024

Complex systems are characterized by emergent patterns created by the nontrivial interplay between dynamical processes and the networks of interactions on which these processes unfold. Topological or dynamical descriptors alone are not enough to fully embrace this interplay in all its complexity, and many times one has to resort to dynamics-specific approaches that limit a comprehension of general principles. To address this challenge, we employ a metric—that we name Jacobian distance—which captures the spatiotemporal spreading of perturbations, enabling us to uncover the latent geometry inherent in network-driven processes. We compute the Jacobian distance for a broad set of nonlinear dynamical models on synthetic and real-world networks of high interest for applications from biological to ecological and social contexts. We show, analytically and computationally, that the process-driven latent geometry of a complex network is sensitive to both the specific features of the dynamics and the topological properties of the network. This translates into potential mismatches between the functional and the topological mesoscale organization, which we explain by means of the spectrum of the Jacobian matrix. Finally, we demonstrate that the Jacobian distance offers a clear advantage with respect to traditional methods when studying human brain networks. In particular, we show that it outperforms classical network communication models in explaining functional communities from structural data, therefore highlighting its potential in linking structure and function in the brain.

Jacobian distance | latent geometry | network-driven processes

The vast majority of systems that we observe in everyday life—whether natural, technological, or social—are made up of dynamic interactions among many constituents. Such a structural backbone is typically described as a complex network, where the nodes represent the constituents and the interaction or relationship between them is encoded in the links. These complex networks are characterized by a wide spectrum of geometric properties, like self-similarity (1, 2), that can be explained in terms of latent metric spaces (3, 4) (see ref. 5 for a review). However, the latent geometries that are induced by topological measures of distance, like shortest-path distance (6), do not take into account the actual dynamical process, which plays an essential role in determining the function of a network (7).

It is well known that the collective behavior of complex systems is shaped by the nontrivial interplay between the particular type of dynamical process that unfolds on the network and the specific map of interconnections (8, 9). Only recently, growing interest has been directed at the geometry induced by specific dynamical processes (10, 11) (12). Crucially, these works demonstrate that several hidden geometries can emerge from network-driven processes and, overall, that it is not trivial to characterize a complex system by inspecting only its structural units and subunits.

All these works focus on latent geometries induced by particular types of dynamics, or they are limited to canonical propagation dynamics like cascades, random walks, or diffusion dynamics (13). Thus, a general framework unveiling which details and properties of dynamical models drive the emergence of functional pathways, as well as how their latent geometry influences information propagation patterns on them, is still missing. Recently, this issue has been partially addressed in ref. 14, where a metric called universal temporal distance, was introduced and tested over a general class of nonlinear dynamical systems. This quantity, based on the response time of a time-invariant perturbation that spreads across the network, was used to discriminate between three highly distinctive dynamic regimes (universal propagation classes), which turn out to depend on the leading powers of the dynamics (i.e., the physics interaction) but not on the topological details of the network. They do not even depend on the

# Significance

Understanding how intricate dynamic patterns emerge from complex structural interactions is a formidable challenge. Often, the problem is simplified by focusing on either the structure or dynamics of these systems, or on stereotypical propagation models. Our approach diverges from this norm, by introducing a theory that uncovers the geometric aspect of dynamic processes within complex networks. By leveraging on a metric, the Jacobian distance, to trace perturbation propagation, we gauge the interplay between a system's structure and its functions, particularly concerning how dynamic processes interact with mesoscopic topological features. Crucially, our research offers valuable insights into the emergence and persistence of functional modules across multiple domains, from systems biology and ecology to epidemiology and engineering.

Author contributions: M.D.D. designed research; G.B., O.A., S.S., and M.D.D. performed research; G.B., O.A., S.S., and M.D.D. analyzed data; and G.B., O.A., S.S., and M.D.D. wrote the paper.

The authors declare no competing interest.

This article is a PNAS Direct Submission.

Copyright © 2024 the Author(s). Published by PNAS. This article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND).

To whom correspondence may be addressed. Email: manlio.dedomenico@unipd.it.

This article contains supporting information online at https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.2317608121/-/DCSupplemental.

Published July 5, 2024.

parameters of the dynamical model (i.e., the particular environmental conditions), as far as some requirements are satisfied, such as the avoidance of multiple steady states or phase transitions. These remarkable classes, however, were derived under the assumptions that the perturbation is time-independent, which might not be suitable for describing a variety of empirical settings, and that the network is structurally uncorrelated, hence failing to establish clear cause-effect relations among the dynamics-induced latent geometry and the complex hierarchical and modular mesoscopic structures of many real-world networks. At present, these important questions remain unanswered.

Here, we tackle this challenge by introducing a framework, named Jacobian geometry, which allows us to systematically translate the interplay between network topology and any dynamical process into a latent geometry (5). Specifically, we define a metric distance, the Jacobian distance, based on the comparison of the spatiotemporal evolution of instantaneous perturbations. This framework, which reduces to the diffusion geometry for simple diffusive processes such as random walks (10), allows us to perform a multiscale investigation on how dynamical processes shape the hierarchical relations between the units since nodes that have similar propagation patterns are also close with each other in their latent space (12). As a consequence of this, we unveil the highly nontrivial mesoscopic organization of network-driven processes whose linearized operator need not be in the family of Laplacian operators (10, 12), finding when and how functional modules cannot be mapped to structural ones.

To test our framework, we implement a pool of nonlinear models from diverse application fields, such as epidemic spreading, neuronal activation, ecological and gene regulation dynamics, among others. By investigating the various distances induced by these dynamics on synthetic networks, we find a zoo of different patterns. While in some cases the distance between two nodes correlates with simple topological indicators, in other cases the latent geometry is nontrivial, highlighting patterns that emerge from the intricate interaction between dynamics and topology. Moreover, we find that even for a given dynamical model, changes in the value of its parameters can lead to completely different geometries, hence breaking the dynamic classes determined by the leading powers of the dynamics.

To ensure that our framework can be useful in empirical contexts, we scrutinize its validity when applied to modular networks, which are ubiquitous in biological, ecological, social, and technological systems (15). Remarkably, our findings evince that topological communities are not always trivially mapped into latent communities. The trivial map happens only for the family of dynamics in which we observe a gap in the spectrum of the Jacobian matrix between nonlocalized community eigenvectors and localized bulk eigenvectors. Thus, the time-scale separation between fast intracommunity and slow intercommunity signaling processes is not a universal feature of modular networks but depends on the specific dynamics. We validate our framework on empirical networks, by comparing our process-driven modules to the ones obtained from popular algorithms for community detection. We find that process-driven clusters in the latent space might be different from the detected topological communities, in accordance with the results reported on synthetic modular networks with planted communities.

To demonstrate the relevance of our method, we provide a successful application in the context of neuroscience. In particular, we show that the process-driven communities derived from a structural brain network exhibit a stronger match with the canonical functional patterns compared to state-of-the-art

network communication models, thereby bridging the gap between brain structure and function.

# Results

Jacobian Geometry. Let us consider a networked dynamical system  $\dot{x}_k = f_k(x_1, \ldots, x_N) \equiv f_k(\mathbf{x})$ , where  $x_k(t)$  is a variable representing the state of node  $k = 1, \ldots, N$  at time  $t$ . The dependency on the network topology is in the vector fields  $\{f_k\}$  through the adjacency matrix, even though it is not explicitly indicated to lighten the notation. See Table 1 for some examples. The steady state  $\mathbf{x}^*$  of the system is given by  $f_k(\mathbf{x}^*) = 0 \forall k$ . It is important to emphasize that the specific value of  $\mathbf{x}^*$  depends on the functional dynamics, the assigned coupling values, and the network structure. If we slightly perturb the steady state of a given node, the effects of the perturbation will spread across the entire network following, in a nontrivial, inhomogeneous manner, all possible paths. It is reasonable to assume that nodes that easily share information have similar propagation patterns. Thus we can exploit the interaction between dynamics and topology to investigate the hidden geometric space induced by the spatiotemporal patterns of the perturbations.

To analytically substantiate this idea, we assume the steady state of a node  $i$  is slightly perturbed such that  $x_{i}^{*}\rightarrow x_{i}^{*} + \mathrm{d}x_{i}$  where  $|\mathrm{d}x_i|\ll |x_i^*|$  is the (instantaneous) intensity of the perturbation. Identifying the initial state of the system as  $\delta \mathbf{x}_{(i)}(0)\equiv (0,\dots,\mathrm{d}x_i,\dots,0)$ , where the subset  $(i)$  indicates that the state of the system is conditioned to an initial perturbation placed on node  $i$ , the time evolution of the perturbation on any node  $k$  follows, then,

$$
\begin{array}{l} \dot {x} _ {k} (t) \equiv \delta \dot {x} _ {k} (t) = f _ {k} (\mathbf {x} ^ {*} + \delta \mathbf {x} (t)) \\ = \sum_ {l = 1} ^ {N} \left. \frac {\partial f _ {k}}{\partial x _ {l}} \right| _ {\mathbf {x} ^ {*}} \delta x _ {l} (t) + \mathcal {O} (\delta \mathbf {x} ^ {2} (t)). \tag {1} \\ \end{array}
$$

In vectorial notation, we have that  $\delta \mathbf{x}_{(i)}(0) \equiv \mathrm{d}x_{i}\mathbf{e}_{i}$ , where  $\mathbf{e}_i$  is the unitary vector in the  $i$ -direction, and  $\delta \dot{\mathbf{x}}_{(i)}(t) \approx \mathrm{J}(\mathbf{x}^*)\delta \mathbf{x}_{(i)}(t)$ , where  $\mathrm{J}(\mathbf{x}^*)$  is the Jacobian matrix evaluated at the steady state, which in general depends both on the specific functional form of the vector fields  $\mathbf{f}$  and the topology. The general solution is given by  $\delta \mathbf{x}_{(i)}(t) = e^{\mathrm{J}(\mathbf{x}^*)t}\delta \mathbf{x}_{(i)}(0)$ . The Jacobian distance is then defined as the temporal evolution of the difference between two perturbations of intensity  $\mathrm{d}x_{i}$  and  $\mathrm{d}x_{j}$  initially placed in nodes  $i$  and  $j$  (Fig. 1A),

$$
\begin{array}{l} d _ {\tau} (i, j) \equiv \| \delta \mathbf {x} _ {(i)} (\tau) - \delta \mathbf {x} _ {(j)} (\tau) \| \\ = \left\| e ^ {\mathrm {J} \left(\mathbf {x} ^ {*}\right) \tau} \left[ \mathrm {d} x _ {i} \mathbf {e} _ {i} - \mathrm {d} x _ {j} \mathbf {e} _ {j} \right] \right\|. \tag {2} \\ \end{array}
$$

In Materials and Methods we report the demonstration that the Jacobian distance respects all the requirements for being a metric. Note that the Jacobian distance is a generalization of the diffusion distance; see SI Appendix for details.

It is instructive to understand the qualitative behavior of the Jacobian distance. It will be small when, between the two perturbed nodes, many paths connect them, thus allowing information to be easily exchanged. In other words, two nodes are close in their latent space if they are connected by multiple pathways that facilitate information exchange in a timescale  $\tau$ . For small timescales  $\tau$ , the perturbation will mainly affect the neighborhood of the initially perturbed nodes. For longer timescales, the influence of the perturbation impacts larger parts

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/15e0072adbddf278e1f012d5d4ec8d4373c6a7e43981f9d4eb0831361792f64a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/e4b7cc21ed7bba1bfd21abf32f76cb3ec5bc6e30c134a1612995835c60588c5f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/a36cf766e4c1d14898c9b322dc3613cc8c4e0fe9ef7d8ce75553804a6cd1523e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/fce88e9d52f4fd2d9d57ac393ed085d6a57c98fe8d8281e66b3f6a27f9c561c4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/5116f89f994bdcbcd344c4b53be16c7bc06aaeb00ab15e0f249991e533faaaa3.jpg)



Fig. 1. The Jacobian geometry offers insights into the latent geometry induced by a general network-driven process. (A) Given a networked dynamical system, the Jacobian distance between two nodes is defined as the temporal evolution of the difference between time-varying perturbations initially placed on these nodes. As a test bench, we implement several dynamic models reported in Table 1; NoisyVM stands for the noisy voter model dynamics. (B) Jacobian distance matrices correspond to different times  $(\tau = 1,5,20)$ , leading to different investigation scales. (C) To unravel the mesoscale organization, we average the distance matrices up to a temporal cutoff  $\tau_{\mathrm{max}} \approx N$ .


of the network, gradually reaching all nodes, while the system relaxes to the steady state, which, depending on the dynamics and on the intensity and location of the perturbations, can be either the unperturbed  $\mathbf{x}^*$  or a new one  $\mathbf{x}^{\prime *}$ . Thus, the parameter  $\tau$  acts as a multiresolution parameter at which we can build a distance matrix, whose elements are given by Eq. 2, induced by the dynamical process (Fig. 1B).

Since we are interested in unveiling the emergent patterns that are most persistent at the mesoscale, it is natural to average the distance matrices,

$$
\bar {d} (i, j) = \frac {1}{\tau_ {\max }} \sum_ {\tau = 1} ^ {\tau_ {\max }} d _ {\tau} (i, j), \tag {3}
$$

up to a certain cutoff that we fix  $\tau_{\mathrm{max}} \approx N$  (Fig. 1C). In this way, emergent mesoscale patterns, if any, are highlighted. In addition, to provide a fair comparison between different dynamics, we normalize the Jacobian by its smallest (i.e., largest in modulus) eigenvalue, such that the fastest timescale becomes the same for all dynamics. Additional details on the numerical implementation can be found in Materials and Methods.

Disentangling Network Dynamics and Topology. To investigate the geometric relationship between the network units in the latent space, we use the average Jacobian distance as a measure of dissimilarity for a cluster analysis. With that, we obtain a dendrogram of the hierarchical relationship between the nodes in the emerging latent geometry (Fig. 2  $A - D$ ) that can be used to infer the dynamics-induced mesoscale organization. To have a quantitative comparison between the different geometries, we compute the cophenetic correlation coefficient between the dendrograms (18) (Materials and Methods). A high correlation value is found when

the system units share similar effective relationships, meaning that the dynamics lead to similar propagation of the perturbation. On the other hand, a low correlation coefficient suggests differences in the emergent effective geometries.

In Fig. 2E, we display the correlation between the emergent geometries of different types of dynamics running on Erdős-Rényi networks (19). Remarkably, we observe an intricate, nontrivial relation between dynamical processes. Some dynamics behave very similarly among them and with all the others, such as Population and the Epidemics with  $R = 1$  ones, or the Mutualistic and Neuronal ones. On the other hand, there are some dynamics whose process-induced geometry is pretty unique,


Table 1. Dynamical equations of the models employed in this work to investigate the Jacobian geometry


<table><tr><td>Dynamics</td><td>∂τx_i =</td></tr><tr><td>Biochemical</td><td>F - Bx_i - R ∑j A_ij x_i x_j</td></tr><tr><td>Epidemics</td><td>-Bx_i + R ∑j A_ij (1 - x_i) x_j</td></tr><tr><td>Mutualistic</td><td>Bx_i (1 - x_i) + R ∑j A_ij x_i x_j^b/1+x_j^b</td></tr><tr><td>Neuronal</td><td>-Bx_i + C tanh x_i + R ∑j A_ij tanh x_j</td></tr><tr><td>Noisy voter model</td><td>A - Bx_i + C/k_i ∑j A_ij x_j</td></tr><tr><td>Population</td><td>-Bx_i^b + R ∑j A_ij x_j^a</td></tr><tr><td>Regulatory</td><td>-Bx_i^a + R ∑j A_ij x_j^h/1+x_j^h</td></tr><tr><td>Synchronization</td><td>ω_i + R ∑j A_ij sin(x_j - x_i)</td></tr></table>


$A_{ij}$  corresponds to the entries of the adjacency matrix, and they are 1 if nodes  $i$  and  $j$  are connected, and are 0 otherwise. Throughout the article, we consider connected, undirected, and unweighted networks. The relevant constants can be set according to the references (see refs. 8, 14, 16, and 17).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/2a1a047bb0e73f1e585265d4b03474c9e2ec1c40639b620ddb233539541d3975.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/d1d350ede42d1e737938c2da0e96e0ffadfe8a97ec9f5f624eee4d2e06d22235.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/fcea45559f35499561490b089628ed24665a50fddf4327902ce32809bcded8cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/ea08a7ff4fca792f34ccb6163a8f9e3b8937ed0d3e9bd54ad006d63a77bf56fb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/7f386698f3d1a3294c472723fbf2378bf0002e2817e8c432e6b55136d8e044f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/1211369e926ff205452980a3fa7a04a19ae00f60168f9395bb5eabf9366b4514.jpg)



Fig. 2. Comparison of Jacobian geometry emerging from topologically uncorrelated networks and distinct dynamical processes. (A-D) Examples of average distance matrices, computed from Eq. 3, and related dendrogram. (E) Cophenetic correlation coefficient (18). (F) In some cases, the average Jacobian distance between two nodes correlates with the sum of the inverse of the degrees, while in other cases it does not follow topological parameters. To visually compare different dynamics, we have normalized with respect to the largest value of the average distance. All dynamics have been simulated on Erdős-Rényi networks of size  $N = 128$  and mean degree  $\langle k \rangle = 10$ . The values of the parameters are fixed to unity unless otherwise stated.


showing no similarities at all with any of the rest, such as the Noisy Voter Model or the Epidemics with  $R = 0.05$ . For the remaining dynamics, we observe that they can display both high and low correlations among them, yet without any apparent criterion. In addition, we notice that the simple change of a model coefficient can lead to very different geometries, for instance, the exponent that governs the self-interaction in regulatory dynamics. Such exponent distinguishes between mortality ( $a = 1$ ) or pairwise annihilation ( $a = 2$ ). Similarly, the same effect can also be obtained by changing the parameters, like in the case of epidemics, neuronal or biochemical dynamics (SI Appendix, Fig. S1). The value of the parameters is specified by the actual environmental conditions surrounding the dynamical process, determining its steady state. For instance, the ratio between the transmission and the healing rate, named reproduction rate, specifies the phase of the epidemic. In the regime of high reproduction rate, the Jacobian is mostly governed by the diagonal terms, meaning that the perturbation relaxes locally, proportionally to the degree of the node (SI Appendix). This means that once almost all the nodes are completely infected, adding a fraction of the infected does not affect the neighboring nodes. Instead, for low values of the reproduction rate, the addition of new infected diffuses to the nearest nodes in a nontrivial manner governed by the specific topology. We also point out that such differences appear for operating points in the same region of the parameter space, i.e., above the epidemic threshold, and are not related to a qualitative change due to a phase transition. Nevertheless, in other cases, a modification in the parameters turns out to have no effect, as in the case of population dynamics (SI Appendix, Fig. S1). This happens because while the steady state is modified accordingly, the Jacobian remains invariant as we have shown

analytically (SI Appendix). Thus, both the changes in the physics of the process (i.e., the exponents) and the actual environmental conditions (i.e., the strength of interaction) can indeed influence the resultant distances, although this will crucially depend on the type of dynamics.

We close this section by shedding light on the qualitative relation between the Jacobian distance and local network properties. In some cases, we find that the Jacobian distance is simply predicted by the degree of the nodes (Fig. 2F). Interestingly, it corresponds to the thermodynamic limit of the resistance distance (20). The resistance distance is based on the hitting time, i.e., the time needed by a random walker to travel to a target node. For large enough graphs, it becomes independent of the initial conditions. Thus, in such cases (e.g., population dynamics, epidemics dynamics with high reproduction rate), the average Jacobian distance does not reflect any global properties of the network. Nevertheless, this correlation with the degrees does not hold for other types of dynamics (e.g., noisy voter model, epidemics with low reproduction rate), meaning they are sensitive to higher-order topological properties of the network. We find that these results are consistent on other types of networks, like small-world and scale-free (SI Appendix, Fig. S2).

Communities Explained by Jacobian Spectra. Empirical networks are typically nonrandom, displaying topological correlations of different sorts. A prominent feature is the presence of topological communities, that is, groups of nodes that are tightly connected between them and scarcely connected among each other (23). It is claimed that, in certain contexts, such modular structures facilitate information propagation (24), for instance, by producing timescale separations: fast and slow processes are

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/d1a9a3c5762595ac327b49806cb8a3d9626d3e13ef01c286556dd8b1b8f78ebd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/fc7420510704697634b6275ba94cc3506d8b862ec8309c037d6586eda9d65ee0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/4a78e161067117577c368c91baadb79c64a4b2111d2fba938a21fdaeec09eb8c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/e53994f0e4843b7d9ec819577a0304c1182bb2807e842c05ed9ab32fdf5a7a52.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/dbc9af0a0071aa3e03e2e7d64d99490a1a2c5e89965a66b857126cce25c6ffad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/f0b5a13f1dea1fc6dc568694ca4838125548a5f893464e32eaff879af622e8c0.jpg)



Fig. 3. Jacobian geometry on hierarchical modular networks. (A) Following refs. 21 and 22, we start with four blocks (Erdős-Rényi networks, size  $N = 32$ , average degree  $\langle k_0 \rangle = 20$ ) and we hierarchically connect pairs of blocks by randomly adding a fixed number of links (average number of links in the first level  $\langle k_1 \rangle = 4$ , the average number of links in the second level  $\langle k_2 \rangle = 1$ ). (B and C) For some dynamics, the average distance respects the communities and the hierarchy, while in other cases this is not true. (D) Again, we can explore the zoo of different geometries by computing the cophenetic correlation between the resulting dendrograms. (E) Sensitiveness to the topological communities can be explained by looking at the Jacobian spectra. A gap between consecutive eigenvalues, sorted from largest to smallest, suggests different timescales, while the participation ratio tells us about the localization of the eigenvectors.


differentiated, respectively, at the intramodular and intermodular scales (25). In light of this, it is crucial to understand how topological communities are embedded in the Jacobian geometry and whether the timescale separation and the Jacobian distance can be somehow linked to shed light on the functional properties of a network.

To address this issue, we first apply our framework on synthetic networks with planted communities, ordered in two hierarchical levels (21, 22) (Fig. 3A). The graphs we generate consist of four densely connected blocks representing the basic organizational level of the community. These modules are then connected in pairs in supermodules by establishing a fixed number of random connections between the elements of each module, which define the first organizational level of the network. The same procedure is repeated by connecting the newly formed blocks, forming the second hierarchical level. This type of network is chosen for illustrative purposes, and we show that the results reported in the following are robust in other typical networks with communities, such as those created with the Lancichinetti-Fortunato-Radicicchi benchmarking algorithm (26) (SI Appendix, Fig. S4). We find that, in some cases, nodes that are in the same topological community are also close in the latent space and the topological hierarchy is respected (e.g., Fig. 3B). Such results can be predicted by inspecting the spectrum of the Jacobian. Indeed, if we rank the eigenvalues from smallest to largest (in absolute value), the gaps between consecutive eigenvalues tell us about the relative differences of time scales (21). For those dynamics that are sensitive to the communities, we observe two jumps in the first part of the spectrum, in accord with the two planted hierarchical levels (Fig. 3E). By looking at the participation

ratio (27, 28) (Materials and Methods), we notice that the gap is between localized bulk eigenvectors and community eigenvectors with increasing delocalization. Thus, at the faster timescales, the perturbation is localized on a few nodes; thus, it spreads to the nodes belonging to the same topological community. At long timescales, the perturbation diffuses to other blocks following the topological hierarchy.

In other cases, we observe that the process-driven geometry is not influenced by the topological communities, e.g., in the case of biochemical dynamics (Fig. 3C). This happens because the delocalized community eigenvectors are associated with the smallest eigenvalues, i.e., they are the fastest to respond to the perturbation. Such an apparently counterintuitive behavior was also found in ref. 14 in the case of composite propagation.

Similarly to the case of uncorrelated networks, the sensitivity (or lack thereof) of the Jacobian geometry to the structural communities can be achieved by changing the coefficients and parameters of the dynamical models. In fact, we see that these changes induce a transition from localized to delocalized (or the other way around) eigenvectors. For instance, in the case of epidemic dynamics, in the regime of a high reproduction rate, the latent geometry is not modular. This is because all the eigenvectors are localized, thus the perturbation can easily spread to all the nodes regardless of the modular segregation. Instead, if the reproduction rate becomes smaller, the latent geometry turns modular, and we can observe the aforementioned timescale separation, where the perturbation remains local for longer times (SI Appendix, Fig. S3).

As a final note, we want to highlight the role played by the structural network topology in influencing the Jacobian

geometry. Our findings show that the (dis)similarities between different dynamics in uncorrelated networks are different from those observed in networks with communities. For instance, population dynamics and synchronization are now similar to the noisy voter model.

Process-Driven Geometries of Empirical Networks. So far all the topologies used have been synthetically created. We next evaluate the Jacobian geometry in three empirical networks from different domains. Real-world architectures frequently display a much more complex and richer community structure than networks with planted communities built from generative models. On top of that, other topological correlations might be also present. Therefore, it is instructive to verify to which extent the results obtained so far for synthetic networks are held when these new conditions are met. Furthermore, testing whether or not these results stand up is a necessary step to safely use the Jacobian framework to provide system-specific insights.

We consider the structure of the nervous system of the nematode Caenorhabditis elegans (C. elegans) (29), the transcriptional regulation networks that control gene expression in the bacterium Escherichia coli (E. coli) (30), and a social network of cooperation among university students (31). On them, we run, respectively, the neuronal, regulatory, and epidemic dynamics. For completeness, we also compute the corresponding diffusion distance on these networks, to show that it provides rather different results.

When dealing with empirical networks, we do not have a priori information on the topological communities to be compared with mesoscale clusters in the latent geometry. To find them, we apply the arguably two most used community detection algorithms: the Louvain (32) and Infomap (33) methods. Since such methods are greedy, each realization might generate different community assignations to the nodes. Thus, we run each algorithm 1,000 times, and we provide the final communities as consensus assignments (34). Since in empirical networks, one may not have clear information on which is the most representative level to cut the dendrogram constructed from the resultant Jacobian geometry, we compare the dynamics-induced clusters at every cut through the Adjusted Mutual Information (AMI) (35), an information-theoretic measure commonly used for evaluating the similarities between partitions (36). Briefly, the AMI quantifies the information shared by two different clusterings of the same elements, correcting by the baseline value of agreement solely due to chance.

In Fig. 4, we report the resultant geometries obtained from the connectome of the C. elegans. First, we observe that, as expected, the geometries induced by diffusion and by the particular dynamical process can be completely different. This can be appreciated at first sight by comparing the dendrograms of the average distance matrices (Fig. 4C). It implies that one should be cautious when drawing conclusions on the potential role of functional communities when using the diffusion distance (or any other dynamical model) since the emerging geometries are highly process-dependent. Regarding the comparison between dynamics-induced communities and topological communities, we observe, in general, low AMI values, suggesting that the communities obtained from the Jacobian geometry are different from those obtained from purely topological algorithms (Fig. 4B). In most cases, the score is higher with Infomap. This can be ascribed to the fact that Infomap uses random walkers to detect the network communities (33). Moreover, in most cases, the diffusion geometry leads to high scores. Instead, for the

neuronal dynamics, the AMI score decreases by increasing the ratio between the interaction with the nearest nodes and the self-interaction term. We perform this by changing the interaction strength  $R$ . For high values, the dynamics are mostly governed by the strong interactions between the nodes, leading to a "supercritical" regime of high activity, where the perturbations easily propagate and are integrated along the network. Therefore, in this regime, there are no emergent functional communities, as we can also notice from the flat dendrogram. On the other hand, for small values, the effect of the perturbation is mostly local, thus the system is in a "subcritical" regime characterized by segregated activity.

A similar picture can be deduced from the resultant geometries of the  $E$  coli network and the social network (Fig. 4 D-I). Thus, for high values of interaction strength, regulatory information, and epidemic spreading can easily move between communities without being hindered by boundaries. On the other hand, if the interaction strength is low, the introduction of new genes or infected individuals will be absorbed locally within the community before spreading to the rest of the network.

To further explore this, we quantitatively estimate the most relevant dynamical clusters emerging from the various Jacobian geometries through the Partition Stability Index (37) (see SI Appendix, Fig. S8 for its implementation). In particular, we extensively examined the epidemic dynamics on the social network across various reproduction rate values (SI Appendix, Fig. S9). Our findings reveal a significant dependency of the dynamical clusters on environmental parameters. This dependency manifests as a transition from a few clusters in the endemic phase to an almost uniform average Jacobian distance in the pandemic phase, resulting in a lack of discernible dynamical clusters. Importantly, we observe that these dynamical clusters differ from those obtained through purely diffusive dynamics, underscoring the importance of employing appropriate dynamical models.

Emergent Process-Driven Geometry of the Human Connectome Is Informative on Brain Functional Networks. To demonstrate the viability of our framework, we showcase its application in the analysis of emergent geometries in human connectomes. Over the past few decades, advancements in neuroimaging techniques, particularly fMRI, have revealed that even in the absence of explicit tasks or stimuli, the brain exhibits intrinsic patterns of synchronized activity across spatially distinct regions. These coherent patterns, known as resting state networks, have been consistently associated with specific cognitive functions and play a pivotal role as functional modules underlying various aspects of cognition (41, 42). This macroscale functioning of the brain is profoundly influenced by the structural backbone of white matter fibers (43). Such an intricate network, which also displays a modular architecture (24), enables the efficient transmission of signals and facilitates the integration of neural activity across various brain areas, thus shaping the information flow between distant regions of the brain. Despite significant correlations between the two modalities, a large part of the variance in functional connectivity is unexplained by direct structural connectivity (44, 45). In particular, there is a weak correspondence between structural and functional communities (46) and the mismatch is most evident for multimodal brain systems involved in high-order cognition (47).

In recent years, models of network communication have been shown to bridge this gap (see refs. 48 and 49 for comprehensive reviews). These models range from routing via efficient, selectively accessed paths to diffusive spreading along multiple

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/72811dd60094829bc83ef3ab98520537de875e2f3d7126a399b6cde642f56711.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/b9a365354967fb5faa4d510cf5bc6ba4bacb65dd8bb6b80467c843c62c5ddc25.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/710ac524d1631f3de006d890194c6c46ef7a638d72b8db3d81552261b7d36929.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/338906b7e10566134aa6ac504e4943d3dfd67f24f3a9df6bd9159495a3ae546d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/81eda57db395040a1a644d6310d556f3679027b0e418abfcce586863b7919689.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/893558dc50ad18ee9412cafffd9d05b96370a44d744764bd77c8205ef4fc8483.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/111cd615dee32a1cc077721219c67eff32a4f999d8bd2173043858bc38c003a4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/db3c1d2826a552bdbfe0b34b20665616dd9329fa6afed4072ed8aabdbc1d340c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/0654a41fe97b7a1dfd716f5652a3c4010c3546b904b5fb762d6b41798dea12e2.jpg)



Fig. 4. Jacobian geometry of empirical networks: the connectome of nematode C.elegans (29) (A); the regulation networks of bacterium E. coli (30) (B); social network of university students (31) (C). Each color represents a different community identified from the consensus of 1,000 realizations of the Louvain algorithm. Through the adjusted mutual information  $(D, E, \text{and} F)$ , we compare the different partitions of the dendrograms obtained from the Jacobian geometry with topological communities obtained with Louvain (dots) and Infomap (stars) algorithms.  $(G, H, \text{and} I)$  Examples of average Jacobian distance, and corresponding dendrogram.


network fronts, thus capturing communication between both structurally connected and unconnected regions. In particular, such models have been proven to unveil a modular network architecture that more closely resembles the brain's established functional systems (40). Here, we show that the Jacobian distance outperforms such communication models in predicting the canonical functional communities.

We apply our framework on high-resolution structural and functional connectivity data from a cohort of unrelated healthy adults from the Human Connectome Project (38), which are publicly available (additional details on subject inclusion and data preprocessing are provided in ref. 50). Structural networks are group-averaged (Fig. 5A) and nodes are parcellated according to a widely used functional template (39). Thus, each node is attributed to a specific resting state network, serving as a reference partition of the mesoscale functional organization (Fig. 5B).

Next, we construct the Jacobian geometry of the brain's structural network using the neural dynamics at some values of the coupling parameter. Additionally, we evaluate the diffusion geometry for the sake of comparison. Utilizing these geometries, we determine the dynamics-induced clusters at each cut of the corresponding dendrograms, as previously explained, and compare them to the reference partitions (SI Appendix).

Following ref. 40, we employ a similar multiresolution approach to partition the structural and communicability matrices

into communities. Specifically, we construct communities by aggregating consensus assignments from 1,000 runs of the Louvain algorithm across a wide range of resolution parameters (SI Appendix, Fig. S7). We focus our analysis on the communicability model since it has been validated as the most effective in reproducing the canonical communities, as evidenced by previous results (40, 47).

Fig. 5C displays the peak values of the AMI for the various models examined. We observe that the structural modules yield a maximum AMI of approximately 0.25, while incorporating communicability enhances the matching accuracy by a certain percentage, aligning with previous findings (40). These modules exhibit compact and contiguous spatial characteristics, primarily confined to a single hemisphere (Fig. 5D and E). Intriguingly, we find that the empirical communities derived from the neuronal-induced geometry provide a better fit to the canonical communities. Notably, this improvement is evident at an intermediate coupling parameter value, while the results deteriorated for higher or lower values. This outcome underscores the crucial role of dynamical parameters in shaping process-driven communities. Furthermore, our process-driven communities exhibit enhanced integration across both hemispheres (Fig. 5F). It should be mentioned that these findings could potentially be enhanced by conducting model selection to identify the most appropriate functional form of the dynamical process, as well as by performing


A


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/44ddbed8de71a8ea1f9d725faa641ea2645ffde7aeaee47353ddc152378f8b8e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/f4749e72581582f204c2179e8edfd0cf34c79bf644b75e6b47caeee74572fde7.jpg)



B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/9c0fbff8c3af62a67359557c195e19155dbb473f7c26fdcc87bee320164129a9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/49df02410b9b53982c752ee1623f454d261e6225426d4b2ae97ba051566f0a98.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/f3497b7b89888877b0629ac2d842ef39f3ae628360b40a73d18e061568566d09.jpg)



C


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/1fa1d6af6a2fcc63f587369ce61af0012182b61e01b18d2372f1324d44af1009.jpg)



D



Structural connectivity


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/56db561523ffc29802208a6cc826acb1fff72f1ab3e8996888addebf1cf203e8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/dfff8cf6e466c5395fcd933e45215b5815899ac796c9ad1ffe304d685768ed81.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/965061aa28b0ea06d89daec7ae6736cf7bad6608fa71197ced39a73162c89a3f.jpg)



E



Network communicability


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/88f55fae3e7190edec0a23a377f4968b6dcd177553566ebd67031eed6c5b35c4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/a9775746d98e44754fd5cc6f78ef14944e95aa37564a4f0c299dada46b9635dc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/023ad13196acce0c152ef3df1b3891a378492e69584c62dfd876d7d092e2af51.jpg)



F



Jacobian distance (Neuronal,  $R = 0.1$ )


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/1afc8b0d6803b525e471dbc06590e8c772aaf8551de589ae1878e86ab533e48c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/74bf9be5382db9fe5025c5165c398e608969649a5f502e14b12e9fbe0a3c654b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-02/2e1acc28-4154-405b-b3cb-a84122111362/2dddf68e8a6effb57d854b0c1a95430eb48e3a3484c4ed7f4b5609e5debbb070.jpg)



Fig. 5. Process-driven geometries of human connectomes. (A) We analyze a group-averaged structural network of 100 healthy subjects from the HCP dataset (38). (B) Each node is assigned to a canonical functional system following the well-established Schaefer parcellation (39). (C) We compute the process-driven modules obtained from the Jacobian distance with the neural dynamics  $(B = 1, C = 0.01$ , and  $R = \{0.5, 0.1, 0.05\})$ . The best partition is chosen as the peak of the Adjusted Mutual Information between the empirical and the process-driven geometries. As a comparison, we apply a multiresolution approach for finding the best partitions directly from the structural matrix and the communicability model (40). (D and E) These modules are spatially contiguous and primarily confined to a single hemisphere. (F) Instead, the Jacobian geometry predicts more integrated modules.


a more comprehensive parameter search. However, these aspects are beyond the scope of our current work.

# Discussion

The network of interconnections between the elements of an empirical system plays a crucial role in shaping the dynamic interactions within that system. In recent years, there has been a growing interest in predicting the dynamical patterns of empirical systems from their structural network (8, 10-12, 14). However, purely topological descriptors often fail to characterize complex dynamical propagation patterns. To bridge this gap, we present a framework where a latent space identified by a metric distance derives from structure and dynamics, simultaneously. We define a metric, the so-called Jacobian distance, that maps the similarity of propagation patterns into a measure of distance in such a process-induced geometric space. Our approach is general and can be applied to arbitrary network topologies and arbitrary dynamics under mild assumptions. The only requirement is that the considered dynamical model admits a stationary state that can be perturbed to map how information propagates between nodes and evolves over time.

Although various studies introduce metrics derived from the interplay between dynamics and structure, they often concentrate on a particular type of dynamical process (10, 11) or are constrained to canonical propagation dynamics like cascades, random walks, or diffusion dynamics (12, 13), thereby restricting the broader applicability of these approaches. To overcome this constraint, we introduce a pool of nonlinear dynamics that are commonly used to describe physical phenomena from different

application domains, for which we test the Jacobian distance on a set of synthetic and empirical benchmark networks. To be noted, the parameters in Table 1 were chosen to guarantee the system reaches a stable steady state. When otherwise not stated, they were fixed to 1. As expected, the Jacobian geometry highlights differences in the emergent relationships between nodes in the latent space induced by the dynamical processes. To quantitatively compare the induced geometries, we take a multiscale approach where the entire dendrograms of distance matrices are analyzed and compared.

Understanding Which Network Structural Properties Matter. In some cases, the average distance in the latent space trivially correlates with the sum of the inverse of the degree. For other dynamics, such a trivial relation between topological and latent descriptors does not exist, suggesting the emergence of new patterns induced by the dynamics. To enrich even further this picture, the latent geometries of dynamics that behave similarly on a specific network can behave differently on others, as well as their behavior can strongly depend on the parameters and coefficients of the dynamical equations. Indeed, our findings tell us that the particular form of the dynamics is exceedingly relevant. On the one hand, the functional form encodes the underlying mechanisms of the process, such as the physical laws or rules that govern the behavior of the system. For example, the functional form of a biological process may encode the biochemical reactions and interactions that take place within the system, while the functional form of a neural process may encode the electrical and chemical signaling pathways within the brain. Similarly, the functional form of a social process may

encode the social norms, rules, and interactions that govern the behavior of individuals within a group. On the other hand, dynamics are also specified by the particular parameters, which account for the typical interaction rates and the effect of external conditions.

We found that changes in these parameters have a profound influence on the interplay between the perturbation and the topology. In particular, we investigate the zoo of emergent geometries in the case of topologies with strong modular structures organized hierarchically. Interestingly, we found that topological communities are not always translated into the dynamic counterpart. The different behaviors can be predicted by inspecting the spectrum of the Jacobian matrix. Specifically, the presence of a gap between localized fast modes and nonlocalized slow modes indicates that there is a one-to-one relationship between hierarchical modular topological structures and temporal scale of the dynamics as already found in the case of synchronization patterns (21, 28). Instead, for some types of dynamics, the spectrum of the Jacobian displays no gap, and the eigenvectors are not localized, or the delocalized community modes are the fastest to respond, making the community boundaries transparent to the propagation of the perturbation. Summarizing, this provides solid evidence that the timescale separation induced by modularity is not a universal dynamical signature, as previously stated in some contexts, e.g., in ref. 25.

A possible explanation of these different behaviors was proposed in the work by Barzel and collaborators (14). Within their framework, they were able to classify analytically the different dynamical models into three regimes based on the propagation time of a time-invariant perturbation. In the so-called distance-limited regime, the perturbations follow the expected structural hierarchy, giving rise to the time-scale separation in the presence of modular structures. Instead, in the degree-limited regimes, low-degree nodes play the role of effective bottlenecks of signal propagation, while in the composite regime the community boundaries are effectively transparent for dynamic signal propagation. Despite the soundness of such universality and its crucial role in understanding the corresponding phenomena, such results are obtained in the limit of large uncorrelated networks, for nonfactorable dynamical models, and for time-independent perturbations. Yet, most empirical systems are finite and topologically correlated, characterized by loops and higher-order degree-degree correlations, and exposed to perturbations that may not be sustained in time and for complex dynamics that may not factorize. Therefore, the Jacobian geometry can be a powerful alternative tool for investigating the pattern of perturbations of a general dynamics occurring on a network, since it does not require any assumption on the type of network nor the specific parameters.

Moreover, our framework may help predict and design perturbation protocols. For instance, in the case of biochemical dynamics, the addition of new compounds will lead to rapidly diffused modifications in the local concentrations. Instead, in the case of epidemics, the addition or removal of new infected on top of a node in the supercritical regime will distribute uniformly over the network, thus structural modification on the network will not alter the patterns. Crucially, in this regime, the knowledge of the structural substrate does not provide valuable insights for formulating the most effective intervention strategies. Conversely, when we are near the epidemic threshold, the transmission of new infections occurs gradually, following the inherent structural hierarchy. In such a case, it is possible to

mitigate the spread by modifying the interaction network, such as by introducing additional layers in the hierarchy or by reducing the potential pathways between modules.

Process-Driven Geometries of Real-World Networks Link Structure and Function. We also demonstrate its applicability in linking emergent functional patterns to the underlying structural backbone in real-world scenarios. As proof of concept, we show that the process-driven clusters bridge the gap between the modular structural and functional communities in the human brain, delving into the still unresolved structure-function relationships (47). In particular, we show that our framework outperforms models of communication dynamics (40), which have garnered attention as potential generative models for brain functional connectivity (51, 52) and explaining its clinical alterations (53–55). These models assume various routing strategies to guide polysynaptic signal propagation along the structural connectome. However, they rely on a complete knowledge of the whole network structure by each local node, which is unrealistic, or they are limited to a diffusive dynamics. Instead, within our framework, the signals explore all possible paths in the network guided by a specific dynamical process. Such communication models have also been demonstrated to predict the pattern of propagation of electrical stimulation (56). Therefore, our Jacobian distance framework seems a promising avenue for the prediction of propagation patterns in brain-oriented problems and the design of noninvasive stimulation protocols (57, 58). It is important to note that our framework, as well as the communication models, does not take into account regional heterogeneity, such as transcriptomic, cytoarchitectonic, and neuromodulatory information (47), which may explain the missing gap between the experimental and the process-driven functional modules.

Range of Validity of the Jacobian Distance Framework. The theoretical development of the Jacobian distance is based on several assumptions that can forestall a complete understanding of the complex propagation patterns of the perturbations.

For instance, the computation of the Jacobian distance involves the linearization of general nonlinear systems of equations in the vicinity of their steady state. The absence of such states is not a limitation, though. Indeed, we do not perforce need to work with systems that display equilibrium or stationary states. This will occur in singular points (or regions) in the parameter space, for instance, in the noisy voter model dynamics with vanishing noise. Our Jacobian framework can be adapted to deal with these cases if we approximate the nonlinear process through a linearization around a (stable) trajectory (59). Consequently, the propagator becomes time-dependent, requiring in principle the full expression of the trajectory for integration of the linearized differential equation. Yet, the analytical expressions of the trajectories, in general, do not exist. To overcome this obstacle, one potential solution is to introduce an effective average (60) or employ a dynamic mean-field analysis (61). These approaches are rather convoluted and it is still not clear whether they can be systematically addressed to obtain general, dynamics-independent results. Such analyses fall outside the scope of the current work and are left for future studies.

Another of these cases concerns complex systems that display coexisting fixed points. A paradigmatic example is related to the regulatory dynamics, under the condition  $a < h$  (14). In this multistable scenario, a strong enough perturbation can trigger an abrupt and possibly irreversible shift to the basin of

stability of an alternative equilibrium, thus inducing dramatic qualitative changes in the system's behavior (62, 63) (SI Appendix, Fig. S5). Another example is related to dynamical models that undergo phase transitions. In the vicinity of a critical point, emergent phenomena and system-wide correlations arise, and the linear approximation the Jacobian distance is based on may not be sufficient to capture the model's dynamics in its full complexity (64). Some of these phenomena include the emergence of sustained oscillations (65), or extreme events, such as avalanches (66). In fact, the divergence of the system susceptibility can lead to nonlinear and sustained propagation of perturbations, accompanied by the occurrence of critical slowing down, where the system's dynamics become increasingly sluggish (67). We exemplify with the epidemic dynamics that it is indeed close to the phase transition where the theoretical predictions for the Jacobian distance show the largest disagreement with the actual values computed from the simulations (SI Appendix, Fig. S6). It remains a theoretical challenge to explore in the future the systematization of all these cases.

# Conclusions

We conclude by highlighting that the Jacobian distance can extend several applications based on the diffusion distance to more general network-driven processes, such as dimensionality reduction (68), functional (10) and multiscale (69) community detection, the estimation of local and global network dimension (70), as well as node centrality (71). Moreover, in this work, we focus on undirected and unsigned networks, but the framework can be suitably extended to work on a wider range of structures, e.g., multilayer (72).

# Materials and Methods

Jacobian Distance Is a Metric Distance. We show in the following that the geometry of the network-driven process is a bona fide metric space. Indeed, if we assume that  $\tau$  is small enough not to be in equilibrium, the Jacobian distance Eq. 2 is a metric, since it satisfies,  $\forall \tau$ , the following properties:

1.  $d_{\tau}(i,i) = 0,$

2. Positivity:  $d_{\tau}(i,j) > 0 \forall i \neq j$ ,

3. Symmetry:  $d_{\tau}(i,j) = d_{\tau}(j,i)$ ,

4. Triangle inequality:  $d_{\tau}(i,j) \leq d_{\tau}(i,k) + d_{\tau}(k,j)$ .

Properties 1 and 3 are trivially satisfied by the definition of the Jacobian distance Eq. 2. Property 2 is satisfied if rows of the matrix  $e^{\mathrm{J}(\mathbf{x}^{*})\tau}$  are not equal, i.e.,  $\operatorname{det}(e^{\mathrm{J}(\mathbf{x}^{*})\tau}) \neq 0$ . Finally, Property 4 follows from the triangular inequality of the norm, since the Jacobian distance is the Euclidean distance in  $\mathbb{R}^N$  between the row vectors of the Jacobian matrix.

Dendrogram Comparison. To unveil the persistent mesoscale structures of the network, we averaged the Jacobian distance matrices up to a time cutoff, that we choose to be the size of the system,  $\tau_{\mathrm{max}} \approx N(10)$ . From the average matrix, we build a hierarchical clustering using the average group clustering method (73) to investigate the hierarchy of interactions between the units in the space induced by the perturbations.

To compare the hierarchical partitions induced by two dynamics on the same topology, we evaluate the cophenetic correlation coefficient between the two related dendrograms (18). First, from each dendrogram, we compute the cophenetic distance, which translates the original Jacobian distance between nodes  $i$  and  $j$  into the distance between the two clusters the nodes belong to. Then, we compute the correlation coefficient between the cophenetic distance of all pairs of nodes. In this way, nodes that are grouped in a similar way will lead to a high correlation.

Perturbation Analysis and Localization. The time-evolution of the perturbation initially placed on node  $k$ ,  $\delta \mathbf{x}_{(k)}(0) = \mathrm{d}x_{k}\mathbf{e}_{(k)}$ , can be written in terms of the eigenvectors and eigenvalues of the Jacobian

$$
\delta \mathbf {x} _ {(k)} (t) = \mathrm {d} x _ {(k)} \sum_ {\alpha} \frac {\mathbf {e} _ {(k)} \cdot \mathbf {u} _ {\alpha}}{\mathbf {u} _ {\alpha} \cdot \mathbf {v} _ {\alpha}} e ^ {\lambda_ {\alpha} t} \mathbf {v} _ {\alpha}, \tag {4}
$$

where  $\mathbf{u}_{\alpha}, \mathbf{v}_{\alpha}$ , and  $\lambda_{\alpha}$  are respectively the left, the right eigenvectors, and the corresponding eigenvalues. Thus, the perturbation spread along each mode  $\alpha$  with amplitude  $A_{\alpha}^{(k)} = \mathrm{d}x_{(k)}(\mathbf{e}_{(k)} \cdot \mathbf{u}_{\alpha}) / (\mathbf{u}_{\alpha} \cdot \mathbf{v}_{\alpha})$ , while the (inverse of the) eigenvalue controls the timescale of the relaxation toward the equilibrium. We note that to obtain a stable steady state, all the eigenvalues have to be nonpositive.

To have an indication of which nodes each mode is localized, we computed the participation ratio (27, 28)

$$
P R _ {\alpha} = \frac {\left(\sum_ {i} v _ {i , \alpha} u _ {i , \alpha}\right) ^ {2}}{\sum_ {i} \left(v _ {i , \alpha} u _ {i , \alpha}\right) ^ {2}}, \tag {5}
$$

that indicates the number of nodes on which an eigenvector is significantly different from zero. A large participation ratio  $(PR \approx N)$  indicates that the mode is mostly delocalized on all the nodes of the network, while a small participation ratio  $(PR \ll N)$  means that the propagation of the perturbation along the mode will affect only a small fraction of nodes, i.e., it is localized.

Numerical Simulations. We test our framework over different dynamics; see Table 1 for details. To evaluate the Jacobian matrix, we have to compute numerically the equilibrium. To do that, we integrate the equation with Runge-Kutta 45 starting from an arbitrary initial condition and let the system reach the steady state. To numerically realize this limit, we consider the termination condition

$$
\max  _ {i} \left| \frac {x _ {i} \left(t _ {n}\right) - x _ {i} \left(t _ {n - 1}\right)}{x _ {i} \left(t _ {n}\right) \Delta t} \right| <   \epsilon , \tag {6}
$$

where we set a tolerance of  $\epsilon = 10^{-6}$

Data, Materials, and Software Availability. The datasets used in this work are publicly available. Python code for numerical simulations has been deposited on GitHub (https://github.com/gbarzon/jacobian.Geometry) and it will be made available before the time of publication. Previously published data were used for this work (29-31, 38).

ACKNOWLEDGMENTS. O.A. acknowledges financial support from the Spanish Ministry of Universities through the Recovery, Transformation and Resilience Plan funded by the European Union (Next Generation EU), and the University of the Balearic Island and from the Spanish grant PID2021-128005NB-C22, funded by Ministerio de Ciencia e Innovacion (MCIN)/Agencia Estatal de Investigacion (AEI) MCIN/AEI/10.13039/501100011033. M.D.D. and S.S. acknowledge Istituto Nazionale di Fisica Nucleare (INFN) for Learning Complex Networks (LINCOLN) grant. M.D.D. acknowledges partial financial support from the Human Frontier Science Program Organization (Human Frontier Science Program (HFSP) Ref. RGY0064/2022), from Ministero dell'università e della ricerca (MUR) funding within the Fondo Italiano per la Scienza (FIS) (DD n. 1219 31-07-2023) Project no. FIS00000158 and from the EU funding within the MUR PNRR "National Center for High-Performance Computing, BIG DATA AND QUANTUM COMPUTING" (Project no. CN00000013 CN1).



1. C. Song, S. Havlin, H. A. Makse, Self-similarity of complex networks. Nature 433, 392-395 (2005).





2. C. Song, S. Havlin, H. A. Makse, Origins of fractality in the growth of complex networks. Nat. Phys. 2, 275-281 (2006).





3. M. Á. Serrano, D. Krioukov, M. Boguná, Self-similarity of complex networks and hidden metric spaces. Phys. Rev. Lett. 100, 078701 (2008).





4. D. Krioukov, F. Papadopoulos, M. Kitsak, A. Vahdat, M. Boguná, Hyperbolic geometry of complex networks. Phys. Rev. E 82, 036106 (2010).





5. M. Boguna et al., Network geometry. Nat. Rev. Phys. 3, 114-135 (2021).





6. F. Harary, Graph Theory (Addison-Wesley, 1994).





7. M. A. Porter, J. P. Gleeson, "Dynamical systems on networks" in Frontiers in Applied Dynamical Systems: Reviews and Tutorials 4 (Springer, 2016), p. 29.





8. B. Barzel, A. L. Barabási, Universality in network dynamics. Nat. Phys. 9, 673-681 (2013).





9. C. Meena et al., Emergent stability in complex network dynamics. Nat. Phys., 1-10 (2023).





10. M. De Domenico, Diffusion geometry unravels the emergence of functional clusters in collective phenomena. Phys. Rev. Lett. 118, 168301 (2017).





11. D. Brockmann, D. Helbing, The hidden geometry of complex, network-driven contagion phenomena. Science 342, 1337-1342 (2013).





12. M. T. Schaub, J. C. Delvenne, R. Lambiotte, M. Barahona, Multiscale dynamical embeddings of complex networks. Phys. Rev. E 99, 062308 (2019).





13. G. Zamora-L'pez, M. Gilson, An integrative dynamical perspective for graph theory and the analysis of complex networks. Chaos Interdiscip. J. Nonlinear Sci. 34, 041501 (2024).





14. C. Hens, U. Harush, S. Haber, R. Cohen, B. Barzel, Spatiotemporal signal propagation in complex networks. Nat. Phys. 15, 403-412 (2019).





15. M. E. Newman, Communities, modules and large-scale structure in networks. Nat. Phys. 8, 25-31 (2012).





16. U. Harush, B. Barzel, Dynamic patterns of information flow in complex networks. Nat. Commun. 8, 1-11 (2017).





17. A. Carro, R. Toral, M. San Miguel, The noisy voter model on complex networks. Sci. Rep. 6, 1-14 (2016).





18. R. R. Sokal, F. J. Rohlf, The comparison of dendrograms by objective methods. Taxon 11, 33-40 (1962).





19. P. Erdős et al., On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci 5, 17-60 (1960).





20. U. Von Luxburg, A. Radl, M. Hein, Getting lost in space: Large sample analysis of the commute distance. Adv. Neural Inf. Proces. Syst. 23, 2622-2630 (2010).





21. A. Arenas, A. Diaz-Guilera, C. J. Pérez-Vicente, Synchronization reveals topological scales in complex networks. Phys. Rev. Lett. 96, 114102 (2006).





22. G. Zamora-López, Y. Chen, G. Deco, M. L. Kringelbach, C. Zhou, Functional complexity emerging from anatomical constraints in the brain: The significance of network modularity and rich-clubs. Sci. Rep. 6, 1-18 (2016).





23. S. Fortunato, Community detection in graphs. Phys. Rep. 486, 75-174 (2010).





24. D. Meunier, R. Lambiotte, E. T. Bullmore, Modular and hierarchically modular organization of brain networks. Front. Neurosci. 4, 200 (2010).





25. R. K. Pan, S. Sinha, Modularity produces small-world networks with dynamical time-scale separation. Europhys. Lett. 85, 68006 (2009).





26. A. Lancichinetti, S. Fortunato, F. Radicchi, Benchmark graphs for testing community detection algorithms. Phys. Rev. E 78, 046110 (2008).





27. S. Suweis, J. Grilli, J. R. Banavar, S. Allesina, A. Maritan, Effect of localization on the stability of mutualistic ecological networks. Nat. Commun. 6, 1-7 (2015).





28. A. P. Millán, J. J. Torres, G. Bianconi, Complex network geometry and frustrated synchronization. Sci. Rep. 8, 1-10 (2018).





29. S. J. Cook et al., Whole-animal connectomes of both Caenorhabditis elegans sexes. Nature 571, 63-71 (2019).





30. S. S. Shen-Orr, R. Milo, S. Mangan, U. Alon, Network motifs in the transcriptional regulation network of Escherichia coli. Nat. Genet. 31, 64-68 (2002).





31. M. Fire, G. Katz, Y. Elovici, B. Shapira, L. Rokach, "Predicting student exam's scores by analyzing social network data" in Active Media Technology: 8th International Conference, R. Huang, et al., Eds. (Springer, Berlin, Heidelberg, 2012), pp. 584-595.





32. V. D. Blondel, J. L. Guillaume, R. Lambiotte, E. Lefebvre, Fast unfolding of communities in large networks. J. Stat. Mech. Theory Exp. 2008, P10008 (2008).





33. M. Rosvall, C. T. Bergstrom, Maps of random walks on complex networks reveal community structure. Proc. Natl. Acad. Sci. U.S.A. 105, 1118-1123 (2008).





34. D. S. Bassett et al., Robust detection of dynamic community structure in networks. Chaos Interdiscip. J. Nonlinear Sci. 23, 013142 (2013).





35. N. X. Vinh, J. Epps, J. Bailey, "Information theoretic measures for clusterings comparison: is a correction for chance necessary?" in Proceedings of the 26th Annual International Conference on Machine Learning (JMLR.org, 2009), pp. 1073-1080.





36. S. Fortunato, M. E. Newman, 20 years of network community detection. Nat. Phys. 18, 848-850 (2022).





37. P. Villegas, A. Gabrielli, A. Poggialini, T. Gili, Multi-scale laplacian community detection in heterogeneous networks. arXiv Preprint (2023). http://arxiv.org/abs/2301.04514 (Accessed 9 January 2024).





38. D. C. Van Essen et al., The human connectome project: A data acquisition perspective. Neuroimage 62, 2222-2231 (2012).





39. A. Schaefer et al., Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cereb. Cortex 28, 3095-3114 (2018).





40. C. Seguin et al., Network communication models narrow the gap between the modular organization of structural and functional brain networks. Neuroimage 257, 119323 (2022).





41. M. D. Fox et al., The human brain is intrinsically organized into dynamic, anticorrelated functional networks. Proc. Natl. Acad. Sci. U.S.A. 102, 9673-9678 (2005).





42. J. S. Damoiseaux et al., Consistent resting-state networks across healthy subjects. Proc. Natl. Acad. Sci. U.S.A. 103, 13848-13853 (2006).





43. C.J. Honey, R. Kotter, M. Breakspear, O. Sporns, Network structure of cerebral cortex shapes functional connectivity on multiple time scales. Proc. Natl. Acad. Sci. U.S.A. 104, 10240-10245 (2007).





44. C. J. Honey et al., Predicting human resting-state functional connectivity from structural connectivity. Proc. Natl. Acad. Sci. U.S.A. 106, 2035-2040 (2009).





45. H. J. Park, K. Friston, Structural and functional brain networks: From connections to cognition. Science 342, 1238411 (2013).





46. B. Mišić et al., Network-level structure-function relationships in human neocortex. Cereb. Cortex 26, 3285-3296 (2016).





47. L. E. Suárez, R. D. Markello, R. F. Betzel, B. Misco, Linking structure and function in macroscale brain networks. Trend Cogn. Sci. 24, 302-315 (2020).





48. A. Avena-Koenigsberger, B. Misic, O. Sporns, Communication dynamics in complex brain networks. Nat. Rev. Neurosci. 19, 17-33 (2018).





49. C. Seguin, O. Sporns, A. Zalesky, Brain network communication: Concepts, models and applications. Nat. Rev. Neurosci. 24, 1-18 (2023).





50. S. Larivière et al., The enigma toolbox: Multiscale neural contextualization of multisite neuroimaging datasets. Nat. Methods 18, 698-700 (2021).





51. S. B. Laughlin, T. J. Sejnowski, Communication in neuronal networks. Science 301, 1870-1874 (2003).





52. J. Goni et al., Resting-brain functional connectivity predicted by analytic measures of network communication. Proc. Natl. Acad. Sci. U.S.A. 111, 833-838 (2014).





53. A. Raj, A. Kuceyeski, M. Weiner, A network diffusion model of disease progression in dementia. Neuron 73, 1204-1215 (2012).





54. S. C. de Lange et al., Shared vulnerability for connectome alterations across psychiatric and neurological brain disorders. Nat. Hum. Behav. 3, 988-998 (2019).





55. E. Lella, E. Estrada, Communicability distance reveals hidden patterns of Alzheimer's disease. Netw. Neurosci. 4, 1007-1029 (2020).





56. C. Seguin et al., Communication dynamics in the human connectome shape the cortex-wide propagation of direct electrical stimulation. Neuron 111, 1391-1401 (2023).





57. E. Dayan, N. Censor, E. R. Buch, M. Sandrini, L. G. Cohen, Noninvasive brain stimulation: From physiology to network dynamics and back. Nat. Neurosci. 16, 838-844 (2013).





58. D. Momi et al., Network-level macroscale structural connectivity predicts propagation of transcranial magnetic stimulation. Neuroimage 229, 117698 (2021).





59. S. Suweis, J. A. Carr, A. Maritan, A. Rinaldo, P. D'Odorico, Resilience and reactivity of global food security. Proc. Natl. Acad. Sci. U.S.A. 112, 6902-6907 (2015).





60. C. Tu, P. D'Odorico, S. Suweis, Dimensionality reduction of complex dynamical systems. iScience 24, 101912 (2021).





61. T. Galla, Dynamically evolved community size and stability of random Lotka-Volterra ecosystems (a). Europhys. Lett. 123, 48004 (2018).





62. P.J. Menck, J. Heitzig, N. Marwan, J. Kurths, How basin stability complements the linear-stability paradigm. Nat. Phys. 9, 89-92 (2013).





63. V. Dakos et al., Ecosystem tipping points in an evolving world. Nat. Ecol. Evol. 3, 355-362 (2019).





64. F. Peruzzo, M. Mobilia, S. Azaele, Spatial patterns emerging from a stochastic process near criticality. Phys. Rev. X 10, 011032 (2020).





65. G. Barzon, G. Nicoletti, B. Mariani, M. Formentin, S. Suweis, Criticality and network structure drive emergent oscillations in a stochastic whole-brain model. J. Phys. Complex 3, 025010 (2022).





66. G. Nicoletti, L. Saravia, F. Momo, A. Maritan, S. Suweis, The emergence of scale-free fires in Australia. iScience 26, 106181 (2023).





67. L. Dai, D. Vorselen, K. S. Korolev, J. Gore, Generic indicators for loss of resilience before a tipping point leading to population collapse. Science 336, 1175-1177 (2012).





68. R. R. Coifman, S. Lafon, Diffusion maps. Appl. Comput. Harmon. Anal. 21, 5-30 (2006)





69. R. Lambiotte, "Multi-scale modularity in complex networks" in 8th International Symposium on Modeling and Optimization in Mobile, ad hoc, and Wireless Networks (IEEE, 2010), pp. 546-553.





70. R. Peach, A. Arnaudon, M. Barahona, Relative, local and global dimension in complex networks. Nat. Commun. 13, 1-11 (2022).





71. A. Arnaudon, R. L. Peach, M. Barahona, Scale-dependent measure of network centrality from diffusion dynamics. Phys. Rev. Res. 2, 033104 (2020).





72. G. Bertagnolli, M. De Domenico, Diffusion geometry of multiplex and interdependent systems. Phys. Rev. E 103, 042301 (2021).





73. L. Kaufman, P. J. Rousseeeuw, Finding Groups in Data: An Introduction to Cluster Analysis (John Wiley & Sons, 2009).

