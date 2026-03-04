下面补一组 2024–2025 年、比较适合作为“综述/系统性梳理/评测框架”来引用的 population synthesis 相关文献（我按“更像综述、更适合放 related work 骨架”的优先级排在前面；后面几篇是“综述型/框架型/大规模对比型”，对你们写 top 期刊的 evaluation/positioning 也很有用）。

1. La, Vu, Kamruzzaman, Miller (2025). Population synthesis: a problem-based review. Transport Reviews.
   一句话摘要：把 PopSyn 从“方法谱系”改为“问题驱动”的系统综述，按挑战（如数据缺口、稀疏高维、约束一致性、验证与泛化等）组织文献并指出关键缺口。
   建议位置：Related Work 的骨架（尤其适合你们强调“边际约束下恢复异质 copula / 条件信息质量决定性能”的叙事）。 ([Monash University][1])

2. Rezvany, Kukic, Bierlaire (2024). A Review of Activity-Based Disaggregate Travel Demand Models. Findings.
   一句话摘要：从“选择/行为建模”的角度综述 activity-based 模型，并把 synthetic population generator 作为 ABM 关键模块一并讨论（适合作为 PopSyn 在 ABM pipeline 中的定位综述引用）。
   建议位置：Introduction/Related Work（ABM 与 PopSyn 的关系、PopSyn 对 ABM 的约束与需求）。 ([Findings][2])

3. Nguyen, Schweizer, Rupi (2025). Large-scale activity-based demand generation modeling: A literature review and exploration of potential approaches. Transportation Engineering.
   一句话摘要：面向“大规模 ABM demand generation”的综述，明确把“合成人口 + 活动计划生成”作为核心，并对方法、数据需求、输出与适用性做对比。
   建议位置：Related Work（把你们的 PopSyn 问题放进更大 ABM demand generation 的语境里，尤其当你们要强调空间异质性对下游 ABM 的影响时）。 ([科学直达][3])

4. Roxburgh, Paolillo, Filatova, Cottineau, Paolucci, Polhill (2025). Outlining some requirements for synthetic populations to initialise agent-based models. Review of Artificial Societies and Social Simulation.
   一句话摘要：站在 ABM 初始化角度提出“合成人口方法的愿望清单”，特别强调真实世界属性之间的相互关联/协方差结构若被破坏会带来偏差与错误推断。
   建议位置：Discussion/Related Work（非常契合你们“copula/依赖结构是关键，不能只对齐边际”的论点）。 ([TU Delft Repository][4])

5. Darsel, Come, Oukhellou (2025). Robust and Reproducible Evaluation Framework for Population Synthesis Models — Application to Probabilistic and Deep Generative Models. SSRN working paper.
   一句话摘要：提出 PopSyn 的可复现实证评测框架与指标，并做了概率模型与深度生成模型（包含 diffusion）的系统 benchmark；明确讨论宏观分布相似、微观真实性与隐私等评测维度。
   建议位置：Methods/Experiments（作为“评估口径先例 + diffusion 在 PopSyn/tabular 生成中的定位”引用很合适）。 ([SSRN][5])

6. Sané, Vandanjon, Belaroussi, Hankach (2025; online 2024). A comprehensive investigation of variational auto-encoders for population synthesis. Journal of Computational Social Science.
   一句话摘要：面向实践者的“VAE 做 PopSyn”的系统性讲解 + 大量结构/超参/样本量/指标的对比实验，并把它与传统方法（如 IPF）在能力与风险上做清晰对照。
   建议位置：Related Work（DL-based PopSyn 的综述型引用）+ Experiments（DL baseline/指标选择的依据）。 ([Springer Nature Link][6])

7. Mensah, Badu-Marfo, Farooq (2025). Robustness Analysis of Deep Learning Models for Population Synthesis. Transportation Research Procedia.
   一句话摘要：讨论“深度生成 PopSyn 在不同数据集/不同样本量下的稳健性评估”，用 bootstrap 置信区间等方式比较 CTGAN 与 VAE，并在多年份出行日记数据上做实证。
   建议位置：Experiments/Discussion（你们要强调“条件信息质量/数据口径”对结果的影响时，这篇很顺手）。 ([科学直达][7])

8. Bigi, Rashidi, Viti (2024). Synthetic Population: A Reliable Framework for Analysis for Agent-Based Modeling in Mobility. Transportation Research Record.
   一句话摘要：从交通 ABM 的使用视角提出“合成人口可靠性/分析框架”，强调 SPG 过程与结果的可信度与验证维度。
   建议位置：Experiments（评价维度、可靠性/验证框架的先例引用）。 ([UNSW Sites][8])

如果你们只想挑“严格意义上的 review paper（综述论文）”，目前这批里最“纯综述”的是 La et al. 2025（problem-based review）以及 Nguyen et al. 2025（literature review，覆盖 demand gen pipeline 并包含 PopSyn）。其余几篇更偏“综述型框架/benchmark/立场文”，但对 top 期刊写作往往更有用：它们能给你们 evaluation 与“为什么要关心 copula/协方差结构”的叙事提供权威背书。

下面把以上条目汇总成 BibTeX（你可以直接粘到 references.bib；其中 Findings/ROFASSs/SSRN 属于非传统期刊条目，我用 @article/@misc/@techreport 做了最通用的写法）。

```bibtex
@article{La2025PopSynProblemReview,
  title   = {Population synthesis: a problem-based review},
  author  = {La, Duc Minh and Vu, Hai L. and Kamruzzaman, Liton and Miller, Eric},
  journal = {Transport Reviews},
  volume  = {45},
  number  = {3},
  pages   = {366--389},
  year    = {2025},
  doi     = {10.1080/01441647.2025.2469069}
}

@article{Rezvany2024ABMReview,
  title   = {A Review of Activity-Based Disaggregate Travel Demand Models},
  author  = {Rezvany, Negar and Kukic, Marija and Bierlaire, Michel},
  journal = {Findings},
  year    = {2024},
  month   = {12},
  doi     = {10.32866/001c.125431}
}

@article{Nguyen2025DemandGenReview,
  title   = {Large-scale activity-based demand generation modeling: A literature review and exploration of potential approaches},
  author  = {Nguyen, Ngoc An and Schweizer, Joerg and Rupi, Federico},
  journal = {Transportation Engineering},
  volume  = {20},
  pages   = {100329},
  year    = {2025},
  month   = {6},
  doi     = {10.1016/j.treng.2025.100329}
}

@misc{Roxburgh2025PopSynthRequirements,
  title        = {Outlining some requirements for synthetic populations to initialise agent-based models},
  author       = {Roxburgh, Nick and Paolillo, Rocco and Filatova, Tatiana and Cottineau, Cl{\'e}mentine and Paolucci, Mario and Polhill, J. Gareth},
  howpublished = {Review of Artificial Societies and Social Simulation},
  year         = {2025},
  month        = {1},
  url          = {https://rofasss.org/2025/01/29/popsynth/}
}

@article{Sane2025VAEPopSyn,
  title   = {A comprehensive investigation of variational auto-encoders for population synthesis},
  author  = {San{\'e}, Abdoul Razac and Vandanjon, Pierre-Olivier and Belaroussi, Rachid and Hankach, Pierre},
  journal = {Journal of Computational Social Science},
  volume  = {8},
  pages   = {13},
  year    = {2025},
  doi     = {10.1007/s42001-024-00332-0},
  note    = {Article number 13; published online 11 Dec 2024}
}

@article{Mensah2025RobustnessDLM,
  title   = {Robustness Analysis of Deep Learning Models for Population Synthesis},
  author  = {Mensah, Daniel Opoku and Badu-Marfo, Godwin and Farooq, Bilal},
  journal = {Transportation Research Procedia},
  volume  = {82},
  pages   = {3790--3806},
  year    = {2025},
  doi     = {10.1016/j.trpro.2024.12.016}
}

@article{Bigi2024ReliableFrameworkABM,
  title   = {Synthetic Population: A Reliable Framework for Analysis for Agent-Based Modeling in Mobility},
  author  = {Bigi, Federico and Rashidi, Taha Hossein and Viti, Francesco},
  journal = {Transportation Research Record: Journal of the Transportation Research Board},
  volume  = {2678},
  number  = {11},
  pages   = {1--15},
  year    = {2024},
  doi     = {10.1177/03611981241239656}
}

@techreport{Darsel2025EvalFrameworkPopSyn,
  title       = {Robust and Reproducible Evaluation Framework for Population Synthesis Models---Application to Probabilistic and Deep Generative Models},
  author      = {Darsel, Vianey and Come, Etienne and Oukhellou, Latifa},
  institution = {SSRN},
  type        = {Working Paper},
  year        = {2025},
  month       = {6},
  doi         = {10.2139/ssrn.5295092},
  url         = {https://ssrn.com/abstract=5295092},
  note        = {Posted 14 Jun 2025; last revised 20 Nov 2025}
}
```

如果你希望我把这些条目按你们 essay 的章节结构（Introduction/Related Work/Methods/Discussion）“嵌回去”，我也可以直接给一段可粘贴的 related-work 段落骨架，并把每篇综述/框架在你们“copula 异质性 + distribution-level diffusion + 条件信息质量”的叙事里对应到具体句子。

[1]: https://research.monash.edu/en/publications/population-synthesis-a-problem-based-review/ "https://research.monash.edu/en/publications/population-synthesis-a-problem-based-review/"
[2]: https://findingspress.org/article/125431-a-review-of-activity-based-disaggregate-travel-demand-models "https://findingspress.org/article/125431-a-review-of-activity-based-disaggregate-travel-demand-models"
[3]: https://www.sciencedirect.com/science/article/pii/S2666691X25000296 "https://www.sciencedirect.com/science/article/pii/S2666691X25000296"
[4]: https://repository.tudelft.nl/file/File_437c2678-a135-4af4-b553-0470dc75ae60 "https://repository.tudelft.nl/file/File_437c2678-a135-4af4-b553-0470dc75ae60"
[5]: https://papers.ssrn.com/sol3/Delivery.cfm/5295092.pdf?abstractid=5295092&mirid=1 "https://papers.ssrn.com/sol3/Delivery.cfm/5295092.pdf?abstractid=5295092&mirid=1"
[6]: https://link.springer.com/article/10.1007/s42001-024-00332-0 "https://link.springer.com/article/10.1007/s42001-024-00332-0"
[7]: https://www.sciencedirect.com/science/article/pii/S2352146524003144 "https://www.sciencedirect.com/science/article/pii/S2352146524003144"
[8]: https://www.unsw.edu.au/research/rciti/about-us/publications "https://www.unsw.edu.au/research/rciti/about-us/publications"
