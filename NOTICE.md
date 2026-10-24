# 来源与归属说明

AgentSkillOps 是基于 [AIMING Lab / SkillRL](https://github.com/aiming-lab/SkillRL) 实验体系开展的
个人研究扩展，不是 SkillRL 官方发行版。

上游项目提供了技能抽取、分层技能库、ALFWorld Agent 评测和强化学习训练基础。本仓库新增并
重点展示以下部分：

- 技能效用的配对测量、时间线存储与可靠性审计；
- 跨 checkpoint 的状态估计和可识别性检查；
- `KEEP / RETIRE / READMIT` 生命周期规则与 VOI 复测排序；
- 2×2 因子分析、非劣检验和独立任务验证；
- 针对上述模块的自检、结果整理和中文技术文档。

为避免混淆贡献边界，仓库不包含上游的 `verl` 训练框架副本、Agent 环境实现、模型权重或数据集。
引用或复用上游成果时，请同时遵循上游仓库的许可与引用要求。
