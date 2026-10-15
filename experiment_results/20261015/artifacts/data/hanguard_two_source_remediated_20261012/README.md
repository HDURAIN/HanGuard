# 同切分类别子集 20261012

来自 `../three_source_remediated_20261012/` 的中文整理/JailBench已知类型记录，共37,042条，训练／验证／测试 **29,634／3,704／3,704**。有害类型监督分别 **17,440／2,180／2,180**。不再次切分，完整输入、binary、known category、group和split逐条继承父数据。683条native未知类型从本视图排除，但仍在三来源binary主集中。

安全0保留用于门控六类评估；1–5参加类型损失。`descriptions.json`继承历史已冻结描述，不当作新重标依据。注册必须给匹配新版binary的父模型；旧父模型或错版/错split数据会拒绝。详见父数据README及 `../../docs/training.md`。
