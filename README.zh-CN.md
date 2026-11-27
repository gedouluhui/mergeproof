# MergeProof

检查 LoRA 合并后的文件是否满足你声明的权重更新关系，而不只是确认合并命令执行成功。

```sh
python -m pip install .
mergeproof demo --output demo
```

打开 `demo/good.html` 与 `demo/bad.html`。坏样例真的把更新系数加倍，由检查器读取文件后定位错误，不是预写结论。输出目录须不存在。

当前版本只支持单文件 safetensors、FP32、标准非量化 Linear LoRA。需要显式提供 base/A/B 对应键与 alpha；不自动猜层名，不执行模型，不需要 GPU 或 API key。CLI、清单格式与误差公式见[英文文档](README.md)。

判定采用向外舍入的区间包络；区间无法判定的元素用精确有理数（Fraction）按固定行序复核，共享 `--max-exact-terms` 预算（默认 100000 项，0 关闭）。预算耗尽报 INDETERMINATE，绝不报 PASS。补偿求和只作为诊断残差展示，不参与判定。

数值策略比较更新量，而不是按巨大 base 权重放宽容差。默认容差不保证适合所有真实合并后端；请结合误差证据判断，不要为了通过而盲目调大。若原始 base 本身也能通过，报告 INDETERMINATE，而不是证明合并发生过。

PEFT 路径已在 CPU 上以 PEFT 0.19.1 / torch 2.12.0 实测 rank 1、4、8：真实合并 PASS，alpha 加倍 FAIL，未合并的 base FAIL。其他版本与后端未验证。

退出码：PASS 0、FAIL 1、ERROR 2、INDETERMINATE 3、UNSUPPORTED 4。未知变体、低精度与不完整输入不会默认为通过。

这不是模型质量评测、训练来源证明或后门检测工具。目前尚无真实用户采用数据；MIT 许可，AI 辅助实现。提交问题时请使用自制小样例，不上传私有权重。
