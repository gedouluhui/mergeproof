# MergeProof

检查 LoRA 合并后的文件是否满足你声明的权重更新关系，而不只是确认合并命令执行成功。

```sh
python -m pip install .
mergeproof demo --output demo
```

打开 `demo/good.html` 与 `demo/bad.html`。坏样例真的把更新系数加倍，由检查器读取文件后定位错误，不是预写结论。输出目录须不存在。

首版只支持单文件 safetensors、FP32、标准非量化 Linear LoRA。需要显式提供 base/A/B 对应键与 alpha；不自动猜层名，不执行模型，不需要 GPU 或 API key。CLI、清单格式与误差公式见[英文文档](README.md)。

数值策略比较更新量，而不是按巨大 base 权重放宽容差。默认容差不保证适合所有真实合并后端；请结合误差证据判断，不要为了通过而盲目调大。若原始 base 本身也能通过，报告 INDETERMINATE，而不是证明合并发生过。

退出码：PASS 0、FAIL 1、ERROR 2、INDETERMINATE 3、UNSUPPORTED 4。未知变体、低精度与不完整输入不会默认为通过。

这不是模型质量评测、训练来源证明或后门检测工具。目前尚无真实用户采用数据；MIT 许可，AI 辅助实现。提交问题时请使用自制小样例，不上传私有权重。

## 快速开始与支持状态

### 问题
验证本地 LoRA 合并是否把声明的更新应用到正确张量，并且未改变非目标字节。

### 安装
```sh
python -m pip install .
```

### 成功命令
```sh
mergeproof demo --output demo
```
预期：生成的正确对照通过，演示写出可检查的 HTML/JSON 证据；退出码 0。

### 故意失败命令
```sh
mergeproof verify --base demo/base.safetensors --adapter demo/adapter.safetensors --merged demo/bad.safetensors --manifest demo/manifest.json
```
预期：故意加倍的更新被报告为 FAIL；退出码 1。

### 证据与覆盖边界
证据包括手算矩阵、错误缩放、同形状层交换、缺失/未使用键、非目标变更、非有限值、分块不变性、零更新歧义和 CLI 输出测试。覆盖范围是显式 FP32 标准 Linear LoRA 方程，不是模型行为或形式化验证。

### 支持状态
版本 0.1.0 支持 Python 3.10+、单文件 FP32 safetensors 和标准非量化 Linear LoRA。文档路径是源码检出。维护为尽力而为，不提供 SLA。

### 限制
不支持分片索引、FP16/BF16、DoRA、rsLoRA、量化、偏置更新、额外模块和修改后的基座语义。哈希标识字节但不认证来源；检查器也不是对抗性并发文件系统沙箱。
