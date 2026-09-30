# Codex 接入：互斥单角色、20用户

> 原始 Mock v2 包的来源记录，保留供追溯。当前启动与验证入口见 [README](../README.md)，当前约束见 [CONTRACTS](../CONTRACTS.md)。

## 目标

消费这份SQL关联数据派生的ACL和20个本地mock用户，验证知识库检索的权限边界；暂不对接外部登录流程。

## 必须满足

1. RoleIds恰好1项，且来自MSFT=1 / Partner=2 / 21V=3。混合角色不是待决业务场景，而是非法输入。不能去重或合并后放行。
2. 五组各4个：MSFT全业务、21V全业务、Partner仅NCE、Partner仅OSPA、Partner双业务。20条都为正常模拟身份，不增加admin、无角色、Partner空业务等合法profile。
3. 使用effective_business_categories。内部默认[0,1]；Partner使用明确授权。本地业务NCE=0、OSPA=1，不要直接套外部反向编号。
4. 原始SQL的Role_MenuItem不修改、不按标题或RolePermission补授权。当前三个角色菜单授权相同是源数据事实，不是需要修复的缺陷。
5. SQL数据内角色限定和业务限定都要检查，内部角色不得绕过角色菜单权限。
6. 先运行离线37项测试，再将真实ES结果与expected_access.json和2600条authorization_cases.jsonl比较。不能将离线PASS标为DLS已验证。
7. Source Document.Status发布枚举未核验，ACL期望不是开放决定。真实ES测试须另设明确的本地开放步骤；不能静默写成published。
8. 使用同一问题“合同的生成”检索时，Document25是NCE受控、Document99是OSPA受控。用按ID访问补足精确安全断言；没有越权不代表相关性已经达标。
9. 同组用户同权是正常的。20身份可用于会话、查询缓存隔离和回归，不能虚构20种文档授权范围。
10. fixtures/expected_access.json、source_acl_tables.json及authorization_cases.jsonl是服务端测试资料，不返回前端、不落入用户可检索知识。
11. es_metadata_candidates.json只提供用户metadata投影，不包含ES账号创建动作、密码或DLS角色。已有本地ES绑定机制需单独接入。这里没有模拟有效JWT或真实ClaimTypes字符串。
12. 默认关闭mock，只有显式development/test启用，生产拒绝；用户只选择预设profile，前端不能自报角色/业务列表。

## 接口替换点

保持 UserAuthenticationInfo原始形状 → normalize_authentication_info → Principal 的单一规范化入口。
将未来真正登录返回值接到同一入口，不复制salt、用户组查询、服务Token登录等流程到fixture。
若上游开始直接返回内部用户[0,1]，规范化仍然得到相同effective结果；原始值保留供调试。

## 输出结果需分开报告

- 离线输入/规则测试结果。
- ES认证身份、metadata及受限读取角色的实际结果。
- 文档ACL集合验证结果（不等于全文查询TopK）。
- 自然语言检索效果、发布条件、撤权与缓存失效等尚未验证部分。

不要覆盖旧源码或Docker数据卷，也不要将这份mock上传成生产用户记录。
