# Playbook ASK AI：本地权限检索 Roadmap v1

日期：2026-09-28  
状态：2026-09-29 完成S0–S6本地MVP；20身份DLS、真实HTTP、页面、冻结金标及隔离更新撤权验收已通过。生产认证和上游发布语义仍未完成。  
目标环境：Claus 的 Mac、Docker、本地 Codex。  
S0 产物：[契约](CONTRACTS.md)、[项目约束](AGENTS.md)、[验证台账](VALIDATION_STATUS.md) 和 uv 离线基线检查；不包含在线应用实现，不修改原始 SQL 或权限预期。

## 1. 本轮交付目标

把现有 Playbook 静态阅读 HTML 接到真实后端和 Elasticsearch：开发者选择一个预置 mock 身份，以自然语言或关键词发起检索；系统在该用户可访问且已获得本地测试开放的文档范围内，返回最多 K 个不同 Document，显示命中依据，并允许打开对应中英文正文。

ASK AI 是交互入口名称，首版能力明确标注为「文档检索 · Elasticsearch」。不调用聊天模型，不生成答案，不伪装成已完成 RAG；命中摘录来自原文。

不需要从零制作新 UI。保留原来的菜单、阅读区、语言切换和视觉样式，改造数据来源与搜索交互。

### 1.1 本轮范围

- 单节点 Docker Elasticsearch、认证与 TLS、DLS 功能检查。
- SQL 抽取数据 → Markdown 原文、检索文本、metadata 和 ACL → ES。
- 20 个互斥角色 mock 用户的后端会话、有效权限与 ES 身份映射。
- 原生 DLS、授权范围内的 BM25 检索、中英文分析与小规模审核同义词。
- 菜单、搜索、详情、原文、下载统一走受控 API。
- HTML ASK AI 接入及用户切换隔离。
- 权限、安全、相关性、变更、重启与可复现验收。

### 1.2 不纳入首版

- ACN 内容导入或跨知识库联合搜索。
- 真实外部登录流程、用户组查询、生产 SSO。
- 向量、混合检索、Rerank、LLM 回答、Agent、多轮指代与长期记忆。
- 完整权限后台、生产多租户、生产高可用、复杂工作流。
- 自动抓取链接目标、PDF/DOCX 附件、OCR、图片理解。
- 根据名称猜测多菜单授权、父菜单权限继承或发布枚举。

## 2. 已有输入与权威顺序

|输入|处理原则|
|---|---|
|playbook-sampledata.sql|本地测试快照的原始来源；保留原始字节，不执行为生产变更|
|Playbook_SQL_Markdown_ACL_Extraction_v1.zip|仅复制 tools/extract_playbook_sql.py；已验证可从本地 SQL 在新目录重建 Markdown、metadata 和 ACL，原包未复制的产物不是前置缺口|
|Playbook_Mock_Users_20_v2.zip|唯一有效用户 fixture；使用互斥角色 v2，不混用旧版合法混合角色|
|Playbook_Portal_20Users_v1.zip|保留 src/template.html、src/style.css 和必要交互；拆除全量嵌入数据路径|
|Playbook_知识库_20用户演示.html|只作为 UI/行为参考和独立离线样例；不能作为在线服务的可下载备份|
|Playbook_ES_Lab_Codex_Starter_v0.1.zip|只复核并选择性复用基础设施；其 Excel 导入器、旧权限数量与旧 ID 示例已经过时|
|UserService.cs / UserAuthenticationInfo.cs|说明身份返回结构和转换来源；本轮不复刻外部认证协议|
|现有项目 README|维持授权范围、来源追溯、先小闭环后扩大等产品目标；不要重写无关模块|

原始 SQL 是内容与关系依据，最近确认的业务规则是权限语义依据。派生包必须与其核对，不能用旧 Excel 或旧报告替代新 SQL。测试期望是独立 oracle，不允许因实现失败而修改期望来获得 PASS。

### 2.1 本快照的基线，不是生产不变量

- Document 130，MenuItem 164，MenuItem_Document 130，Role 3，Role_MenuItem 195；Notice 4 不并入 Document。
- 130 个逻辑 Document、260 个中英文源字段文件；每篇文档在此快照中恰好一个有效菜单关联。
- Public 76（NCE43、OSPA33）；受控 NCE26、OSPA28。
- 三个角色在当前 SQL 中具有相同的 65 个受控菜单授权；不人为制造真实用户的额外限制。
- 这些是 ACL 统计。Document.Status 的发布语义未核验，不能直接将 1 转换为 published。
- SQL 的 Document146 不再是 OSPA 访问权限说明；后者为 Document148。安全样例使用 Document25 和99。

### 2.2 身份及 ACL 规则

角色：MSFT=1、Partner=2、21V=3。业务：NCE=0、OSPA=1。

有效用户必须恰好一个角色；空角色、混合角色和重复角色均拒绝。MSFT/21V 的 effective_business_categories 为 [0,1]；Partner 必须是 [0]、[1] 或 [0,1]。内部用户没有菜单权限豁免。

访问规则：

    用户会话有效
    AND 文档符合开放条件
    AND 文档 ACL 可解析
    AND (
        public
        OR (角色拥有菜单权限 AND 用户拥有对应业务权限)
    )

沿用本地样例的直接单菜单规则。Active=1、Role.Status=1 是暂定的保守实验门槛；正式语义仍待核对。不推导父级继承。遇到多菜单先隔离而非自动合并。

原始 BusinessCategories 与 effective_business_categories 分开保存。附件 ToClaims() 仅输出已有值；内部默认业务权限在受控的规范化层补齐。不可直接使用外部反向编号（外部 key0=OSPA，key1=NCEI）。

## 3. 推荐架构与信任边界

    SQL快照 / 后续上游导出
            ↓
    已有抽取器：Markdown + metadata + ACL
            ↓
    校验、正文解析、版本与开放状态处理
            ↓
    Elasticsearch：1个逻辑Document = 1条索引记录
            ↑
    API：会话 → 有效身份 → 对应ES只读身份 → DLS读取
            ↑
    现有HTML：菜单 + ASK AI + Markdown阅读区

建议先用一个轻量 Python API 服务（可选 FastAPI）同时提供 UI 静态资源与 /api，单一 origin。ES 单独在 Docker 中；API 也可容器化以便复现，本机开发可热重载。Kibana 不是必需服务，不增设 Redis、消息队列或用户数据库作为前置条件。

### 3.1 开发身份路径的推荐默认值

1. 浏览器只能在 development/test 模式中选择20个预设 profile之一。
2. 后端从服务器 fixture取得身份、做单角色校验与有效业务规范化，创建或切换会话。
3. 为20个 profile建立对应的 ES native user，使用统一的受限 DLS reader 角色和各自 metadata。
4. API按已验证会话映射到对应ES身份。第一版可将随机凭据仅保存在服务端秘密存储中，查询客户端按请求/身份绑定；禁止修改共享客户端全局 Authorization 导致并发串号。
5. 浏览器不持有ES密码、API Key、任意RoleIds或BusinessCategories授权参数。开发用户选择器不是生产登录。
6. 未来可通过 ESIdentityResolver 替换成 run_as 或正式身份集成，本轮不同时实现多种机制。

DLS是本轮推荐主路径；若许可证/版本无法支持，暂停该路径并形成独立决策，不能静默降级、移除DLS或使用管理员读取后过滤。应用层强制filter是可单独评估的替代方案，不与DLS验收混为一谈。

### 3.2 DLS并不保护HTML源码与磁盘文件

现有 HTML 明确内嵌全部正文、metadata 和 mock 用户。在线版必须移除它们，包括隐藏的原文、预渲染HTML、下载模板和 expected_access 数据。

只对专门的 frontend/dist 静态目录提供服务；不要将仓库根、SQL、ZIP、Markdown数据目录、日志、旧全量HTML或测试fixture暴露在HTTP下。

搜索、菜单、详情、原文、下载、历史路径都必须授权。API不得用管理员读出全文再交给浏览器隐藏。中英文内容使用同一Document ACL。

## 4. 里程碑、依赖与退出条件

|阶段|目标|依赖|退出后获得的能力|
|---|---|---|---|
|S0|冻结输入、边界与接口契约|已有附件|Codex不会混用旧Excel/旧用户/旧ID|
|S1|Mac Docker ES基础环境|S0|真实集群、认证/TLS、DLS能力可验证|
|S2|内容、metadata和ACL入库|S0；入库依赖S1|130个逻辑文档可复现导入|
|S3|20身份与DLS访问隔离|S1+S2|M1：权限化知识库基线|
|S4|后端会话、目录、检索与详情API|S3|M2：授权检索API|
|S5|HTML ASK AI与按需读取|S4|M3：可使用的本地ASK AI文档搜索|
|S6|变更回归、搜索评测与交付|S3+S4+S5|M4：可复现、可度量的本地MVP|

本项目按用户要求一次只执行一个阶段；离线解析的技术可并行性不改变 S0→S1→S2→S3→S4→S5→S6 的执行边界。真实 DLS 通过前不能把界面展示当成安全验收。S0 已记录问题集/Golden Set 缺失及审核规则；S2 开始整理候选，不到 S6 才临时设计。

## 5. 分阶段 To-do

### S0 — 基线与数据/API契约

目标：先消除包与规则之间的不一致，不启动全仓库重写。

- [x] S0-01 盘点已有仓库结构与AGENTS说明，确定一个独立Playbook实验边界；保护现有ACN及functional-demo。
- [x] S0-02 校验SQL与派生包来源一致；确认只使用20用户v2；归档旧Excel对应数量与映射，禁止作为新版断言。
- [x] S0-03 记录角色互斥、内部默认双业务、Partner显式业务和public跨业务规则。
- [x] S0-04 冻结DocumentRecord、Principal、SearchRequest/SearchResponse最小契约。
- [x] S0-05 区分publication_state=unverified与独立lab_release.approved；开放行为需要显式操作，不能默认将130篇视为生产发布。
- [x] S0-06 冻结ASK AI首版为检索：结果卡、证据摘录、打开正文；不包含LLM回答或多轮上下文。
- [x] S0-07 标明发布/Active/多菜单/继承/时区等未决项及安全的本地处理方式；不等待真实认证接口才能开始实验。
- [x] S0-08 在项目中维护ROADMAP、CONTRACTS、VALIDATION_STATUS和必要AGENTS约束；不为每个小功能新建一套文档系统。

产出：基线清单、CONTRACTS、源数据映射、验证台账。

退出条件：能从Document25/99追溯到正式菜单和角色；明确哪些已确认、哪些为本地实验假设；Codex知道哪些旧文件不可沿用。

实际完成记录：见 [VALIDATION_STATUS.md](VALIDATION_STATUS.md)。uv + CPython 3.12.12 下 39 项离线测试通过；S0 未启动 ES、创建用户或开放文档。

### S1 — Docker与Elasticsearch运行底座

- [x] S1-01 确认Mac架构（arm64/x86_64）、Docker可用内存、端口冲突和磁盘位置。
- [x] S1-02 锁定实际可拉取并与客户端/分析插件兼容的ES版本与镜像；不用latest，也不未经核验沿用旧包版本号。
- [x] S1-03 单节点Compose、命名数据卷、ES认证、TLS/CA验证和healthcheck；对外端口仅本机回环。
- [x] S1-04 初始化管理员只用于bootstrap；另建导入/维护身份和只读身份，运行时API不持有管理员回退路径。
- [x] S1-05 检查当前许可证与DLS实际能力；需要trial时显式激活并记录到期日。许可失败不得关闭安全设置绕过。
- [x] S1-06 固定中文分析方案，先以Smart Chinese等候选实测术语token；记录安装和重建方式。
- [x] S1-07 检查容器重启后索引、用户、安全角色和证书仍有效；销毁数据卷必须显式人工操作。

产出：运行环境、版本/许可证/插件记录、可复现启动停止说明。

退出条件：TLS证书验证通过、无凭据访问失败、集群可读写测试成功、DLS能力检查真实通过；不以YAML静态解析代替Docker验收。

实际完成记录：见 [VALIDATION_STATUS.md](VALIDATION_STATUS.md) 与 [S1_CHECKS.json](reports/S1_CHECKS.json)。ES/Smart Chinese 9.5.4，回环 19200；14 条独立合成数据、两个 native reader 的真实 DLS 与容器重启检查通过。Trial 到期 2026-10-29 07:27:15 UTC。未执行 S2 真实入库、S3 20 用户验收或运行时 API。

### S2 — Markdown、检索文本、metadata与ACL入库

- [x] S2-01 复用SQL解码和260个原文字段的保真检查；不依赖Excel正文、不要求先恢复SQL Server。
- [x] S2-02 为每篇文档保留原文、解析文本、业务metadata和ACL四类数据。不能将SQL包装、SVG path、CSS或完整测试期望索引为知识。
- [x] S2-03 Markdown/内嵌HTML解析：提取标题、章节和正文，保持步骤及合并表格的关联；复杂表格保留HTML原文和有语义的检索文本，不暴力strip后丢失行列关系。
- [x] S2-04 原文文件不改写；用于阅读的HTML按允许列表清理。英文源字段可能混有中文，相同中英字段不擅自翻译或删掉，避免重复字段造成无意双倍加权。
- [x] S2-05 冻结1个Document对应1条ES记录，双语为字段；使用source_instance+Document.Id的稳定key，不能用标题当主键。菜单不是知识正文索引中的额外文档。
- [x] S2-06 明确mapping：ID/业务/ACL做精确字段，文本做分析字段；保持source、时间原值、内容/ACL版本标识。无时区datetime不自行追加UTC标记。
- [x] S2-07 保留Document→菜单的正式关系及ACL来源证据；缺失、多挂载、不一致、未知状态先隔离，不推断授权。
- [x] S2-08 实现导入前dry-run、批量错误清单与幂等写入；相同快照重入不增加逻辑文档。
- [x] S2-09 显式执行本地开放步骤，记录快照和操作者；未开放内容在用户端不可读取。内容候选可存储，索引成功不等于发布成功。
- [x] S2-10 为同义词先建立小型、可版本化、经业务确认的配置与_analyze用例；首版不做术语管理后台。

产出：索引mapping、导入清单、错误/隔离报告、原文与检索文本抽样、开放记录。

退出条件：130篇单文档、260份原文一致；重复导入仍130；Document25/99/148映射正确；复杂表格与中英混排有检查证据；开放/未开放可区分。

实际完成记录：见 [验证台账 S2](VALIDATION_STATUS.md#s2实际修改了什么)、[ES 检查](reports/S2_CHECKS.json)、[表格抽样](reports/S2_CONTENT_SAMPLES.md)。独立索引130条、原文260份一致，导入三次计数130/130/130；受限NCE探针开放前0条、开放后102条，仍拒绝99/148。用户确认的三组实验同义词已验证；32条候选问题仍待业务审核，不是Golden Set。离线46项通过；S3尚未执行。

### S3 — 20用户与DLS授权门槛（M1）

- [x] S3-01 消费mock_users.json并规范化，使用effective_business_categories；20个合法用户外不增加管理员业务profile。
- [x] S3-02 创建20个对应ES native user及可信metadata；所有查询用户只授予同一个受限reader角色。
- [x] S3-03 配置完整DLS：开放条件 AND ACL可解析 AND（public OR（角色交集 AND 业务匹配））。角色和业务不能拆成两个ES角色期待自动取交集。
- [x] S3-04 使用每个真实ES身份检查_authenticate，确认请求不是管理员或同一个宽权限账号。
- [x] S3-05 在统一测试开放条件下，以match_all等取得完整可访问ID集合，与expected_access逐ID比对；不只比较数量。
- [x] S3-06 对20×130组合检验按ID读取/批量读取与搜索路径；初次DLS证明不附加重复应用ACL来掩盖策略缺陷。
- [x] S3-07 建立单独合成索引/临时内存策略测试，覆盖角色专属、无授权、多角色非法输入；不得修改真实SQL来制造角色差异。
- [x] S3-08 验证只读用户无法写正文/ACL、改角色或修改自身授权；检索结果元数据不得暴露其他不可读文档。
- [x] S3-09 凭据仅服务端保存、生成时不打印、Git忽略；fixture metadata不能作为浏览器自报授权。

基线：MSFT/21V/Partner双业务各130；Partner-NCE为102；Partner-OSPA为104。前提是同一完整快照且全部合法文档已获实验开放。这是ACL集合，不是每个问题的召回数。

退出条件：2,600个逐文档预期无误、角色与业务两个条件都经过独立反例验证；旧Excel42/88结论和旧Document146示例不参与测试。

实际完成记录：见 [S3实测报告](reports/S3_CHECKS.json) 与 [验证台账](VALIDATION_STATUS.md#s3实际修改了什么)。20个独立native身份、统一reader；2,600个真实组合及440个合成组合均在逐ID搜索/GET/_mget通过，120次写入/改权/越界操作403；51项离线回归通过。用户同时批准32条问题作为V1验证问题，后续补齐身份/相关性标注。S4尚未执行。

### S4 — 后端会话与权限内检索API（M2）

- [x] S4-01 建立开发mock会话：仅允许预设profile，返回当前Principal；切换/注销使旧会话失效。列表只返回开发选择所需标签，不返回全量用户授权/测试矩阵。
- [x] S4-02 每个请求由会话解析用户，不接受query body中的role_ids、业务授权、ES DSL或任意ES凭据；请求scope只能缩小可见范围。
- [x] S4-03 用服务端ESIdentityResolver绑定到对应native user；并发查询不共享可变身份请求头；无身份/映射异常必须拒绝。
- [x] S4-04 提供只含可见导航的menu API。父节点不能访问但其子文档可访问时，沿用「上级目录」占位，不发送未授权父菜单名称或扩大文档权限。
- [x] S4-05 实现文档级BM25：标题/章节/术语/中英正文，默认TopK=5、最大20作为建议初值；明确字段权重为待评测参数。
- [x] S4-06 将中文、英文、缩写和中英同义词放入受控查询处理；不能无条件把取消/删除/暂停等非等价动作合并。
- [x] S4-07 统一返回文档ID、标题、可展示来源、命中摘录、命中语言、版本指纹、retrieval_mode和request_id。摘录是源证据，不是生成答案；_score不转换为可信度百分比。
- [x] S4-08 详情、原文和下载使用相同ES身份/授权边界；磁盘原文不能通过公开静态目录绕过。未知ID与无权ID避免泄露存在性。
- [x] S4-09 区分空结果、身份失效、权限拒绝、ES故障/超时与参数错误；ES失败不得回退全量本地关键词检索。
- [x] S4-10 首版关闭搜索/正文业务缓存以降低复杂度；如加入缓存，key必须包含身份/有效授权指纹、索引代次和查询条件，并有失效机制。
- [x] S4-11 记录request_id、身份标识、索引/策略/分析器版本、结果ID与耗时；默认不记录密码、Token或全文，问题文本按调试需要脱敏。

S4已实现API（具体响应见CONTRACTS）：

|接口|职责|
|---|---|
|GET /api/dev/profiles|仅本地开发的预设用户标签|
|POST /api/dev/session|选定profile建立/切换会话，不接受任意授权列表|
|DELETE /api/session|注销并失效当前会话|
|GET /api/me|当前身份与有效权限展示|
|GET /api/menu|授权后的菜单与可见数量|
|POST /api/search|问题→TopK文档证据|
|GET /api/documents/{key}|授权后返回正文/渲染所需内容|
|GET /api/documents/{key}/raw|授权后返回某个语言源字段|
|GET /api/documents/{key}/download|授权后导出Markdown或允许的metadata|

POST /api/search 的最小输入：query、top_k、language、可选business_scope。
最小输出：request_id、retrieval_mode=bm25、items、took_ms。item包含document_key、document_id、title、business_category、matched_language、snippets、source、content_revision、acl_revision；后两者首版为本地派生标识，不假冒上游版本。

内部检索入口为 `bm25_search()`，本轮只有BM25实现，不增加只有一个实现的抽象类或通用Agent编排。用户不能通过接口强行选择未开放provider或查询其他索引。

退出条件：不通过HTML也能完成会话→菜单→搜索→详情→下载；同一问题按身份变化但不越权；并发两个不同用户没有串号；API不能用客户输入扩大授权。

实际完成记录：见 [S4 HTTP报告](reports/S4_CHECKS.json)、[完整性复核](reports/S4_INTEGRITY.json) 与 [验证台账](VALIDATION_STATUS.md#s4实际修改了什么)。20个身份菜单逐ID集合一致；720次文档读取、80次并发请求及160次V1问题检索通过，最终共1,127次真实HTTP请求。57项回归通过；故障/在途会话验证采用受控传输注入，不冒称实际停用许可测试。原SQL、ACL oracle、同义词与旧UI未变，S5尚未执行。

### S5 — 现有HTML改造与ASK AI（M3）

- [x] S5-01 保留原页面样式、左菜单和右阅读区，从源码而非巨大的生成HTML中修改。
- [x] S5-02 移除portal-data中的所有全量正文、预渲染HTML、用户权限、ACL oracle、离线下载载荷。浏览器只保存必要UI状态。
- [x] S5-03 通过本机同源HTTP打开页面，不再依赖file://单文件自给自足；初始页面仅应用壳，不预取全量文档。
- [x] S5-04 用户选择器调用后端开发会话接口；菜单、可见计数和正文来自API，不以JS canDoc结果作为安全判定。
- [x] S5-05 ASK AI面板提供提问、搜索中、结果、无结果、错误和重试状态；首版显示「文档检索 · Elasticsearch」，不展示虚构AI回答。
- [x] S5-06 结果卡支持标题、业务/公开标识、命中摘录、来源/版本，点击定位菜单并打开正文；默认检索最近一次提交文本，不宣称多轮上下文。
- [x] S5-07 原菜单关键词过滤可只过滤已授权导航名称；全文找文档统一用后端搜索，不保留全库本地搜索兜底。
- [x] S5-08 中文/英文/双语切换沿用相同文档权限；保留安全HTML表格、原文和附件缺失提示；结果高亮必须转义并限制标签。
- [x] S5-09 切换身份/注销立即清除正文、原文、元数据、ASK结果和会话历史，取消在途请求；用身份版本/请求序号拒绝旧身份的迟到响应。
- [x] S5-10 测试浏览器返回、深链、复制URL、刷新、不同标签页、下载URL、接口失败；服务端会话变更时不能靠旧前端profile放行。
- [x] S5-11 审查网络与静态文件：旧完整HTML、源SQL、ZIP、expected_access、260份原文目录不可被下载；全量开发自检只在CLI执行或受控管理入口，不发到普通页面。

退出条件：Network能看到真实/api请求；原文不在初始HTML中；用户切换后无旧正文闪回；Document25/99的对照和public跨业务访问可演示；保留桌面/窄屏阅读体验。

实际完成记录：见[浏览器检查](reports/S5_BROWSER_CHECKS.json)、[网络记录](reports/S5_NETWORK.json)、[完整性检查](reports/S5_INTEGRITY.json)及[验证台账](VALIDATION_STATUS.md#s5实际修改了什么)。58项Python回归与6项客户端竞态检查通过；实际页面完成会话、目录、BM25检索、原文和阅读验证。下载HTTP边界/字节已验证，浏览器原生保存及直接下载URL导航仍列未验证。S6尚未执行。

### S6 — 更新、评测与本地MVP验收（M4）

- [x] S6-01 冻结用户审核的32条问题、逐身份参考和3条无答案样例，覆盖中文/英文/术语/缩写/同名跨业务/HTML表格。
- [x] S6-02 金标按身份和开放快照计算，不可访问参考不计漏召回；权限oracle与相关性golden保持独立。
- [x] S6-03 用户确认每类身份Hit@5≥90%，实测96.3%–100%；记录MRR@5和已标注参考Recall@5，未修改金标或调参换取通过。
- [x] S6-04 无答案3×5组合均有文档返回，单列限制并补充原文确认提示；未设未经标定的BM25阈值、未跨权限补足K。
- [x] S6-05 记录480次单并发检索/API的p50/p95、25次暖机和硬件资源，不作生产延迟承诺。
- [x] S6-06 独立合成索引上实测正文/ACL更新、业务撤权、撤回、删除、重复导入、部分写入后失败重试及ES容器重启。
- [x] S6-07 手工完整快照＋差异报告，检查正式关系及完整性声明；删除单独确认，不只依赖UpdatedTime。
- [x] S6-08 维护窗口先关闭读权限、整批写入比对后再发布当前菜单/profile/generation；失败不暴露混合批次。
- [x] S6-09 成功确认后旧会话失效，新请求按新DLS读取；人工同步窗口及前端15秒核对周期已说明，已有版本保护继续拒绝迟到结果。
- [x] S6-10 正文回滚保留最新ACL、授权、撤回和删除；旧物理索引、私有文件、raw及download旁路验证拒绝。
- [x] S6-11 日志/匿名/生产mock/原文净化和用户写入拒绝已核查；许可过期为离线故障验证，未冒称真实自然到期测试。
- [x] S6-12 提供自动验收串联脚本、验证台账和后续入口；离线、ES、HTTP及浏览器证据分别记录。

退出条件：从受控源数据可以复现环境→导入→身份→检索→页面；安全失败项为零；相关性与性能有实际报告；生产认证与发布语义仍清楚标为未完成。

实际记录：[S6评测](reports/S6_EVALUATION.json)、[变更与恢复](reports/S6_CHANGES.json)、[浏览器](reports/S6_BROWSER_CHECKS.json)、[完整性](reports/S6_INTEGRITY.json)及[验证台账](VALIDATION_STATUS.md#s6实际修改了什么)。66项Python回归、7项客户端检查通过；变更最终运行737次HTTP和199次用户ES检查；重启后20身份DLS和1,127次HTTP复查通过。真实业务索引未切换，变更实测明确限定于隔离合成数据。

## 6. 三个核心验收场景

### 场景A：同一句问题，不同受控资料

问题「合同是如何生成的？」：
- Partner-NCE：Document25有资格，Document99不可读。
- Partner-OSPA：Document99有资格，Document25不可读。
- MSFT/21V/Partner双业务：两者均有资格。

自然语言TopK是否包含允许的目标文档是相关性测试；按ID读取拒绝与完整集合比较是安全测试。不要把二者混成一个断言。

### 场景B：Public跨业务

Partner-OSPA可读NCE public Document3；Partner-NCE可读OSPA public Document81。页面的business_scope可主动缩小展示范围，但不能被误认为用户业务权限，也不能扩大ACL。

### 场景C：切换用户时的异步竞态

用OSPA用户打开Document99并发起检索，模拟延迟响应，立即切换到NCE用户。旧请求到达后不得恢复99的标题、正文、摘录、元数据或下载状态。新API调用仍要独立授权；前端取消请求不是服务器授权的替代。

## 7. 关键风险与应对

|风险|处置|
|---|---|
|旧包混用Excel/旧用户/旧ID|S0只认SQL快照与互斥角色v2|
|浏览器仍包含全量知识|S5移除嵌入数据，限制静态目录，按需API读取|
|ES始终看到管理员|S3实际身份检查；S4请求级ES身份映射|
|内部角色绕过菜单授权|DLS仍检查allowed_role_ids，另建角色专属合成反例|
|DLS许可不可用或到期|S1检查并记录，明确失败；不静默降级|
|菜单泄露隐藏文档/上级标题|后端裁剪并使用不含敏感名称的结构占位|
|ES异常时回退本地全库|禁止；展示可重试错误|
|只有UpdatedTime增量漏撤权|一致性关系快照比较，分别同步ACL和内容|
|同义词导致错误业务展开|审核配置与回归，不展开ACL字段|
|把实验开放称为已发布|保留unverified，独立记录lab approval|
|伪造20种权限|承认5类身份/3个源数据集合，额外角色反例单独测试|

## 8. 后续演进：不要反向扩大本轮范围

|后续工作|启动依据|必须保持不变的边界|
|---|---|---|
|术语/同义词持续调优|真实BM25失败案例|权限不参与同义词推断，配置可回归|
|章节级召回|长文小节难以被检索或证据定位不足|TopK仍按逻辑Document去重，所有片段继承ACL|
|多语言向量检索|口语改写/跨语言问题在词项检索中漏召回|每条向量候选链路同样先限制授权范围|
|BM25+向量混合|两路召回有经评测的互补增益|统一doc_key、权限前置、受控候选融合|
|Rerank|正确文档在候选里但排名不佳|模型只接收已授权候选|
|真正RAG回答|检索质量达标且有生成评测/引用要求|另设生成模块，回答引用与原文访问仍受权|
|真实上游认证与同步|拿到稳定接口/会话与撤权契约|替换IdentityProvider和SourceAdapter，不重写搜索接口|

对于ES近邻检索，后续应专门验证knn.filter等预过滤路径和所选版本的DLS支持；不是先全库取K再剔除越权项。混合检索不必额外引入另一套向量数据库，具体选择等评测与部署约束明确后再决定。

真实认证与数据同步属于独立集成工作，不必等到向量检索或RAG之后，但不是当前本地MVP的前置条件。

## 9. 给Codex的执行规则（下一轮实施时采用）

1. 先读ROADMAP和现有代码，报告计划中的文件改动；每次只执行指定阶段，不一次重写全部模块。
2. 每个阶段先确认输入与断言，再实现最小功能，真实运行测试，记录结果和下一阶段入口。
3. 既有测试报告只能证明旧产物的已测范围；不得把其PASS复制成新环境已通过。
4. 不改原SQL，不补造授权，不把expected_access作为搜索实现，不把测试金标当知识入库。
5. 不关闭认证/TLS/DLS，不使用管理员搜索替代用户验证，不打印密钥；不自动销毁数据卷。
6. DLS、身份或读取接口失败时，暂停依赖阶段；不要先做出好看的页面再掩盖后台问题。
7. 不实现本轮范围外的向量/Agent/聊天生成，不接外部用户组查询，不重写ACN或原Demo。
8. 在进入S5前，S3的权限集合检查与S4的API链路必须真实通过。

S0–S6本地MVP已完成。后续由用户单独指定V2问题/无答案策略或生产认证与上游同步契约，不自动追加阶段；继续按实际验证结果更新台账，不把规划或旧报告当成当前证据。

## 10. 依据与参考

### 已有来源

- `playbook-sql-extract/reports/REVIEW.md`：§2–5确认130篇、菜单关联、角色同权、76/26/28分布和格式边界；§6–9说明发布和同步未决项。
- `Playbook_Mock_Users_20_v2/README.md`、`fixtures/mock_users_effective.json`、`fixtures/expected_access.json`：互斥角色、内部业务规范化、20用户及逐ID期望；oracle仅ACL，不等于搜索效果。
- `Playbook_Portal_HTML_v1/README.md`、`src/app.js`及生成HTML：当前静态全量嵌入、JS授权、本地关键词搜索与清理渲染，非真实身份/DLS。
- `playbook-es-lab/docs/VALIDATION_STATUS.md`：旧Excel基线和未执行的Docker/DLS/API状态，不作为新版通过证据。
- `UserAuthenticationInfo.cs`：角色/业务字段和ToClaims输出；内部默认业务补齐不是该DTO已实现行为。
- 用户最近确认的业务规则：单角色互斥、内部默认双业务、Partner显式业务、公开例外。

### 官方技术依据（2026-09-28查阅）

这些来源支持技术选择，不用于替代上游权限业务语义：

- Elastic — Controlling access at the document and field level：DLS、用户metadata模板、只读边界、角色OR合并。
  https://www.elastic.co/docs/deploy-manage/users-roles/cluster-or-deployment-auth/controlling-access-at-document-field-level
- Elastic — Start a single-node cluster in Docker：Docker部署与资源要求。
  https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-docker-basic
- Elastic — Manage your license in a self-managed cluster：Basic、30日trial、许可有效期。
  https://www.elastic.co/docs/deploy-manage/license/manage-your-license-in-self-managed-cluster
- Elastic — Create or update users：native user和metadata配置。
  https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-security-put-user
- Elastic — Multi-match query：多字段检索和字段权重。
  https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-multi-match-query
- Elastic — Synonym graph token filter：搜索分析器中的多词同义词。
  https://www.elastic.co/docs/reference/text-analysis/analysis-synonym-graph-tokenfilter
- Elastic — Knn query：预过滤与候选检索。
  https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query
- Elastic — The refresh parameter：写入后搜索可见性。
  https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter
- OpenAI — Custom instructions with AGENTS.md：Codex项目指令。
  https://developers.openai.com/codex/guides/agents-md/
