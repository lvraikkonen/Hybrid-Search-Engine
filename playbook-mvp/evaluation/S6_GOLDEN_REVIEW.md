# S6 V1 相关性与无答案集审核稿

状态：用户已确认草案及每类身份 Hit@5 ≥ 90% 目标；标注冻结，检索结果另行实测。

32条问题和参考ID沿用已批准V1；以下标注依据源文档，未使用本轮搜索排名。五类代表身份分别为MSFT、21V、Partner-NCE、Partner-OSPA、Partner双业务。

等级2=源文档直接相关；等级1=只提供相关外链/背景。未标注文档为未知，不默认判为不相关。核心命中评价采用等级≥1；目标为每类身份可访问核心问题Hit@5≥90%，同时记录MRR@5和已标注参考集Recall@5；不宣称穷尽全库相关文档或总体召回率。

无可访问参考的身份/问题不计入核心命中分母，单列为无可访问已标注参考，不能直接称为全库无答案。

| 问题 | 参考文档 / 等级 | 具有可访问参考的身份 | 源依据（节选） |
|---|---|---|---|
| Q01 NCE 的合同文件如何生成，使用哪些系统？ | 25《合同的生成》 / 2 | msft, 21v, partner_nce, partner_both | 通过审批的信息表将提交至微软CE用以生成合同文件。 / 相关方：合作伙伴客户经理、客户、微软合作伙伴经理、微软CE、21Vianet合约管理部、法务部。 / 相关系统： / NCE业务操作平台（https://nce.21vbluecloud.com），点击这里下载 NCE业务操作平台使用手册。 / MSX Proposal , 点击这里下载21VCA-E Proposal Flow指导手册，如遇问题，请遵循标准支持流程，在MSX Proposal 里提交工单。（仅限微软 |
| Q02 OSPA Direct 和 Indirect 模式分别由谁生成合同？ | 99《合同的生成》 / 2 | msft, 21v, partner_ospa, partner_both | Direct Model: 客户确认购买产品后，微软销售联系微软CE生成合同文件。 / Indirect Model: 客户确认购买产品后，合作伙伴/微软CE出合同，生成合同文件。 / 相关系统：Hermes、E-Sign世纪互联蓝云电子签系统。 |
| Q03 OSPA Playbook 的访问权限如何划分？ | 148《OSPA Playbook访问权限》 / 2 | msft, 21v, partner_ospa, partner_both | Catalogue[1]：1 ／ Catalogue[2]：由世纪互联运营的微软智能云 Microsoft Cloud Products Operated by 21Vianet ／ Title[3]：1.1 ／ Title[4]：由世纪互联运营的微软智能云 Microsoft Cloud Products Operated by 21Vianet ／ Customer[5]：✔️ ／ Partner[6]：✔️ ／ Sales[7]：✔️ / Catalog |
| Q04 21VCA-E Playbook 的哪些栏目可供合作伙伴查看？ | 63《21VCA-E Playbook访问权限》 / 2 | msft, 21v, partner_nce, partner_both | Catalogue[1]：1 ／ Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E ／ Title[3]：1.1 ／ Title[4]：21VCA-E新商务模式介绍 Introduction to 21VCA-E New Commerce ／ Customer[5]：✔️ ／ Partner[6]：✔️ ／ Sales[7]：✔️ / Catalogue[1]：1 ／ Catalogue[2]：21VCA-E简介 I |
| Q05 世纪互联客户协议包含什么内容？ | 14《世纪互联客户协议》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 由世纪互联运营的 Microsoft Azure 上的新商务模式（即“21VCA-E”）以一种简化高效的数字化流程取代世纪互联在线服务高级协议 (OSPA)协议繁杂的签约流程、续约周期及文书工作。世纪互联客户协议（“21Vianet CA”）的长期有效特性大幅减少客户在协议期内对产品进行调整所需花费的流程时间，同时，客户可以灵活选择多家合作伙伴进行采购。通过全新商务模式购买由世纪互联运营的Microsoft Azure，将满足客户所在组织对降本增效的进一步 |
| Q06 世纪互联在线服务高级协议是什么？ | 88《世纪互联在线服务高级协议》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 注册于中国大陆地区且有效存续的客户实体，可通过签署世纪互联在线服务高级协议（OSPA）购买由世纪互联运营的 Azure、M365、D365 和 Power Platform 在线服务。 / 💡特别说明： / 对于计划订阅Microsoft Azure云服务的客户——无论是新签客户或OSPA即将到期的客户，请选择 “新商务模式”（21VCA-E）签约，具体操作指南请参照21VCA-E Playbook。 / 对于已有OSPA Azure合约的客户，希望继续使用由世纪互联 |
| Q07 NCE 信用审核需要怎样操作？ | 23《信用审核Credit check》 / 2 | msft, 21v, partner_nce, partner_both | 什么是Credit Check / 新商务模式（21VCA-E）模式下，默认支持30标准账期支付订单金额，对于是否可以支持对应订单30天账期，在下单前需要对合作伙伴的实时可用担保额度、欠款等信用情况进行审核。 / 如何申请Credit Check / 21VCA-E采用间接模式进行销售，在信审环节，需要合作伙伴向21Vianet 提交对应订单的信用申请，审核通过合作伙伴会收到通知，订单进入下单步骤。 / 信用申请系统 / 合作伙伴信用管理系统 / 信用审核流程概览 / 世纪互联客户协 |
| Q08 Direct Model OSPA 如何提交信用审核申请？ | 97《Direct Model OSPA信用审核申请》 / 2 | msft, 21v, partner_ospa, partner_both | 适用范围： / 用户： 微软客户经理。 / 适用场景： 由世纪互联运营的Microsoft Azure、Microsoft 365、Microsoft Dynamics 365 和Microsoft Power Platform在线服务产品的OSPA Direct Model订单所对应的信审申请。 / 新用户注册与登录： / 此信审审批模块搭建在世纪互联的 NCE业务操作平台（即，NCE平台）中，对于现有NCE 平台的用户，世纪互联已为您开通信审模块的使用权，本次无需重新 |
| Q09 Indirect Model OSPA 如何申请信用审核？ | 98《Indirect Model OSPA信用审核申请》 / 2 | msft, 21v, partner_ospa, partner_both | 适用范围： / 用户: 合作伙伴。 / 适用场景： 由世纪互联运营的Microsoft Azure、Microsoft 365、Microsoft Dynamics 365 和Microsoft Power Platform在线服务产品的OSPA Indirect Model 订单所对应的信审申请。 / 信用申请系统 / URL：https://pcm.21vbluecloud.com/ / 可联系世纪互联 PAM团队获取《合作伙伴信用管理系统用户手册》以获取更为详细的说明。 |
| Q10 How do I add a billing account owner? | 38《如何新增计费账户所有者(Billing account owner)》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | Step 1: Log in to Azure Portal, search for Azure Entra ID, and create a new user in Azure Entra ID. / 1. After entering the Entra ID, click on the user / 2. Click on New User - New User / 3. Fill in the relevant information of the new u |
| Q11 如何新增 Billing Profile 的角色账户？ | 39《如何新增billing profile角色账户》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 第一步：在Entra ID中创建一个新的账户。 / 具体步骤请参考：如何新增计费账户所有者(Billing account owner) 中的 第一步 / 第二步:进入 成本管理+计费，点击左侧的 计费对象信息，点击 计费对象信息名称 / 点击 访问控制 / 点击 添加 > 选择 计费对象信息所有者，在下拉列表中选择新创建的账户,点击页面下方的添加. / 添加完成后，刷新页面，即可显示出新添加的管理员账户。 |
| Q12 How do I create an invoice section? | 41《如何创建发票科目（invoice section）》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | Each billing object information defaults to including an Invoice Section. You can create multiple billing accounts to group the costs in the billing statements for billing object information. Users with the Invoice Section role ca |
| Q13 Azure RI 预留实例是什么？ | 42《什么是Azure预留（RI）？》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | Azure 预留项（RI）通过承诺多种产品的一年期或三年期计划，帮助你节省资金。 通过承诺预付，能够以折扣价购买所用资源。 预留可显著降低资源的成本，最多可减少至即用即付价格的 72%。 预订提供计费折扣，并且不会影响资源的运行时状态。 购买预留项后，折扣会自动应用于匹配的资源。 |
| Q14 如何购买预留实例？ | 44《购买预留实例（RI）》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 备注：若要购买预留，需要是订阅的所有者或者订阅级别的预留购买者。 / 通过Portal购买RI: / 1. 登录到Azure Portal / 2. 在页面上方搜索 成本管理+计费 / 3. 点击页面左侧的 预留+混合权益 / 4. 点击 添加 按钮 / 5. 进入到购买页面，选择 范围，订阅ID，机型，期限，区域以及账单频率，并点击 添加到购买车 和 View Cart 按钮 / 6. 在此页面可以添加更多RI产品，或更改账单频率和编辑购买此机型的数量，然后点击 下一步 查看+购买 |
| Q15 RI 的续订在哪里设置？ | 47《设置RI的续订》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 若要续订预留，必须满足以下条件： / •你必须是现有预留的所有者。 / •如果预留局限于单个订阅或资源组，你必须是订阅的所有者。 / •如果订阅具有共享范围或管理组范围，你必须是订阅的所有者。 / 设置RI续订的操作步骤： / 1. 登录到 Azure Portal / 2. 在页面上方的搜索框中搜索 成本管理+计费 / 3. 点击左侧的 预留+混合权益 / 4. 选择需要续订的预留实例 / 5. 点击左侧的 续订，并点击 保存。 / 更多关于 RI自动续订的内容可参考：自动续订 Azure 预 |
| Q16 如何查看 RI 利用率？ | 49《如何查看RI的利用率》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 注意：若要查看预留利用率，必须拥有对预留项的 Azure RBAC 访问权限，或者拥有可管理所有 Azure 订阅和管理组的访问权限。 / 1. 登录到 Azure Portal / 2. 在页面上方的搜索框中搜索 成本管理+计费 / 3. 点击左侧的 预留+混合权益 / 4. 在RI的右侧可以显示利用率百分比。 / 5. 点击 其中一个RI 进入到详情页面，在 概述 页面可以显示利用率详细的信息。 |
| Q17 What is an Azure Savings Plan? | 50《什么是Azure 节省计划（Savings Plan）》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | Azure savings plan for compute is a flexible pricing model that provides savings up to 65 percent off pay-as-you-go pricing when you commit to spend a fixed hourly amount on compute services for one or three years. Savings plan di |
| Q18 Savings Plan 覆盖哪些计算产品？ | 52《Savings Plan涵盖哪些计算产品》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 有两个节省计划 - 用于计算的节省计划和数据库的节省计划。 计算储蓄计划作为 1 年或 3 年的承诺提供，而数据库储蓄计划作为 1 年承诺提供。 / 节省计算计划适用于一组广泛的 Azure 计算服务的基础结构成本，包括： / Azure 虚拟机 / Azure App 服务 / Azure Functions 高级计划 / Azure 容器实例 / Azure 专用主机 / Azure 容器应用 / 适用于企业的 Azure Spring Apps / 注意： / 虚拟机不包括 Av1 系列。 |
| Q19 如何购买节省计划？ | 53《如何购买Savings Plan》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 谁可以购买Savings Plan? / 拥有计费对象信息参与者权限和更高权限的客户可以从“成本管理 + 计费”>“节省计划”体验购买节省计划。 不需要特定于订阅的权限。 / 在计费对象信息的至少一个订阅中拥有订阅所有者或节省计划购买者角色的用户可以从“主页”>“节省计划”购买节省计划。 / 更多关于如何购买节省计划的内容请参考：购买Azure节省计划 |
| Q20 Savings Plan 能否退款？ | 55《Savings Plan的退款政策》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 所有节省计划购买后均不可变更。 购买节省计划后，无法取消或交换预留，Savings Plan不支持退款。 |
| Q21 如何理解摊销成本账单？ | 59《摊销成本账单》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 一、实际成本 / 实际成本 - 实际成本是指企业在购买Azure服务时实际支付的金额，它通常与购买行为直接相关，并体现在购买当月的账单上。 / 提供要与每月帐单对帐的数据。 此数据包括预留项购买成本和预留项应用详细信息。 使用此数据可以知道哪个订阅、资源组或资源在特定的日期收到了预留折扣。 收到预留折扣的用途的 EffectivePrice 为零。 / 显示帐单上的购买费用。 例如，如果在 2022 年 1 月以 1200 美元的价格购买了一年期预留，并且提前支付了全 |
| Q22 Azure 订阅有哪些状态？ | 60《订阅的状态》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 在 Azure 门户订阅区域中，订阅可能会是以下几种状态。 / 活动：Azure 订阅处于活动状态。 可以使用订阅来部署新资源及管理现有资源。 / 已禁用：Azure 订阅已禁用，不再可用于创建或管理 Azure 资源。 在此状态下，将会解除分配虚拟机，释放临时 IP 地址，存储将变为只读，并禁用其他服务。 订阅可能会保持禁用状态90 天，具体取决于订阅类型。 之后，它将被永久删除。 / 已删除：已删除 Azure 订阅及所有资源/数据。 |
| Q23 NCE 业务下怎样创建订阅？ | 61《创建订阅》 / 1 | msft, 21v, partner_nce, partner_ospa, partner_both | 关于如何创建订阅的步骤可参考：创建 Microsoft 客户协议订阅 - Azure Cost Management + Billing |
| Q24 NCE 的订阅如何取消？ | 62《取消订阅》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 提示： / 在取消订阅之前请做好数据备份。 / 订阅所有者的角色可以取消订阅。 / 取消订阅的操作步骤： / 1. 在 Azure 门户中，导航到“成本管理 + 计费” / 2. 在左侧菜单中，选择“订阅”或“Azure 订阅” / 3. 选择要取消的订阅。 / 4. 在页面顶部，选择“取消”。 / 5. 如果有任何与订阅关联的资源，则会在页面上显示。 在页面顶部，选择“取消订阅” / 6. 选择 取消原因。 / 7. 如果你有支持计划，但没有其他订阅在使用它，请选择“关闭自动续订”。 如果有其 |
| Q25 OSPA 的订阅如何取消？ | 141《取消订阅》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 订阅是 Azure 客户创建服务的资源池，在部署服务前您需要创建订阅，使用中可以编辑订阅属性信息，包括订阅名称，服务管理员及协同管理员。也可以根据您的业务需求随时选择取消订阅，取消后您将无法访问该订阅中之前部署的服务和资源，系统会即时结束该订阅的计费。 / 本文档为您介绍了取消订阅的操作步骤及操作中需注意的问题。 / 用户需要使用帐户所有者账号登陆 Azure 门户进行设置。 / 取消订阅前准备工作 / 备份数据。例如，如果有存储在 Azure 或 SQL 的数据，请下载 |
| Q26 怎样重新激活已取消的订阅？ | 142《激活已取消的订阅》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 订阅是企业协议用户部署 Azure 服务的资源池，您所有的服务及资源均与某一订阅相关联。如发生订阅禁用，会导致订阅中已部署的资源服务被暂停，临时存储中资源将被释放。您仍然可以登陆门户网站查看资源，但无法进行编辑操作。禁用之日起 90 天内，您可以联系世纪互联客户支持帮您重启订阅。超出 90 天后，您的订阅将无法重启。 / 企业协议用户会有以下两种情况导致订阅被禁用。 / 未按时支付超额账单/年度订单合约被停 / 订阅被帐户所有者取消 / 超额/年度订单未支付 / 如您在合约产 |
| Q27 How do I delete an account in the enterprise portal? | 128《如何删除账户》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | Accounts can be flexibly managed in the Enterprise Portal, such as deleting an account: / Log in to the Enterprise Portal, select the account you wish to delete from the account options, and click the "Delete" option. |
| Q28 如何从企业门户下载使用量报告？ | 130《如何下载使用量报告》 / 2 | msft, 21v, partner_nce, partner_ospa, partner_both | 企业协议用户如何下载使用量报告 / 企业协议用户可以登陆企业门户查看合约中历史使用量和当前剩余金额信息。系统中使用量最多有五天延迟，具体您可以通过下载使用量功能查看明细截至日期。企业门户为客户提供了三种查看方式：每月报告下载，高级报告下载，API 访问密钥。在每月报告和高级报告中，提供了截止到报告中日期为止每个部门，帐户和订阅每天消耗的服务内容，服务用量及对应的计费金额。 / 用户可参考以下操作方式下载报告： / 1. 登陆 Azure 门户，在左边视图中选择"报表" |
| Q29 NCE 的 CICR 流程怎么操作？ | 19《CICR》 / 2 | msft, 21v, partner_nce, partner_both | CICR，指针对已生效的协议“修改客户或合作伙伴联系人的流程（Contact Information Change Request）”。 / A. 覆盖场景 / CICR包括两种场景： / 1.场景一： / 修改世纪互联客户协议（21VCA）中的客户账户负责人、客户公司地址及合作伙伴应收账款联系人。 / 2.场景二： / 修改客户Azure Portal中，计费账户-属性里的联系人邮箱。此邮箱会收到系统层面的维护通知邮件、价格调整相关的通知邮件。 / 特别注意: / 世纪互联客户协议（2 |
| Q30 OSPA 的 Contact Information Change Request 是什么流程？ | 105《CICR》 / 2 | msft, 21v, partner_ospa, partner_both | CICR： 指“合约联系人变更（Contact Information Change Request）”，针对已生效的协议，客户/合作伙伴提出要变更合同联系人信息。 / 💡注：只适用于变更联系人信息，不能变更公司名称 / A.不同合约需使用不同的版本 / CICR direct 版本 / CICR indirect 版本 / CICR文档模版可以在Hermes首页下载： / B.变更流程 / 1. 文档填写说明 / a. Direct合同 / 申请人按照文档要求，填写完成CICR文档，使用合 |
| Q31 NCE 的实体名称变更 COEN 如何处理？ | 20《COEN》 / 2 | msft, 21v, partner_nce, partner_both | COEN指“公司实体名称变更（Change of Entity Name）”，仅限于客户实体名称在市场监督管理局完成公司实体名称变更登记的场景。若曾用名和变更后的公司实体名称同时使用，并且不能提供市场监督管理局实体变更说明函，则不属于本场景。具体步骤参考如下： / A. 供所需资料 / 邮件提供如下文件盖章扫描件到21Vianet Sales Support team <salessupport@oe.21vianet.com>，21Vianet在收到完整文件后， |
| Q32 NCE 更换合同合作伙伴 COCP 有什么流程？ | 18《COCP》 / 2 | msft, 21v, partner_nce, partner_both | COCP指“将原经销商即付款方合作伙伴换为新的合作伙伴（Change of Contract Partner）”。 如果COCP的需求来自于客户，需要请新的合作伙伴到Hermes首页下载最新版的COCP文档，并填写相关信息。 / 具体参考步骤如下： / A. 新合作伙伴提交信审申请 / 新合作伙伴需要在启动COCP前为客户提交零值订单的信审申请（需填写预估合同金额-未来3个月用量预估）并审批通过。 / 特别注意：新合作伙伴需为已经完成合作伙伴上线流程具有21VCA-E下单 |

## 待确认的无答案问题

- **NA01** 员工年假有多少天，如何申请休假？ — 本快照没有人事休假制度。
- **NA02** What will the weather be in Shanghai tomorrow? — 本快照没有实时天气数据。
- **NA03** 2027年12月中国北部3 Standard_D4s_v5 虚拟机实时每小时价格是多少？ — 现有文档可能介绍价目表入口，但没有指定未来时点的实时机型价格。

无答案样例单独记录返回率、返回ID及摘录；不设置未经标定的BM25分数阈值。返回相关文档不等于这些文档包含所问答案，界面需明确这一点。无答案返回率本轮只测量，不设臆定达标值。

Q23只有外部创建订阅文档链接，按等级1记录，不声称本库包含完整步骤。人工确认后冻结本文件对应JSON指纹；后续改标注须新版本，不能为得到PASS修改金标。

批准原文：确认草案与 90% 目标

记录时间：2026-09-29T23:11:36.875862+00:00；冻结指纹：`81005463989a950e1bb689bcea4daf0dfe7485e8787d2025d8c70a261276e118`。
