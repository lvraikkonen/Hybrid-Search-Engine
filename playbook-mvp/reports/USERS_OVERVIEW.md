# 20 个模拟用户（角色严格互斥）

以下仅为当前 SQL 的 ACL 可见集合；不是已发布数量、检索命中数或 DLS 测试结果。

|用户ID|用户名|唯一角色|有效业务权限|ACL允许|ACL拒绝|
|---|---|---|---|---:|---:|
|mock-001|pb_msft_01|MSFT|NCE+OSPA|130|0|
|mock-002|pb_msft_02|MSFT|NCE+OSPA|130|0|
|mock-003|pb_msft_03|MSFT|NCE+OSPA|130|0|
|mock-004|pb_msft_04|MSFT|NCE+OSPA|130|0|
|mock-005|pb_21v_01|21V|NCE+OSPA|130|0|
|mock-006|pb_21v_02|21V|NCE+OSPA|130|0|
|mock-007|pb_21v_03|21V|NCE+OSPA|130|0|
|mock-008|pb_21v_04|21V|NCE+OSPA|130|0|
|mock-009|pb_partner_nce_01|Partner|NCE|102|28|
|mock-010|pb_partner_nce_02|Partner|NCE|102|28|
|mock-011|pb_partner_nce_03|Partner|NCE|102|28|
|mock-012|pb_partner_nce_04|Partner|NCE|102|28|
|mock-013|pb_partner_ospa_01|Partner|OSPA|104|26|
|mock-014|pb_partner_ospa_02|Partner|OSPA|104|26|
|mock-015|pb_partner_ospa_03|Partner|OSPA|104|26|
|mock-016|pb_partner_ospa_04|Partner|OSPA|104|26|
|mock-017|pb_partner_both_01|Partner|NCE+OSPA|130|0|
|mock-018|pb_partner_both_02|Partner|NCE+OSPA|130|0|
|mock-019|pb_partner_both_03|Partner|NCE+OSPA|130|0|
|mock-020|pb_partner_both_04|Partner|NCE+OSPA|130|0|

20 个不同身份、5 种角色/业务组合、3 种实际 ACL 文档集合。同类用户拥有相同 ACL，未引入个人或组织级限制。
