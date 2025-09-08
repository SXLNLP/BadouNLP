#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
創建完整的小米公司知識圖譜
"""

from py2neo import Graph

def create_complete_xiaomi_graph():
    """創建完整的小米公司知識圖譜"""
    try:
        # 連接到Neo4j
        graph = Graph("neo4j://localhost:7687", auth=("neo4j", "admin852"))
        print("✅ 成功連接到Neo4j數據庫！")
        
        # 清空數據庫
        print("🗑️  清空數據庫...")
        graph.run("MATCH (n) DETACH DELETE n")
        
        # 創建完整的知識圖譜
        print("🔨 創建完整的小米知識圖譜...")
        
        # 創建所有節點
        cypher_statements = [
            # 主要實體節點
            "CREATE (xiaomi:公司 {NAME:'小米集團', 中文名:'小米集團', 英文名:'Xiaomi Corporation', 成立時間:'2010年4月', 創始人:'雷軍', 總部地址:'北京市海淀區', 公司類型:'科技公司', 主要業務:'智能手機、IoT設備、互聯網服務', 上市時間:'2018年7月9日', 上市地點:'香港交易所', 股票代碼:'01810'})",
            "CREATE (leijun:人物 {NAME:'雷軍', 中文名:'雷軍', 英文名:'Lei Jun', 出生日期:'1969年12月16日', 出生地:'湖北省仙桃市', 職業:'企業家、投資人', 畢業院校:'武漢大學', 主要成就:'小米集團創始人、金山軟件董事長', 擔任職務:'小米集團董事長兼CEO'})",
            "CREATE (phone:產品 {NAME:'小米手機', 中文名:'小米手機', 英文名:'Xiaomi Phone', 產品類型:'智能手機', 品牌:'小米', 首次發布:'2011年8月', 主要特點:'性價比高、MIUI系統', 目標用戶:'年輕用戶群體'})",
            "CREATE (miui:產品 {NAME:'MIUI', 中文名:'MIUI', 英文名:'MIUI', 類型:'操作系統', 基於:'Android', 開發商:'小米', 首次發布:'2010年8月', 更新頻率:'每週更新', 特色功能:'主題商店、安全中心'})",
            "CREATE (ecosystem:其他 {NAME:'小米生態鏈', 中文名:'小米生態鏈', 英文名:'Xiaomi Ecosystem', 類型:'投資平台', 成立時間:'2013年', 主要投資:'智能硬件、生活用品', 投資策略:'參股不控股', 成功案例:'華米科技、石頭科技'})",
            "CREATE (beijing:地點 {NAME:'北京市海淀區'})",
            "CREATE (hongkong:金融機構 {NAME:'香港交易所'})",
            "CREATE (wuhan:教育機構 {NAME:'武漢大學'})",
            "CREATE (android:其他 {NAME:'Android'})",
            "CREATE (xiaomi_brand:其他 {NAME:'小米'})"
        ]
        
        # 執行創建節點的語句
        for i, cypher in enumerate(cypher_statements):
            try:
                graph.run(cypher)
                print(f"  ✅ 創建節點 {i+1}/{len(cypher_statements)}")
            except Exception as e:
                print(f"  ❌ 創建節點 {i+1} 失敗: {e}")
        
        # 創建所有關係
        print("\n🔗 創建關係...")
        relationship_statements = [
            # 公司相關關係
            "MATCH (x:公司 {NAME:'小米集團'}), (l:人物 {NAME:'雷軍'}) CREATE (x)-[:創始人]->(l)",
            "MATCH (x:公司 {NAME:'小米集團'}), (b:地點 {NAME:'北京市海淀區'}) CREATE (x)-[:總部]->(b)",
            "MATCH (x:公司 {NAME:'小米集團'}), (p:產品 {NAME:'小米手機'}) CREATE (x)-[:生產]->(p)",
            "MATCH (x:公司 {NAME:'小米集團'}), (m:產品 {NAME:'MIUI'}) CREATE (x)-[:開發]->(m)",
            "MATCH (x:公司 {NAME:'小米集團'}), (e:其他 {NAME:'小米生態鏈'}) CREATE (x)-[:投資]->(e)",
            
            # 人物相關關係
            "MATCH (l:人物 {NAME:'雷軍'}), (x:公司 {NAME:'小米集團'}) CREATE (l)-[:創立]->(x)",
            "MATCH (l:人物 {NAME:'雷軍'}), (w:教育機構 {NAME:'武漢大學'}) CREATE (l)-[:畢業於]->(w)",
            
            # 產品相關關係
            "MATCH (p:產品 {NAME:'小米手機'}), (m:產品 {NAME:'MIUI'}) CREATE (p)-[:操作系統]->(m)",
            "MATCH (m:產品 {NAME:'MIUI'}), (a:其他 {NAME:'Android'}) CREATE (m)-[:基於]->(a)",
            "MATCH (m:產品 {NAME:'MIUI'}), (x:其他 {NAME:'小米'}) CREATE (m)-[:開發商]->(x)"
        ]
        
        # 執行創建關係的語句
        for i, cypher in enumerate(relationship_statements):
            try:
                graph.run(cypher)
                print(f"  ✅ 創建關係 {i+1}/{len(relationship_statements)}")
            except Exception as e:
                print(f"  ❌ 創建關係 {i+1} 失敗: {e}")
        
        print("✅ 完整的小米知識圖譜創建完成！")
        
        # 驗證結果
        print("\n🔍 驗證圖譜...")
        
        # 檢查節點數量
        node_count = graph.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
        print(f"  節點數量: {node_count}")
        
        # 檢查關係數量
        rel_count = graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
        print(f"  關係數量: {rel_count}")
        
        # 檢查標籤
        labels = graph.run("CALL db.labels() YIELD label RETURN collect(label) as labels").data()[0]['labels']
        print(f"  標籤: {labels}")
        
        # 測試查詢
        print("\n🧪 測試查詢...")
        
        # 測試創始人查詢
        founder = graph.run("MATCH (n:公司 {NAME:'小米集團'})-[:創始人]->(m) RETURN n.NAME as company, m.NAME as founder").data()
        print("創始人查詢:")
        for f in founder:
            print(f"  {f['company']} 的創始人是 {f['founder']}")
        
        # 測試總部查詢
        hq = graph.run("MATCH (n:公司 {NAME:'小米集團'})-[:總部]->(m) RETURN n.NAME as company, m.NAME as hq").data()
        print("總部查詢:")
        for h in hq:
            print(f"  {h['company']} 的總部在 {h['hq']}")
        
        # 測試生產查詢
        production = graph.run("MATCH (n:公司 {NAME:'小米集團'})-[:生產]->(m) RETURN n.NAME as company, m.NAME as product").data()
        print("生產查詢:")
        for p in production:
            print(f"  {p['company']} 生產 {p['product']}")
        
        # 測試開發商查詢
        developer = graph.run("MATCH (n:產品 {NAME:'MIUI'})-[:開發商]->(m) RETURN n.NAME as product, m.NAME as developer").data()
        print("開發商查詢:")
        for d in developer:
            print(f"  {d['product']} 的開發商是 {d['developer']}")
        
        print("\n🎉 圖譜創建和驗證完成！")
        print("您可以在Neo4j瀏覽器中查看圖譜：http://localhost:7474")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    create_complete_xiaomi_graph()

