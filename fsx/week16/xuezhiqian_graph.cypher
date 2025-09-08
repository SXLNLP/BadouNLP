// 创建核心人物节点
MERGE (xzq:Person {
  name: "薛之谦",
  birthDate: "1983-07-17",
  birthPlace: "中国上海市",
  graduation: "瑞士格里昂酒店管理学院",
  occupation: ["歌手", "音乐制作人", "影视演员", "词曲作者"],
  bloodType: "O型",
  height: "172cm",
  debutYear: 2005
})

// 创建歌曲节点
MERGE (s1:Song {name: "认真的雪", releaseYear: 2006, downloadCount: "2000万次"})
MERGE (s2:Song {name: "演员", theme: "情感哲思", playCount: "超200亿次"})
MERGE (s3:Song {name: "丑八怪", style: "流行"})
MERGE (s4:Song {name: "天外来物", style: "摇滚"})
MERGE (s5:Song {name: "跃", releaseYear: 2025, theme: "海洋生态"})
MERGE (s6:Song {name: "方的言", theme: "方言文化"})

// 创建专辑节点
MERGE (a1:Album {name: "薛之谦", releaseYear: 2006})
MERGE (a2:Album {name: "怪咖", releaseYear: 2018, award: "华人歌曲音乐盛典年度最佳专辑"})

// 创建巡演节点
MERGE (t1:Tour {
  name: "天外来物世界巡回演唱会",
  timeRange: "2021-2024",
  showsCount: 140,
  locations: ["四大洲9国61城"],
  stageTech: ["全息投影", "巨型宇宙飞船装置"],
  boxOffice: "10亿元"
})
MERGE (t2:Tour {name: "万兽之王巡演", announcedYear: 2025})

// 创建奖项节点
MERGE (aw1:Award {name: "第21届东方风云榜最佳唱作人", year: 2014})
MERGE (aw2:Award {name: "第16届全球华语歌曲排行榜最受欢迎男歌手", year: 2016})
MERGE (aw3:Award {name: "腾讯音乐娱乐盛典年度演唱会", year: 2021, relatedWork: "天外来物巡演"})
MERGE (aw4:Award {name: "中国文化产业年度人物", year: 2025})
MERGE (aw5:Award {name: "年度公益影响力艺人", year: 2024})

// 创建综艺/影视节点
MERGE (v1:VarietyShow {name: "火星情报局", role: "嘉宾"})
MERGE (v2:VarietyShow {name: "明日之子", role: "星推官"})
MERGE (m1:Movie {name: "我们的少年时代", type: "电视剧", role: "陶西"})
MERGE (m2:Movie {name: "守村人", type: "公益微电影", year: 2024})

// 创建品牌节点
MERGE (b1:Brand {name: "Dangerous People", type: "潮牌", sales: "年销售额破亿"})
MERGE (b2:Brand {name: "上上谦火锅", type: "餐饮", peakStores: 28})

// 创建机构节点
MERGE (o1:Organization {name: "瑞士格里昂酒店管理学院", type: "院校"})
MERGE (o2:Organization {name: "潮石音乐", type: "公司", foundedYear: 2022})

// 创建公益活动节点
MERGE (pw1:PublicWelfare {name: "捐赠音乐教室", count: 30, coverage: "15省"})
MERGE (pw2:PublicWelfare {name: "河南水灾捐款", amount: "100万元", year: 2021})
MERGE (pw3:PublicWelfare {name: "海洋清洁挑战", result: "清理垃圾超千公斤"})

// 创建关系
// 创作歌曲
MERGE (xzq)-[:CREATED {year: 2006}]->(s1)
MERGE (xzq)-[:CREATED]->(s2)
MERGE (xzq)-[:CREATED]->(s3)
MERGE (xzq)-[:CREATED]->(s4)
MERGE (xzq)-[:CREATED {year: 2025}]->(s5)
MERGE (xzq)-[:CREATED]->(s6)

// 发行专辑
MERGE (xzq)-[:RELEASED {year: 2006}]->(a1)
MERGE (xzq)-[:RELEASED {year: 2018}]->(a2)

// 专辑包含歌曲
MERGE (a1)-[:INCLUDE]->(s1)
MERGE (a2)-[:INCLUDE]->(s3)

// 举办巡演
MERGE (xzq)-[:HELD]->(t1)
MERGE (xzq)-[:HELD]->(t2)

// 获得奖项
MERGE (xzq)-[:WON]->(aw1)
MERGE (xzq)-[:WON]->(aw2)
MERGE (xzq)-[:WON]->(aw3)
MERGE (xzq)-[:WON]->(aw4)
MERGE (xzq)-[:WON]->(aw5)

// 参与综艺/影视
MERGE (xzq)-[:PARTICIPATED_IN]->(v1)
MERGE (xzq)-[:PARTICIPATED_IN]->(v2)
MERGE (xzq)-[:PARTICIPATED_IN]->(m1)
MERGE (xzq)-[:PARTICIPATED_IN]->(m2)

// 创立品牌
MERGE (xzq)-[:FOUNDED]->(b1)
MERGE (xzq)-[:FOUNDED]->(b2)

// 关联机构
MERGE (xzq)-[:JOINED {type: "毕业"}]->(o1)
MERGE (xzq)-[:JOINED {type: "所属"}]->(o2)

// 参与公益
MERGE (xzq)-[:ENGAGED_IN]->(pw1)
MERGE (xzq)-[:ENGAGED_IN]->(pw2)
MERGE (xzq)-[:ENGAGED_IN]->(pw3)
