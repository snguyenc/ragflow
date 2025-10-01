# Licensed under the MIT License
"""
Reference:
 - [LightRag](https://github.com/HKUDS/LightRAG)
 - [MiniRAG](https://github.com/HKUDS/MiniRAG)
"""
PROMPTS = {}

PROMPTS["minirag_query2kwd"] = """---Role---

You are a helpful assistant tasked with identifying both answer-type and low-level keywords in the user's query.

---Goal---

Given the query, list both answer-type and low-level keywords.
answer_type_keywords focus on the type of the answer to the certain query, while low-level keywords focus on specific entities, details, or concrete terms.
The answer_type_keywords must be selected from Answer type pool. 
This pool is in the form of a dictionary, where the key represents the Type you should choose from and the value represents the example samples.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have three keys:
  - "answer_type_keywords" for the types of the answer. In this list, the types with the highest likelihood should be placed at the forefront. No more than 3.
  - "entities_from_query" for specific entities or details. It must be extracted from the query.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
Answer type pool: {{
 'PERSONAL LIFE': ['FAMILY TIME', 'HOME MAINTENANCE'],
 'STRATEGY': ['MARKETING PLAN', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['JANE DOE', 'JOHN SMITH'],
 'FOOD': ['PASTA', 'SUSHI'],
 'EMOTION': ['HAPPINESS', 'ANGER'],
 'PERSONAL EXPERIENCE': ['TRAVEL ABROAD', 'STUDYING ABROAD'],
 'INTERACTION': ['TEAM MEETING', 'NETWORKING EVENT'],
 'BEVERAGE': ['COFFEE', 'TEA'],
 'PLAN': ['ANNUAL BUDGET', 'PROJECT TIMELINE'],
 'GEO': ['NEW YORK CITY', 'SOUTH AFRICA'],
 'GEAR': ['CAMPING TENT', 'CYCLING HELMET'],
 'EMOJI': ['🎉', '🚀'],
 'BEHAVIOR': ['POSITIVE FEEDBACK', 'NEGATIVE CRITICISM'],
 'TONE': ['FORMAL', 'INFORMAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS'],
  'CARD_PRODUCT': ['THẺ TÍN DỤNG SACOMBANK', 'THẺ VISA PLATINUM'],
  'PAYMENT_NETWORK': ['VISA', 'MASTERCARD'],
  'BANK': ['SACOMBANK', 'VIETCOMBANK'],
  'CARD_TIER': ['PLATINUM', 'GOLD'],
  'CASHBACK_FEATURE': ['HOÀN TIỀN 2%', 'CASHBACK KHÔNG GIỚI HẠN'],
  'REWARD_PROGRAM': ['TÍCH ĐIỂM MILES', 'QUÀ TẶNG SINH NHẬT'],
  'INSURANCE_BENEFIT': ['BẢO HIỂM DU LỊCH TOÀN CẦU', 'BẢO HIỂM Y TẾ'],
  'AIRPORT_SERVICE': ['PHÒNG CHỜ CIP', 'FAST TRACK SÂN BAY'],
  'DINING_BENEFIT': ['GIẢM 20% NHÀ HÀNG', 'ƯU ĐÃI F&B'],
  'ANNUAL_FEE': ['PHÍ THƯỜNG NIÊN 1.999.000 VND', 'MIỄN PHÍ NĂM ĐẦU'],
  'INTEREST_RATE': ['LÃI SUẤT 0% 45 NGÀY', 'LÃI SUẤT 2.95%/THÁNG'],
  'CREDIT_LIMIT': ['HẠN MỨC 500 TRIỆU', 'HẠN MỨC TÍN DỤNG CAO'],
  'CURRENCY': ['VND', 'USD'],
  'CUSTOMER_SEGMENT': ['KHÁCH HÀNG VIP', 'SINH VIÊN'],
  'SPENDING_CATEGORY': ['SHOPPING ONLINE', 'DU LỊCH QUỐC TẾ'],
  'MERCHANT_PARTNER': ['TIKI', 'GRAB'],
  'PROMOTION': ['KHUYẾN MÃI TẾT', 'ƯU ĐÃI BLACK FRIDAY'],
  'ELIGIBILITY': ['THU NHẬP TỐI THIỂU 15 TRIỆU', 'TUỔI TỪ 18-65']
}}
################
Output:
{{
  "answer_type_keywords": ["STRATEGY","PERSONAL LIFE"],
  "entities_from_query": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"
Answer type pool: {{
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSONAL LIFE': ['DAILY EXERCISE ROUTINE', 'FAMILY VACATION PLANNING'],
 'STRATEGY': ['NEW PRODUCT LAUNCH', 'YEAR-END SALES BOOST'],
 'SERVICE FACILITATION': ['REMOTE IT SUPPORT', 'ON-SITE TRAINING SESSIONS'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'FOOD': ['GRILLED SALMON', 'VEGETARIAN BURRITO'],
 'EMOTION': ['EXCITEMENT', 'DISAPPOINTMENT'],
 'PERSONAL EXPERIENCE': ['BIRTHDAY CELEBRATION', 'FIRST MARATHON'],
 'INTERACTION': ['OFFICE WATER COOLER CHAT', 'ONLINE FORUM DEBATE'],
 'BEVERAGE': ['ICED COFFEE', 'GREEN SMOOTHIE'],
 'PLAN': ['WEEKLY MEETING SCHEDULE', 'MONTHLY BUDGET OVERVIEW'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
 'GEAR': ['PROFESSIONAL CAMERA EQUIPMENT', 'OUTDOOR HIKING GEAR'],
 'EMOJI': ['📅', '⏰'],
 'BEHAVIOR': ['PUNCTUALITY', 'HONESTY'],
 'TONE': ['CONFIDENTIAL', 'SATIRICAL'],
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY'],
  'CARD_PRODUCT': ['THẺ TÍN DỤNG SACOMBANK', 'THẺ VISA PLATINUM'],
  'PAYMENT_NETWORK': ['VISA', 'MASTERCARD'],
  'BANK': ['SACOMBANK', 'VIETCOMBANK'],
  'CARD_TIER': ['PLATINUM', 'GOLD'],
  'CASHBACK_FEATURE': ['HOÀN TIỀN 2%', 'CASHBACK KHÔNG GIỚI HẠN'],
  'REWARD_PROGRAM': ['TÍCH ĐIỂM MILES', 'QUÀ TẶNG SINH NHẬT'],
  'INSURANCE_BENEFIT': ['BẢO HIỂM DU LỊCH TOÀN CẦU', 'BẢO HIỂM Y TẾ'],
  'AIRPORT_SERVICE': ['PHÒNG CHỜ CIP', 'FAST TRACK SÂN BAY'],
  'DINING_BENEFIT': ['GIẢM 20% NHÀ HÀNG', 'ƯU ĐÃI F&B'],
  'ANNUAL_FEE': ['PHÍ THƯỜNG NIÊN 1.999.000 VND', 'MIỄN PHÍ NĂM ĐẦU'],
  'INTEREST_RATE': ['LÃI SUẤT 0% 45 NGÀY', 'LÃI SUẤT 2.95%/THÁNG'],
  'CREDIT_LIMIT': ['HẠN MỨC 500 TRIỆU', 'HẠN MỨC TÍN DỤNG CAO'],
  'CURRENCY': ['VND', 'USD'],
  'CUSTOMER_SEGMENT': ['KHÁCH HÀNG VIP', 'SINH VIÊN'],
  'SPENDING_CATEGORY': ['SHOPPING ONLINE', 'DU LỊCH QUỐC TẾ'],
  'MERCHANT_PARTNER': ['TIKI', 'GRAB'],
  'PROMOTION': ['KHUYẾN MÃI TẾT', 'ƯU ĐÃI BLACK FRIDAY'],
  'ELIGIBILITY': ['THU NHẬP TỐI THIỂU 15 TRIỆU', 'TUỔI TỪ 18-65']
}}

################
Output:
{{
  "answer_type_keywords": ["DATE AND TIME", "ORGANIZATION", "PLAN"],
  "entities_from_query": ["SpaceX", "Rocket launch", "Aerospace", "Power Recovery"]

}}
#############################
Example 3:

Query: "so sánh chi tiết giúp tôi các loại thẻ liên kết có tính
   năng hoàn tiền?"
Answer type pool: {{
 'PERSONAL LIFE': ['MANAGING WORK-LIFE BALANCE', 'HOME IMPROVEMENT PROJECTS'],
 'STRATEGY': ['MARKETING STRATEGIES FOR Q4', 'EXPANDING INTO NEW MARKETS'],
 'SERVICE FACILITATION': ['CUSTOMER SATISFACTION SURVEYS', 'STAFF RETENTION PROGRAMS'],
 'PERSON': ['ALBERT EINSTEIN', 'MARIA CALLAS'],
 'FOOD': ['PAN-FRIED STEAK', 'POACHED EGGS'],
 'EMOTION': ['OVERWHELM', 'CONTENTMENT'],
 'PERSONAL EXPERIENCE': ['LIVING ABROAD', 'STARTING A NEW JOB'],
 'INTERACTION': ['SOCIAL MEDIA ENGAGEMENT', 'PUBLIC SPEAKING'],
 'BEVERAGE': ['CAPPUCCINO', 'MATCHA LATTE'],
 'PLAN': ['ANNUAL FITNESS GOALS', 'QUARTERLY BUSINESS REVIEW'],
 'GEO': ['THE AMAZON RAINFOREST', 'THE GRAND CANYON'],
 'GEAR': ['SURFING ESSENTIALS', 'CYCLING ACCESSORIES'],
 'EMOJI': ['💻', '📱'],
 'BEHAVIOR': ['TEAMWORK', 'LEADERSHIP'],
 'TONE': ['FORMAL MEETING', 'CASUAL CONVERSATION'],
 'LOCATION': ['URBAN CITY CENTER', 'RURAL COUNTRYSIDE'],
 ,
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY'],
  'CARD_PRODUCT': ['THẺ TÍN DỤNG SACOMBANK', 'THẺ VISA PLATINUM'],
  'PAYMENT_NETWORK': ['VISA', 'MASTERCARD'],
  'BANK': ['SACOMBANK', 'VIETCOMBANK'],
  'CARD_TIER': ['PLATINUM', 'GOLD'],
  'CASHBACK_FEATURE': ['HOÀN TIỀN 2%', 'CASHBACK KHÔNG GIỚI HẠN'],
  'REWARD_PROGRAM': ['TÍCH ĐIỂM MILES', 'QUÀ TẶNG SINH NHẬT'],
  'INSURANCE_BENEFIT': ['BẢO HIỂM DU LỊCH TOÀN CẦU', 'BẢO HIỂM Y TẾ'],
  'AIRPORT_SERVICE': ['PHÒNG CHỜ CIP', 'FAST TRACK SÂN BAY'],
  'DINING_BENEFIT': ['GIẢM 20% NHÀ HÀNG', 'ƯU ĐÃI F&B'],
  'ANNUAL_FEE': ['PHÍ THƯỜNG NIÊN 1.999.000 VND', 'MIỄN PHÍ NĂM ĐẦU'],
  'INTEREST_RATE': ['LÃI SUẤT 0% 45 NGÀY', 'LÃI SUẤT 2.95%/THÁNG'],
  'CREDIT_LIMIT': ['HẠN MỨC 500 TRIỆU', 'HẠN MỨC TÍN DỤNG CAO'],
  'CURRENCY': ['VND', 'USD'],
  'CUSTOMER_SEGMENT': ['KHÁCH HÀNG VIP', 'SINH VIÊN'],
  'SPENDING_CATEGORY': ['SHOPPING ONLINE', 'DU LỊCH QUỐC TẾ'],
  'MERCHANT_PARTNER': ['TIKI', 'GRAB'],
  'PROMOTION': ['KHUYẾN MÃI TẾT', 'ƯU ĐÃI BLACK FRIDAY'],
  'ELIGIBILITY': ['THU NHẬP TỐI THIỂU 15 TRIỆU', 'TUỔI TỪ 18-65']
}}

################
Output:
{{
  "answer_type_keywords": ["CARD_PRODUCT", "CASHBACK_FEATURE", "CARD_TIER"],
  "entities_from_query": ["thẻ liên kết", "hoàn tiền", "cashback"]
}}
#############################
Example 4:

Query: "Where is the capital of the United States?"
Answer type pool: {{
 'ORGANIZATION': ['GREENPEACE', 'RED CROSS'],
 'PERSONAL LIFE': ['DAILY WORKOUT', 'HOME COOKING'],
 'STRATEGY': ['FINANCIAL INVESTMENT', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['ALBERTA SMITH', 'BENJAMIN JONES'],
 'FOOD': ['PASTA CARBONARA', 'SUSHI PLATTER'],
 'EMOTION': ['HAPPINESS', 'SADNESS'],
 'PERSONAL EXPERIENCE': ['TRAVEL ADVENTURE', 'BOOK CLUB'],
 'INTERACTION': ['TEAM BUILDING', 'NETWORKING MEETUP'],
 'BEVERAGE': ['LATTE', 'GREEN TEA'],
 'PLAN': ['WEIGHT LOSS', 'CAREER DEVELOPMENT'],
 'GEO': ['PARIS', 'NEW YORK'],
 'GEAR': ['CAMERA', 'HEADPHONES'],
 'EMOJI': ['🏢', '🌍'],
 'BEHAVIOR': ['POSITIVE THINKING', 'STRESS MANAGEMENT'],
 'TONE': ['FRIENDLY', 'PROFESSIONAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS'],
  'CARD_PRODUCT': ['THẺ TÍN DỤNG SACOMBANK', 'THẺ VISA PLATINUM'],
  'PAYMENT_NETWORK': ['VISA', 'MASTERCARD'],
  'BANK': ['SACOMBANK', 'VIETCOMBANK'],
  'CARD_TIER': ['PLATINUM', 'GOLD'],
  'CASHBACK_FEATURE': ['HOÀN TIỀN 2%', 'CASHBACK KHÔNG GIỚI HẠN'],
  'REWARD_PROGRAM': ['TÍCH ĐIỂM MILES', 'QUÀ TẶNG SINH NHẬT'],
  'INSURANCE_BENEFIT': ['BẢO HIỂM DU LỊCH TOÀN CẦU', 'BẢO HIỂM Y TẾ'],
  'AIRPORT_SERVICE': ['PHÒNG CHỜ CIP', 'FAST TRACK SÂN BAY'],
  'DINING_BENEFIT': ['GIẢM 20% NHÀ HÀNG', 'ƯU ĐÃI F&B'],
  'ANNUAL_FEE': ['PHÍ THƯỜNG NIÊN 1.999.000 VND', 'MIỄN PHÍ NĂM ĐẦU'],
  'INTEREST_RATE': ['LÃI SUẤT 0% 45 NGÀY', 'LÃI SUẤT 2.95%/THÁNG'],
  'CREDIT_LIMIT': ['HẠN MỨC 500 TRIỆU', 'HẠN MỨC TÍN DỤNG CAO'],
  'CURRENCY': ['VND', 'USD'],
  'CUSTOMER_SEGMENT': ['KHÁCH HÀNG VIP', 'SINH VIÊN'],
  'SPENDING_CATEGORY': ['SHOPPING ONLINE', 'DU LỊCH QUỐC TẾ'],
  'MERCHANT_PARTNER': ['TIKI', 'GRAB'],
  'PROMOTION': ['KHUYẾN MÃI TẾT', 'ƯU ĐÃI BLACK FRIDAY'],
  'ELIGIBILITY': ['THU NHẬP TỐI THIỂU 15 TRIỆU', 'TUỔI TỪ 18-65']
}}
################
Output:
{{
  "answer_type_keywords": ["LOCATION"],
  "entities_from_query": ["capital of the United States", "Washington", "New York"]
}}
#############################

-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:

"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]
