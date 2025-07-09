import json

# Read the JSON file
with open('/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/korean_culture_qa_V1.0_train+.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count domain occurrences
domain_counts = {}
for item in data:
    domain = item['input']['domain']
    if domain in domain_counts:
        domain_counts[domain] += 1
    else:
        domain_counts[domain] = 1

# Sort domains by count and get top 10
sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
top_10_domains = sorted_domains[:10]

# Print results
print("Top 10 domains by count:")
for domain, count in top_10_domains:
    print(f"{domain}: {count} questions")
